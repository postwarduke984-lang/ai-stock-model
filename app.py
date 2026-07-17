import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import numpy as np
import datetime
import re
from bs4 import BeautifulSoup

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="AI Stock Valuation Model", layout="wide")

# NEVER hardcode API keys in source. Put GROQ_API_KEY in .streamlit/secrets.toml
# (and add that file to .gitignore) or set it as an environment variable.
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)

# SEC requires a descriptive User-Agent with contact info, or it will
# rate-limit / block you. Replace with your real name + email.
SEC_HEADERS = {
    "User-Agent": "AI Stock Valuation App contact@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}
SEC_DATA_HEADERS = {**SEC_HEADERS, "Host": "data.sec.gov"}


# -----------------------------
# FREE AI (Groq Llama-3 / GPT-OSS)
# -----------------------------
def ai_summary(text, instructions="Summarize this 10-K section and extract growth drivers and risks."):
    if not GROQ_API_KEY:
        return "⚠️ No GROQ_API_KEY found in st.secrets. Add one to .streamlit/secrets.toml to enable AI summaries."

    if not text or len(text.strip()) < 50:
        return "⚠️ No usable text was available to summarize (see warning above)."

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    # Keep the payload within a safe context size
    text = text[:18000]

    data = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {"role": "system", "content": "You are a financial analyst. Be concise and specific."},
            {"role": "user", "content": f"{instructions}\n\n{text}"},
        ],
        "temperature": 0.2,
    }

    try:
        r = requests.post(url, headers=headers, json=data, timeout=60)
        payload = r.json()
    except Exception as e:
        return f"Groq API request failed: {e}"

    if "error" in payload:
        return f"Groq API Error: {payload['error'].get('message', payload['error'])}"

    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return f"Unexpected Groq response format: {payload}"


# -----------------------------
# SEC 10-K FETCHER
# -----------------------------
@st.cache_data(ttl=60 * 60 * 24)  # ticker->CIK map rarely changes; refresh daily
def load_cik_map():
    """Build a ticker -> zero-padded CIK map straight from SEC, no local file needed."""
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    return {entry["ticker"].upper(): str(entry["cik_str"]).zfill(10) for entry in data.values()}


def get_cik_from_ticker(ticker):
    try:
        cik_map = load_cik_map()
    except Exception as e:
        st.warning(f"Could not load SEC ticker list: {e}")
        return None
    return cik_map.get(ticker.upper())


def _clean_filing_html(html):
    """Strip tags/scripts/styles and collapse whitespace so the LLM gets readable text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "table"]):
        # Tables in 10-Ks are mostly financial statement grids that don't
        # summarize well as flattened text; drop them and keep narrative text.
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


@st.cache_data(ttl=60 * 60 * 6)
def get_10k(ticker):
    """
    Returns (clean_text, meta) where meta has filing date / URL for display,
    or (None, error_message) on failure.
    """
    ticker = ticker.upper()
    cik = get_cik_from_ticker(ticker)
    if cik is None:
        return None, f"Ticker '{ticker}' not found in SEC's company list."

    subs_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        r = requests.get(subs_url, headers=SEC_DATA_HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return None, f"SEC submissions lookup failed: {e}"

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    primaries = recent.get("primaryDocument", [])
    dates = recent.get("filingDate", [])

    for i, form in enumerate(forms):
        if form in ("10-K", "10-K/A"):
            accession_nodash = accessions[i].replace("-", "")
            primary_doc = primaries[i]
            doc_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{accession_nodash}/{primary_doc}"
            )
            try:
                doc_resp = requests.get(doc_url, headers=SEC_HEADERS, timeout=30)
                doc_resp.raise_for_status()
            except Exception as e:
                return None, f"Failed to download filing document: {e}"

            clean_text = _clean_filing_html(doc_resp.text)
            meta = {"form": form, "filed": dates[i], "url": doc_url}
            return clean_text, meta

    return None, f"No 10-K or 10-K/A filing found in recent filings for {ticker}."


# -----------------------------
# FINANCIALS
# -----------------------------
@st.cache_data(ttl=60 * 60 * 6)
def get_financials(ticker):
    stock = yf.Ticker(ticker)
    try:
        income = stock.financials
        if income is None or income.empty:
            raise ValueError("No financial data found")
        if "Total Revenue" not in income.index or "Operating Income" not in income.index:
            raise ValueError("Missing required financial fields")

        revenue = income.loc["Total Revenue"].iloc[:4].values[::-1]  # oldest -> newest
        op_income = income.loc["Operating Income"].iloc[:4].values[::-1]
        op_margin = np.divide(
            op_income, revenue, out=np.zeros_like(op_income, dtype=float), where=revenue != 0
        )

        df = pd.DataFrame({"Revenue": revenue, "Operating Margin": op_margin})
        return df, True

    except Exception as e:
        st.warning(f"Financial data unavailable ({e}); using placeholder figures.")
        return pd.DataFrame(
            {"Revenue": [100e9, 110e9, 120e9], "Operating Margin": [0.25, 0.26, 0.27]}
        ), False


def historical_growth_rate(financials, fallback=0.05, cap=0.30):
    """CAGR from the revenue history we have, clamped to something sane."""
    revs = financials["Revenue"].values
    revs = revs[revs > 0]
    if len(revs) < 2:
        return fallback
    periods = len(revs) - 1
    cagr = (revs[-1] / revs[0]) ** (1 / periods) - 1
    if not np.isfinite(cagr):
        return fallback
    return float(np.clip(cagr, -cap, cap))


# -----------------------------
# FORECAST & DCF
# -----------------------------
def build_forecast(financials, growth_rate, years_out=5):
    rev = financials["Revenue"].iloc[-1]
    op_margin = financials["Operating Margin"].iloc[-1]

    years, revenues, op_incomes, fcfs = [], [], [], []
    for i in range(1, years_out + 1):
        rev = rev * (1 + growth_rate)
        op_income = rev * op_margin
        fcf = op_income * 0.7  # rough FCF conversion assumption

        years.append(datetime.datetime.now().year + i)
        revenues.append(rev)
        op_incomes.append(op_income)
        fcfs.append(fcf)

    return pd.DataFrame(
        {"Year": years, "Revenue": revenues, "Operating Income": op_incomes, "FCF": fcfs}
    )


def run_dcf(forecast, discount_rate=0.10, terminal_growth=0.02):
    fcfs = forecast["FCF"].values
    discounted = [fcf / ((1 + discount_rate) ** (i + 1)) for i, fcf in enumerate(fcfs)]

    if discount_rate <= terminal_growth:
        return None  # terminal value math breaks down; caller should handle

    terminal_value = fcfs[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
    terminal_value_discounted = terminal_value / ((1 + discount_rate) ** len(fcfs))

    return sum(discounted) + terminal_value_discounted


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("AI Stock Valuation Model (Free Version)")

with st.sidebar:
    st.header("Assumptions")
    discount_rate = st.slider("Discount rate (WACC)", 0.04, 0.20, 0.10, 0.01)
    terminal_growth = st.slider("Terminal growth rate", 0.0, 0.05, 0.02, 0.005)
    manual_growth = st.checkbox("Override revenue growth assumption")
    manual_growth_rate = st.slider("Revenue growth rate", -0.10, 0.40, 0.05, 0.01) if manual_growth else None

ticker = st.text_input("Enter a stock ticker (AAPL, MSFT, TSLA):").strip()

if ticker:
    st.header("1. Market Data")
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5y")
    if hist.empty:
        st.error(f"No market data found for '{ticker}'. Check the ticker symbol.")
        st.stop()
    st.line_chart(hist["Close"])

    st.header("2. Financial Data")
    financials, financials_ok = get_financials(ticker)
    st.write(financials)

    growth_rate = manual_growth_rate if manual_growth else historical_growth_rate(financials)
    st.caption(f"Using revenue growth assumption: {growth_rate:.1%}"
               + (" (manual override)" if manual_growth else " (derived from historical CAGR)"))

    st.header("3. AI Summary of 10-K")
    tenk_text, meta = get_10k(ticker)
    if tenk_text is None:
        st.warning(meta)  # meta holds the error message in this branch
        summary = "Skipped — no 10-K text was available."
    else:
        st.caption(f"Source: {meta['form']} filed {meta['filed']} — [view filing]({meta['url']})")
        with st.spinner("Summarizing 10-K..."):
            summary = ai_summary(tenk_text)
    st.write(summary)

    st.header("4. 5-Year Forecast")
    forecast = build_forecast(financials, growth_rate)
    st.write(forecast)

    st.header("5. DCF Valuation")
    intrinsic = run_dcf(forecast, discount_rate, terminal_growth)
    shares = stock.info.get("sharesOutstanding")

    if intrinsic is None:
        st.error("Discount rate must be greater than terminal growth rate.")
    elif not shares:
        st.error("Shares outstanding unavailable for this ticker; can't compute per-share value.")
    else:
        price_estimate = intrinsic / shares
        current_price = hist["Close"].iloc[-1]
        upside = (price_estimate / current_price - 1) if current_price else None

        col1, col2, col3 = st.columns(3)
        col1.metric("Estimated Intrinsic Value/Share", f"${price_estimate:,.2f}")
        col2.metric("Current Price", f"${current_price:,.2f}")
        if upside is not None:
            col3.metric("Implied Upside/Downside", f"{upside:+.1%}")

        st.subheader("Sensitivity: intrinsic value/share by discount rate")
        rates = [discount_rate + d for d in (-0.02, -0.01, 0, 0.01, 0.02) if discount_rate + d > terminal_growth]
        sens_rows = []
        for r in rates:
            iv = run_dcf(forecast, r, terminal_growth)
            sens_rows.append({"Discount Rate": f"{r:.1%}", "Value/Share": f"${iv / shares:,.2f}"})
        st.table(pd.DataFrame(sens_rows))

    st.header("6. AI Analyst Report")
    if intrinsic is not None and shares:
        with st.spinner("Generating analyst report..."):
            report = ai_summary(
                f"Forecast:\n{forecast.to_string()}\n\nDCF intrinsic value per share: ${price_estimate:,.2f}\n"
                f"Current market price: ${current_price:,.2f}",
                instructions="Write a brief analyst-style verdict (buy/hold/sell reasoning) based on this DCF output.",
            )
        st.write(report)
    else:
        st.info("Analyst report skipped — DCF valuation was not computable above.")
