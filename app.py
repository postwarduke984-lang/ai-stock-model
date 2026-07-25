import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import numpy as np
import datetime
import re
import time
from bs4 import BeautifulSoup

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="AI Stock Valuation Model", layout="wide")

# NEVER hardcode API keys in source. Put GROQ_API_KEY in .streamlit/secrets.toml
# (and add that file to .gitignore) or set it as an environment variable.
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)

# SEC REQUIRES a descriptive User-Agent with a real name + working email.
# Generic/placeholder agents get rate-limited or blocked much more aggressively.
# >>> Replace the line below with your actual name and email before deploying. <<<
SEC_HEADERS = {
    "User-Agent": "AI Stock Valuation App - Replace With Your Name your_email@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}
SEC_DATA_HEADERS = {**SEC_HEADERS, "Host": "data.sec.gov"}


# -----------------------------
# FREE AI (Groq Llama-3 / GPT-OSS)
# -----------------------------
def ai_summary(text, instructions, temperature=0.2):
    if not GROQ_API_KEY:
        return "⚠️ No GROQ_API_KEY found in st.secrets. Add one to .streamlit/secrets.toml to enable AI summaries."

    if not text or len(text.strip()) < 50:
        return "⚠️ No usable text was available to summarize."

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {"role": "system", "content": "You are a precise financial analyst. Follow the requested output format exactly, every time, regardless of company."},
            {"role": "user", "content": f"{instructions}\n\n{text}"},
        ],
        "temperature": temperature,
    }

    try:
        r = requests.post(url, headers=headers, json=data, timeout=90)
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
# TICKER NORMALIZATION
# -----------------------------
def ticker_variants(ticker):
    """SEC and Yahoo Finance don't always agree on share-class formatting
    (e.g. BRK.B vs BRK-B). Try the raw ticker plus common variants."""
    t = ticker.upper().strip()
    variants = [t]
    if "." in t:
        variants.append(t.replace(".", "-"))
    if "-" in t:
        variants.append(t.replace("-", "."))
    # de-dupe, preserve order
    seen = set()
    return [v for v in variants if not (v in seen or seen.add(v))]


# -----------------------------
# SEC 10-K FETCHER
# -----------------------------
def _get_with_retry(url, headers, max_retries=3, timeout=30):
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                last_err = f"HTTP {r.status_code}"
                time.sleep(1.5 * (attempt + 1))  # backoff: 1.5s, 3s, 4.5s
                continue
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed after {max_retries} attempts: {last_err}")


@st.cache_data(ttl=60 * 60 * 24)
def load_cik_map():
    """Build a ticker -> zero-padded CIK map straight from SEC, no local file needed."""
    r = _get_with_retry("https://www.sec.gov/files/company_tickers.json", SEC_HEADERS)
    data = r.json()
    return {entry["ticker"].upper(): str(entry["cik_str"]).zfill(10) for entry in data.values()}


def get_cik_from_ticker(ticker):
    try:
        cik_map = load_cik_map()
    except Exception as e:
        st.warning(f"Could not load SEC ticker list: {e}")
        return None, None

    for variant in ticker_variants(ticker):
        if variant in cik_map:
            return cik_map[variant], variant
    return None, None


def _clean_filing_html(html):
    """Strip tags/scripts/styles and collapse whitespace so downstream parsing
    works on readable text instead of markup."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# Canonical 10-K item headers we care about, in filing order.
ITEM_PATTERNS = [
    ("business", r"item\s+1\.?\s+business"),
    ("risk_factors", r"item\s+1a\.?\s+risk\s+factors"),
    ("properties", r"item\s+2\.?\s+properties"),
    ("legal_proceedings", r"item\s+3\.?\s+legal\s+proceedings"),
    ("mdna", r"item\s+7\.?\s+management.?s\s+discussion"),
    ("market_risk", r"item\s+7a\.?\s+quantitative"),
    ("financial_statements", r"item\s+8\.?\s+financial\s+statements"),
]


def extract_10k_sections(text, max_chars_per_section=4500):
    """
    Split a cleaned 10-K into its major Items by locating header text.
    Returns a dict of section_name -> excerpt. Falls back gracefully if a
    section can't be located (structure varies company to company).
    """
    lower = text.lower()
    matches = []
    for name, pattern in ITEM_PATTERNS:
        m = re.search(pattern, lower)
        if m:
            matches.append((m.start(), name))
    matches.sort()

    sections = {}
    for i, (start, name) in enumerate(matches):
        end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        excerpt = text[start:end].strip()
        sections[name] = excerpt[:max_chars_per_section]

    return sections


@st.cache_data(ttl=60 * 60 * 6)
def get_10k(ticker):
    """
    Returns (sections_dict, meta) on success, or (None, error_message) on failure.
    sections_dict maps Item name -> text excerpt (see ITEM_PATTERNS).
    """
    cik, matched_ticker = get_cik_from_ticker(ticker)
    if cik is None:
        return None, f"Ticker '{ticker.upper()}' not found in SEC's company list (tried: {', '.join(ticker_variants(ticker))})."

    subs_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        r = _get_with_retry(subs_url, SEC_DATA_HEADERS)
        data = r.json()
    except Exception as e:
        return None, f"SEC submissions lookup failed: {e}"

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    primaries = recent.get("primaryDocument", [])
    dates = recent.get("filingDate", [])

    filing_index = None
    for i, form in enumerate(forms):
        if form in ("10-K", "10-K/A"):
            filing_index = i
            break

    # Some companies' "recent" window doesn't include a 10-K (e.g. very
    # frequent other filings). Check the older filings index as a fallback.
    if filing_index is None:
        older_files = data.get("filings", {}).get("files", [])
        for f in older_files:
            try:
                older = _get_with_retry(
                    f"https://data.sec.gov/submissions/{f['name']}", SEC_DATA_HEADERS
                ).json()
            except Exception:
                continue
            o_forms = older.get("form", [])
            for i, form in enumerate(o_forms):
                if form in ("10-K", "10-K/A"):
                    forms, accessions, primaries, dates = (
                        o_forms,
                        older.get("accessionNumber", []),
                        older.get("primaryDocument", []),
                        older.get("filingDate", []),
                    )
                    filing_index = i
                    break
            if filing_index is not None:
                break

    if filing_index is None:
        return None, f"No 10-K or 10-K/A filing found for {matched_ticker} (CIK {cik})."

    accession_nodash = accessions[filing_index].replace("-", "")
    primary_doc = primaries[filing_index]
    doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash}/{primary_doc}"

    try:
        doc_resp = _get_with_retry(doc_url, SEC_HEADERS, timeout=45)
    except Exception as e:
        return None, f"Failed to download filing document: {e}"

    clean_text = _clean_filing_html(doc_resp.text)
    sections = extract_10k_sections(clean_text)
    meta = {
        "form": forms[filing_index],
        "filed": dates[filing_index],
        "url": doc_url,
        "matched_ticker": matched_ticker,
        "full_text": clean_text,  # kept for fallback if section parsing misses everything
    }
    return sections, meta


# -----------------------------
# DETERMINISTIC FINANCIAL METRICS (no LLM involved — pulled straight from data)
# -----------------------------
@st.cache_data(ttl=60 * 60 * 6)
def get_key_metrics(ticker):
    """
    Builds a fixed-schema table of core financials so the output shape is
    identical for every ticker, with 'N/A' where a company doesn't report
    a given line item. Returns (metrics_dict, ok_flag).
    """
    stock = yf.Ticker(ticker)
    fields = [
        "Revenue", "Net Income", "Gross Profit", "Operating Income",
        "Operating Margin", "Net Margin", "EPS (Diluted)",
        "Total Debt", "Cash & Equivalents", "Free Cash Flow",
    ]
    metrics = {f: "N/A" for f in fields}
    ok = True

    try:
        income = stock.financials
        cashflow = stock.cashflow
        balance = stock.balance_sheet

        if income is not None and not income.empty:
            revenue = income.loc["Total Revenue"].iloc[0] if "Total Revenue" in income.index else None
            net_income = income.loc["Net Income"].iloc[0] if "Net Income" in income.index else None
            gross_profit = income.loc["Gross Profit"].iloc[0] if "Gross Profit" in income.index else None
            op_income = income.loc["Operating Income"].iloc[0] if "Operating Income" in income.index else None

            if revenue:
                metrics["Revenue"] = f"${revenue:,.0f}"
            if net_income is not None:
                metrics["Net Income"] = f"${net_income:,.0f}"
            if gross_profit is not None:
                metrics["Gross Profit"] = f"${gross_profit:,.0f}"
            if op_income is not None:
                metrics["Operating Income"] = f"${op_income:,.0f}"
            if revenue and op_income is not None:
                metrics["Operating Margin"] = f"{op_income / revenue:.1%}"
            if revenue and net_income is not None:
                metrics["Net Margin"] = f"{net_income / revenue:.1%}"

        eps = stock.info.get("trailingEps")
        if eps:
            metrics["EPS (Diluted)"] = f"${eps:.2f}"

        if balance is not None and not balance.empty:
            debt_fields = ["Total Debt", "Long Term Debt"]
            for f in debt_fields:
                if f in balance.index:
                    metrics["Total Debt"] = f"${balance.loc[f].iloc[0]:,.0f}"
                    break
            cash_fields = ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]
            for f in cash_fields:
                if f in balance.index:
                    metrics["Cash & Equivalents"] = f"${balance.loc[f].iloc[0]:,.0f}"
                    break

        if cashflow is not None and not cashflow.empty and "Free Cash Flow" in cashflow.index:
            metrics["Free Cash Flow"] = f"${cashflow.loc['Free Cash Flow'].iloc[0]:,.0f}"

    except Exception as e:
        ok = False
        st.warning(f"Some financial metrics unavailable: {e}")

    return metrics, ok


@st.cache_data(ttl=60 * 60 * 6)
def get_financials(ticker):
    """Historical revenue/margin series used for the forecast model."""
    stock = yf.Ticker(ticker)
    try:
        income = stock.financials
        if income is None or income.empty:
            raise ValueError("No financial data found")
        if "Total Revenue" not in income.index or "Operating Income" not in income.index:
            raise ValueError("Missing required financial fields")

        revenue = income.loc["Total Revenue"].iloc[:4].values[::-1]
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
        fcf = op_income * 0.7

        years.append(datetime.datetime.now().year + i)
        revenues.append(rev)
        op_incomes.append(op_income)
        fcfs.append(fcf)

    return pd.DataFrame(
        {"Year": years, "Revenue": revenues, "Operating Income": op_incomes, "FCF": fcfs}
    )


def run_dcf(forecast, discount_rate=0.10, terminal_growth=0.02):
    """Total firm intrinsic value (no buyback adjustment) — kept for reference/sensitivity."""
    fcfs = forecast["FCF"].values
    discounted = [fcf / ((1 + discount_rate) ** (i + 1)) for i, fcf in enumerate(fcfs)]

    if discount_rate <= terminal_growth:
        return None

    terminal_value = fcfs[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
    terminal_value_discounted = terminal_value / ((1 + discount_rate) ** len(fcfs))

    return sum(discounted) + terminal_value_discounted


@st.cache_data(ttl=60 * 60 * 6)
def estimate_buyback_rate(ticker, cap=0.08):
    """
    Rough annual share-count reduction rate, estimated from last year's cash
    spent on repurchases divided by market cap. This is a heuristic (actual
    buyback price varies through the year) — good enough to anchor a default,
    with a manual override always available in the UI.
    Returns (rate, ok_flag).
    """
    try:
        stock = yf.Ticker(ticker)
        cashflow = stock.cashflow
        market_cap = stock.info.get("marketCap")

        if cashflow is None or cashflow.empty or not market_cap:
            return 0.0, False

        repurchase_field = None
        for candidate in ("Repurchase Of Capital Stock", "Repurchase Of Common Stock"):
            if candidate in cashflow.index:
                repurchase_field = candidate
                break
        if repurchase_field is None:
            return 0.0, False

        buyback_cash = abs(cashflow.loc[repurchase_field].iloc[0])
        if not buyback_cash or not np.isfinite(buyback_cash):
            return 0.0, False

        rate = buyback_cash / market_cap
        return float(np.clip(rate, 0.0, cap)), True

    except Exception:
        return 0.0, False


def run_dcf_per_share(forecast, shares0, buyback_rate, discount_rate=0.10, terminal_growth=0.02):
    """
    DCF computed directly on a per-share basis, with the share count shrinking
    each year at `buyback_rate`. This credits the value created when the same
    total cash flow gets divided across fewer future shares — the effect a
    plain firm-level DCF / today's-share-count misses entirely.
    Returns (price_per_share, shares_by_year) or (None, None) if terminal math
    is invalid.
    """
    if discount_rate <= terminal_growth:
        return None, None

    fcfs = forecast["FCF"].values
    n = len(fcfs)
    shares_by_year = [shares0 * ((1 - buyback_rate) ** (i + 1)) for i in range(n)]

    per_share_fcf = [fcf / shares for fcf, shares in zip(fcfs, shares_by_year)]
    discounted = [psf / ((1 + discount_rate) ** (i + 1)) for i, psf in enumerate(per_share_fcf)]

    terminal_psf = per_share_fcf[-1] * (1 + terminal_growth)
    terminal_value_per_share = terminal_psf / (discount_rate - terminal_growth)
    terminal_discounted = terminal_value_per_share / ((1 + discount_rate) ** n)

    price = sum(discounted) + terminal_discounted
    return price, shares_by_year


# -----------------------------
# FIXED-TEMPLATE 10-K NARRATIVE
# -----------------------------
QUALITATIVE_TEMPLATE = """
Using ONLY the filing excerpts below, fill in this exact markdown table. Keep the
row order and labels EXACTLY as shown for every company, no matter what the filing
covers. If the filing excerpts don't address a row, write "Not disclosed in excerpt"
— do not omit rows and do not add extra rows.

| Item | Summary |
|---|---|
| Core business description | |
| Primary products/segments | |
| Geographic footprint | |
| Top 3 risk factors | |
| Legal/regulatory matters | |
| Management's outlook (from MD&A) | |
| Notable concentration exposures (customers/suppliers) | |
| Capital allocation notes (buybacks/dividends/debt, if mentioned) | |

Filing excerpts:
"""


def build_qualitative_summary(sections, full_text_fallback):
    if not sections:
        # No Item headers were locatable — fall back to the first chunk of
        # cleaned text so the model still has *something* to work with.
        combined = full_text_fallback[:12000]
    else:
        parts = []
        for name, _ in ITEM_PATTERNS:
            if name in sections:
                parts.append(f"--- {name.upper()} ---\n{sections[name]}")
        combined = "\n\n".join(parts)[:16000]

    return ai_summary(combined, QUALITATIVE_TEMPLATE)


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

    st.divider()
    st.header("Share Buybacks")
    model_buybacks = st.checkbox("Model share buybacks in the DCF", value=True)
    manual_buyback = st.checkbox("Override buyback rate")
    manual_buyback_pct = (
        st.slider("Annual share count reduction (%)", 0.0, 8.0, 2.0, 0.5)
        if manual_buyback else None
    )
    manual_buyback_rate = manual_buyback_pct / 100 if manual_buyback_pct is not None else None

ticker = st.text_input("Enter a stock ticker (AAPL, MSFT, TSLA, BRK.B):").strip()

if ticker:
    st.header("1. Market Data")
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5y")
    if hist.empty:
        st.error(f"No market data found for '{ticker}'. Check the ticker symbol.")
        st.stop()
    st.line_chart(hist["Close"])

    st.header("2. Key Financial Metrics")
    st.caption("Pulled directly from reported financial data — same fields shown for every company, regardless of what the 10-K narrative covers.")
    metrics, metrics_ok = get_key_metrics(ticker)
    st.table(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))

    financials, financials_ok = get_financials(ticker)
    growth_rate = manual_growth_rate if manual_growth else historical_growth_rate(financials)
    st.caption(f"Forecast revenue growth assumption: {growth_rate:.1%}"
               + (" (manual override)" if manual_growth else " (derived from historical CAGR)"))

    st.header("3. 10-K Qualitative Summary")
    sections, meta = get_10k(ticker)
    if sections is None:
        st.warning(meta)
        summary = None
    else:
        st.caption(f"Source: {meta['form']} filed {meta['filed']} (matched as {meta['matched_ticker']}) — [view filing]({meta['url']})")
        found = [name for name, _ in ITEM_PATTERNS if name in sections]
        if not found:
            st.info("Couldn't locate standard Item headers in this filing; summarizing from the start of the document instead.")
        with st.spinner("Summarizing 10-K..."):
            summary = build_qualitative_summary(sections, meta["full_text"])
        st.markdown(summary)

    st.header("4. 5-Year Forecast")
    forecast = build_forecast(financials, growth_rate)
    st.write(forecast)

    st.header("5. DCF Valuation")

    shares = stock.info.get("sharesOutstanding")

    if model_buybacks:
        est_rate, buyback_ok = estimate_buyback_rate(ticker)
        buyback_rate = manual_buyback_rate if manual_buyback else est_rate
    else:
        buyback_rate, buyback_ok = 0.0, True

    st.caption(
        "Simplified model: holds today's operating margin flat for 5 years, assumes free cash flow "
        "is 70% of operating income, and applies a single terminal growth rate. Real valuations account "
        "for margin trends and capex cycles beyond this — treat this as a rough anchor, not a price target."
    )

    if model_buybacks:
        source_note = "manual override" if manual_buyback else (
            "estimated from last year's buyback spend ÷ market cap" if buyback_ok else "no buyback data found — defaulted to 0%"
        )
        st.caption(f"Buyback assumption: share count shrinks {buyback_rate:.1%}/year ({source_note}).")
    else:
        st.caption("Buyback modeling is off — share count held constant, as in a plain textbook DCF.")

    if not shares:
        st.error("Shares outstanding unavailable for this ticker; can't compute per-share value.")
        price_estimate, shares_by_year = None, None
    else:
        price_estimate, shares_by_year = run_dcf_per_share(
            forecast, shares, buyback_rate, discount_rate, terminal_growth
        )

    if price_estimate is None and shares:
        st.error("Discount rate must be greater than terminal growth rate.")

    if price_estimate is not None:
        # hist["Close"].iloc[-1] can be NaN if the most recent trading day's
        # data is incomplete (common right after market open, or on gappy
        # feeds). Fall back to the last non-NaN close, then to live quote
        # fields from stock.info if the whole history is somehow empty.
        valid_closes = hist["Close"].dropna()
        if not valid_closes.empty:
            current_price = valid_closes.iloc[-1]
        else:
            current_price = stock.info.get("currentPrice") or stock.info.get("regularMarketPrice")

        upside = (
            (price_estimate / current_price - 1)
            if current_price and np.isfinite(current_price)
            else None
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Estimated Intrinsic Value/Share", f"${price_estimate:,.2f}")
        col2.metric(
            "Current Price",
            f"${current_price:,.2f}" if current_price and np.isfinite(current_price) else "Unavailable",
        )
        if upside is not None:
            col3.metric("Implied Upside/Downside", f"{upside:+.1%}")

        if model_buybacks and buyback_rate > 0 and shares_by_year:
            with st.expander("Projected share count under the buyback assumption"):
                share_table = pd.DataFrame({
                    "Year": forecast["Year"].values,
                    "Projected Shares Outstanding": [f"{s:,.0f}" for s in shares_by_year],
                })
                st.table(share_table)

        st.subheader("Sensitivity: intrinsic value/share by discount rate")
        rates = [discount_rate + d for d in (-0.02, -0.01, 0, 0.01, 0.02) if discount_rate + d > terminal_growth]
        sens_rows = []
        for r in rates:
            iv, _ = run_dcf_per_share(forecast, shares, buyback_rate, r, terminal_growth)
            sens_rows.append({"Discount Rate": f"{r:.1%}", "Value/Share": f"${iv:,.2f}"})
        st.table(pd.DataFrame(sens_rows))

    st.header("6. AI Analyst Report")
    if price_estimate is not None:
        with st.spinner("Generating analyst report..."):
            report = ai_summary(
                f"Forecast:\n{forecast.to_string()}\n\nDCF intrinsic value per share: ${price_estimate:,.2f}\n"
                f"Current market price: ${current_price:,.2f}",
                instructions="Write a brief analyst-style verdict (buy/hold/sell reasoning) based on this DCF output, in 3-5 sentences.",
            )
        st.write(report)
    else:
        st.info("Analyst report skipped — DCF valuation was not computable above.")
