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
# UI STYLING
# -----------------------------
def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Section headers: more breathing room, subtle divider */
        h1 { font-weight: 800 !important; letter-spacing: -0.02em; }
        h2 {
            font-weight: 700 !important;
            letter-spacing: -0.01em;
            margin-top: 2.2rem !important;
            padding-bottom: 0.4rem;
            border-bottom: 1px solid rgba(128,128,128,0.2);
        }
        h3 { font-weight: 600 !important; }

        /* st.metric cards: bigger, bolder numbers with small uppercase labels,
           similar to the big-stat tiles on etfrc.com */
        div[data-testid="stMetric"] {
            background: rgba(128,128,128,0.06);
            border: 1px solid rgba(128,128,128,0.15);
            border-radius: 12px;
            padding: 1rem 1.2rem 0.8rem 1.2rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 2.1rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.02em;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            opacity: 0.65;
        }

        /* Verdict banner */
        .verdict-banner {
            border-radius: 14px;
            padding: 1.4rem 1.6rem;
            margin: 0.6rem 0 1.2rem 0;
            text-align: center;
        }
        .verdict-label {
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            opacity: 0.75;
            margin-bottom: 0.2rem;
        }
        .verdict-value {
            font-size: 3rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            line-height: 1.1;
        }

        /* Tables: tighter, cleaner */
        div[data-testid="stTable"] table { font-size: 0.9rem; }
        </style>
    """, unsafe_allow_html=True)


VERDICT_COLORS = {
    "buy": {"bg": "rgba(34, 197, 94, 0.12)", "text": "#16a34a"},
    "hold": {"bg": "rgba(234, 179, 8, 0.12)", "text": "#ca8a04"},
    "sell": {"bg": "rgba(239, 68, 68, 0.12)", "text": "#dc2626"},
}


def render_verdict_banner(report_text):
    """
    Pulls the Buy/Hold/Sell call out of the report's '**Verdict: X**' line and
    renders it as a large color-coded banner, then returns the report with
    that line stripped (so it isn't shown twice).
    """
    match = re.search(r"\*\*Verdict:\s*(Buy|Hold|Sell)\*\*", report_text, re.IGNORECASE)
    if not match:
        return report_text  # couldn't parse a verdict — just show the report as-is

    verdict = match.group(1).capitalize()
    colors = VERDICT_COLORS.get(verdict.lower(), {"bg": "rgba(128,128,128,0.1)", "text": "inherit"})

    st.markdown(f"""
        <div class="verdict-banner" style="background:{colors['bg']};">
            <div class="verdict-label">Analyst Verdict</div>
            <div class="verdict-value" style="color:{colors['text']};">{verdict}</div>
        </div>
    """, unsafe_allow_html=True)

    return report_text[:match.start()] + report_text[match.end():]


def format_large_currency(value):
    """Compact currency formatting: $391.2B, $2.4M, $850.0K, or plain $ for small values."""
    if value is None or not np.isfinite(value):
        return "N/A"
    sign = "-" if value < 0 else ""
    abs_val = abs(value)
    if abs_val >= 1e12:
        return f"{sign}${abs_val / 1e12:.2f}T"
    if abs_val >= 1e9:
        return f"{sign}${abs_val / 1e9:.2f}B"
    if abs_val >= 1e6:
        return f"{sign}${abs_val / 1e6:.2f}M"
    if abs_val >= 1e3:
        return f"{sign}${abs_val / 1e3:.1f}K"
    return f"{sign}${abs_val:,.2f}"


def format_large_number(value):
    """Same compact scaling as format_large_currency but without a $ sign — for share counts etc."""
    if value is None or not np.isfinite(value):
        return "N/A"
    sign = "-" if value < 0 else ""
    abs_val = abs(value)
    if abs_val >= 1e12:
        return f"{sign}{abs_val / 1e12:.2f}T"
    if abs_val >= 1e9:
        return f"{sign}{abs_val / 1e9:.2f}B"
    if abs_val >= 1e6:
        return f"{sign}{abs_val / 1e6:.2f}M"
    if abs_val >= 1e3:
        return f"{sign}{abs_val / 1e3:.1f}K"
    return f"{sign}{abs_val:,.0f}"


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
                metrics["Revenue"] = format_large_currency(revenue)
            if net_income is not None:
                metrics["Net Income"] = format_large_currency(net_income)
            if gross_profit is not None:
                metrics["Gross Profit"] = format_large_currency(gross_profit)
            if op_income is not None:
                metrics["Operating Income"] = format_large_currency(op_income)
            if revenue and op_income is not None:
                metrics["Operating Margin"] = f"{op_income / revenue:.1%}"
            if revenue and net_income is not None:
                metrics["Net Margin"] = f"{net_income / revenue:.1%}"

        eps = stock.info.get("trailingEps")
        if eps:
            metrics["EPS (Diluted)"] = f"${eps:.2f}"  # per-share value — kept precise, not abbreviated

        if balance is not None and not balance.empty:
            debt_fields = ["Total Debt", "Long Term Debt"]
            for f in debt_fields:
                if f in balance.index:
                    metrics["Total Debt"] = format_large_currency(balance.loc[f].iloc[0])
                    break
            cash_fields = ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]
            for f in cash_fields:
                if f in balance.index:
                    metrics["Cash & Equivalents"] = format_large_currency(balance.loc[f].iloc[0])
                    break

        if cashflow is not None and not cashflow.empty and "Free Cash Flow" in cashflow.index:
            metrics["Free Cash Flow"] = format_large_currency(cashflow.loc["Free Cash Flow"].iloc[0])

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


@st.cache_data(ttl=60 * 60 * 6)
def estimate_growth_rate(ticker, _financials_tuple):
    """
    Blends backward-looking historical CAGR with Yahoo Finance's forward-looking
    analyst growth estimates ('research reports' consensus, effectively) so the
    forecast isn't purely extrapolating a stagnant or unusually hot recent stretch.
    _financials_tuple is a hashable (tuple) version of the revenue series, needed
    because st.cache_data can't hash a DataFrame directly.
    Returns (blended_rate, historical_rate, analyst_rate_or_None).
    """
    revs = np.array(_financials_tuple)
    revs = revs[revs > 0]
    if len(revs) >= 2:
        periods = len(revs) - 1
        hist_rate = (revs[-1] / revs[0]) ** (1 / periods) - 1
        hist_rate = float(np.clip(hist_rate, -0.30, 0.30)) if np.isfinite(hist_rate) else 0.05
    else:
        hist_rate = 0.05

    analyst_rate = None
    try:
        info = yf.Ticker(ticker).info or {}
        # revenueGrowth is Yahoo's most recent yoy figure, informed by analyst
        # models — a reasonable stand-in for "what research reports expect."
        candidate = info.get("revenueGrowth")
        if candidate is not None and np.isfinite(candidate):
            analyst_rate = float(np.clip(candidate, -0.30, 0.50))
    except Exception:
        pass

    if analyst_rate is not None:
        blended = 0.5 * hist_rate + 0.5 * analyst_rate
    else:
        blended = hist_rate

    return float(np.clip(blended, -0.20, 0.40)), hist_rate, analyst_rate


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


@st.cache_data(ttl=60 * 60 * 24)
def get_risk_free_rate(default=0.04):
    """10-year US Treasury yield (^TNX) as a risk-free rate proxy for CAPM."""
    try:
        hist = yf.Ticker("^TNX").history(period="5d")
        if hist.empty:
            return default, False
        value = float(hist["Close"].dropna().iloc[-1])
        # Yahoo occasionally reports ^TNX scaled by 10 depending on feed version.
        rate = value / 100 if value < 1 else (value / 10 / 100 if value > 30 else value / 100)
        return float(np.clip(rate, 0.01, 0.10)), True
    except Exception:
        return default, False


@st.cache_data(ttl=60 * 60 * 6)
def estimate_wacc(ticker):
    """
    CAPM-based, company-specific discount rate — no manual guessing required.
    cost_of_equity = risk_free_rate + beta * equity_risk_premium
    cost_of_debt   = interest expense / total debt (after-tax), with a spread-based fallback
    WACC = weight_equity * cost_of_equity + weight_debt * cost_of_debt_after_tax
    Returns (wacc, details_dict). Falls back to a generic 9% if data is too sparse.
    """
    ERP = 0.045  # long-run US equity risk premium assumption (commonly cited ~4-5%)
    risk_free, rf_ok = get_risk_free_rate()

    details = {
        "risk_free_rate": risk_free,
        "equity_risk_premium": ERP,
        "beta": None,
        "cost_of_equity": None,
        "cost_of_debt": None,
        "tax_rate": 0.21,
        "total_debt": 0.0,
        "cash": 0.0,
        "weight_equity": 1.0,
        "weight_debt": 0.0,
        "ok": True,
    }

    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        income = stock.financials
        balance = stock.balance_sheet

        beta = info.get("beta") or 1.0
        details["beta"] = beta
        cost_of_equity = risk_free + beta * ERP
        details["cost_of_equity"] = cost_of_equity

        total_debt, cash = 0.0, 0.0
        if balance is not None and not balance.empty:
            for f in ("Total Debt", "Long Term Debt"):
                if f in balance.index:
                    total_debt = float(balance.loc[f].iloc[0])
                    break
            for f in ("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"):
                if f in balance.index:
                    cash = float(balance.loc[f].iloc[0])
                    break
        details["total_debt"] = total_debt
        details["cash"] = cash

        tax_rate = 0.21
        if income is not None and not income.empty:
            if "Tax Provision" in income.index and "Pretax Income" in income.index:
                pretax = income.loc["Pretax Income"].iloc[0]
                tax = income.loc["Tax Provision"].iloc[0]
                if pretax and np.isfinite(pretax / pretax if pretax else np.nan):
                    if pretax != 0:
                        tax_rate = float(np.clip(tax / pretax, 0.0, 0.35))
        details["tax_rate"] = tax_rate

        interest_expense = None
        if income is not None and not income.empty:
            for f in ("Interest Expense", "Interest Expense Non Operating"):
                if f in income.index:
                    interest_expense = abs(float(income.loc[f].iloc[0]))
                    break

        if interest_expense and total_debt > 0:
            cost_of_debt = interest_expense / total_debt
        else:
            cost_of_debt = risk_free + 0.015  # generic credit spread fallback
        details["cost_of_debt"] = cost_of_debt

        market_cap = info.get("marketCap") or 0.0
        total_capital = market_cap + total_debt
        weight_equity = market_cap / total_capital if total_capital > 0 else 1.0
        weight_debt = 1 - weight_equity
        details["weight_equity"] = weight_equity
        details["weight_debt"] = weight_debt

        wacc = weight_equity * cost_of_equity + weight_debt * cost_of_debt * (1 - tax_rate)
        wacc = float(np.clip(wacc, 0.04, 0.20))
        return wacc, details

    except Exception:
        details["ok"] = False
        return 0.09, details


def estimate_terminal_growth(risk_free_rate):
    """
    Anchors terminal growth to the macro environment (roughly: long-run real
    growth ≈ risk-free rate minus an inflation/real-rate spread) rather than
    an arbitrary fixed 2%. Clipped to a conservative 1-3% band — no company
    should be assumed to out-grow the broader economy forever.
    """
    return float(np.clip(risk_free_rate - 0.02, 0.01, 0.03))


def run_dcf_standard(forecast, shares0, net_cash, discount_rate=0.10, terminal_growth=0.02):
    """
    Standard textbook DCF: discount firm-level unlevered FCF at WACC, add a
    Gordon-growth terminal value, sum to Enterprise Value, then bridge to
    Equity Value by adding net cash (or subtracting net debt if negative),
    and divide by TODAY's share count. This is the conventional structure —
    no share-count projection tricks, no double-counting of buyback value
    (that value is already embedded in the FCF the company generates).
    Returns (price_per_share, enterprise_value, equity_value) or (None, None, None).
    """
    if discount_rate <= terminal_growth:
        return None, None, None

    fcfs = forecast["FCF"].values
    n = len(fcfs)
    discounted = [fcf / ((1 + discount_rate) ** (i + 1)) for i, fcf in enumerate(fcfs)]

    terminal_value = fcfs[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
    terminal_value_discounted = terminal_value / ((1 + discount_rate) ** n)

    enterprise_value = sum(discounted) + terminal_value_discounted
    equity_value = enterprise_value + net_cash  # net_cash negative = net debt, naturally subtracts

    price_per_share = equity_value / shares0
    return price_per_share, enterprise_value, equity_value


@st.cache_data(ttl=60 * 60 * 6)
def estimate_buyback_rate(ticker, cap=0.08, years=3):
    """
    Annual share-count reduction rate, estimated from the average of up to the
    last `years` of cash spent on repurchases divided by current market cap.
    Averaging smooths out lumpy, one-off repurchase spikes. This informs the
    illustrative buyback scenario below — it does not feed the standard DCF
    price (which already captures the value of all future cash use, buybacks
    included, via the Enterprise->Equity bridge).
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

        series = cashflow.loc[repurchase_field].iloc[:years].abs()
        series = series[np.isfinite(series)]
        if series.empty:
            return 0.0, False

        avg_buyback_cash = series.mean()
        rate = avg_buyback_cash / market_cap
        return float(np.clip(rate, 0.0, cap)), True

    except Exception:
        return 0.0, False


def run_dcf_buyback_illustrative(forecast, shares0, buyback_rate, discount_rate=0.10, terminal_growth=0.02):
    """
    Secondary, illustrative view: shows how per-share value compounds if the
    company's cash generation is channeled specifically into shrinking the
    share count at `buyback_rate`/year, rather than the standard assumption
    that all FCF accrues evenly to today's shareholders. Useful as a sense of
    upside from continued aggressive buybacks — not the headline number.
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


@st.cache_data(ttl=60 * 60 * 3)
def get_market_context(ticker):
    """
    Pulls analyst consensus, valuation multiples, and recent news headlines
    from Yahoo Finance so the analyst report can weigh more than just the
    DCF number. Returns a dict; any field we can't find is left as None
    so downstream formatting can show 'N/A' consistently.
    """
    stock = yf.Ticker(ticker)
    info = stock.info or {}

    context = {
        "target_mean_price": info.get("targetMeanPrice"),
        "target_high_price": info.get("targetHighPrice"),
        "target_low_price": info.get("targetLowPrice"),
        "num_analysts": info.get("numberOfAnalystOpinions"),
        "recommendation_key": info.get("recommendationKey"),  # e.g. 'buy', 'hold', 'sell'
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "peg_ratio": info.get("trailingPegRatio"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        "beta": info.get("beta"),
        "news": [],
    }

    # Momentum: % change over the last ~1 and ~3 months of trading history.
    try:
        hist = stock.history(period="6mo")
        closes = hist["Close"].dropna()
        if len(closes) > 21:
            context["momentum_1mo"] = float(closes.iloc[-1] / closes.iloc[-21] - 1)
        if len(closes) > 63:
            context["momentum_3mo"] = float(closes.iloc[-1] / closes.iloc[-63] - 1)
    except Exception:
        pass

    # Recent news headlines — titles/publishers only (short factual metadata,
    # not article text), just enough to give the model a sense of current
    # narrative and sentiment without reproducing any copyrighted content.
    try:
        raw_news = stock.news or []
        for item in raw_news[:6]:
            content = item.get("content", item)  # yfinance news schema varies by version
            title = content.get("title") if isinstance(content, dict) else None
            publisher = None
            if isinstance(content, dict):
                provider = content.get("provider")
                publisher = provider.get("displayName") if isinstance(provider, dict) else None
            if title:
                context["news"].append({"title": title, "publisher": publisher or "Unknown source"})
    except Exception:
        pass

    return context


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
# FIXED-TEMPLATE ANALYST REPORT (buy/hold/sell weighing multiple signals)
# -----------------------------
ANALYST_TEMPLATE = """
You are writing a balanced equity research note. Use ALL the signals provided below —
not just the DCF valuation — to reach a verdict. The DCF is one input among several;
a stock trading above its DCF value can still reasonably be a Hold or even a Buy if
analyst consensus, momentum, and qualitative factors support it, and vice versa.
Weigh disagreements between signals explicitly rather than defaulting to whichever
signal is most extreme.

Fill in this exact template, keeping the section headers identical for every company.
If a data point is marked N/A or missing below, say so plainly rather than guessing:

**Verdict: [Buy / Hold / Sell]**

**DCF valuation vs. market price:** (state the gap and how much weight it deserves)

**Analyst consensus:** (target price range, number of analysts, consensus rating —
note if this agrees or disagrees with the DCF)

**Valuation multiples:** (trailing/forward P/E, PEG if available — cheap, fair, or
expensive relative to what the multiple implies about growth expectations)

**Price momentum:** (1-month and 3-month trend — supportive or a warning sign)

**Recent news themes:** (based on the headlines provided — 1-2 sentences on what's
currently driving sentiment, without inventing details beyond the headlines)

**Bottom line:** (2-3 sentences synthesizing the above into the final call)

Data:
"""


def build_analyst_report(ticker, forecast, price_estimate, current_price, market_context):
    def fmt_pct(x):
        return f"{x:.1%}" if x is not None else "N/A"

    def fmt_num(x, prefix="$"):
        return f"{prefix}{x:,.2f}" if x is not None else "N/A"

    news_lines = "\n".join(
        f"- \"{n['title']}\" ({n['publisher']})" for n in market_context.get("news", [])
    ) or "No recent headlines available."

    data_block = f"""
DCF intrinsic value/share: {fmt_num(price_estimate)}
Current market price: {fmt_num(current_price)}
Implied gap: {fmt_pct((price_estimate / current_price - 1) if price_estimate and current_price else None)}

Analyst target price (mean): {fmt_num(market_context.get('target_mean_price'))}
Analyst target range: {fmt_num(market_context.get('target_low_price'))} - {fmt_num(market_context.get('target_high_price'))}
Number of analysts: {market_context.get('num_analysts') or 'N/A'}
Consensus rating (Yahoo Finance): {market_context.get('recommendation_key') or 'N/A'}

Trailing P/E: {market_context.get('trailing_pe') or 'N/A'}
Forward P/E: {market_context.get('forward_pe') or 'N/A'}
PEG ratio: {market_context.get('peg_ratio') or 'N/A'}
52-week range: {fmt_num(market_context.get('fifty_two_week_low'))} - {fmt_num(market_context.get('fifty_two_week_high'))}
Beta: {market_context.get('beta') or 'N/A'}

1-month price momentum: {fmt_pct(market_context.get('momentum_1mo'))}
3-month price momentum: {fmt_pct(market_context.get('momentum_3mo'))}

Recent news headlines:
{news_lines}
"""
    return ai_summary(data_block, ANALYST_TEMPLATE)


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("AI Stock Valuation Model (Free Version)")
inject_custom_css()

with st.sidebar:
    st.header("How assumptions work")
    st.caption(
        "Discount rate, growth rate, and terminal growth are calculated automatically per "
        "company from its own financials, capital structure, and analyst estimates. "
        "Expand 'Advanced overrides' below the forecast if you want to test different scenarios."
    )

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

    # --- Auto-calculated assumptions, per company ---
    wacc_auto, wacc_details = estimate_wacc(ticker)
    terminal_growth_auto = estimate_terminal_growth(wacc_details["risk_free_rate"])
    growth_auto, growth_hist, growth_analyst = estimate_growth_rate(ticker, tuple(financials["Revenue"].values))
    buyback_auto, buyback_ok = estimate_buyback_rate(ticker)

    with st.expander("⚙️ Advanced overrides (defaults are auto-calculated for this company)"):
        st.caption(
            f"Auto WACC: {wacc_auto:.1%}  (β={wacc_details['beta']:.2f}, "
            f"risk-free={wacc_details['risk_free_rate']:.1%}, "
            f"cost of debt={wacc_details['cost_of_debt']:.1%}, "
            f"equity/debt weight={wacc_details['weight_equity']:.0%}/{wacc_details['weight_debt']:.0%})"
        )
        override_wacc = st.checkbox("Override discount rate (WACC)", key=f"wacc_ovr_{ticker}")
        discount_rate = (
            st.slider("Discount rate (WACC)", 0.04, 0.20, float(round(wacc_auto, 2)), 0.01, key=f"wacc_{ticker}")
            if override_wacc else wacc_auto
        )

        st.caption(f"Auto terminal growth: {terminal_growth_auto:.1%} (anchored to the risk-free rate, capped at 1–3%)")
        override_terminal = st.checkbox("Override terminal growth rate", key=f"term_ovr_{ticker}")
        terminal_growth = (
            st.slider("Terminal growth rate", 0.0, 0.05, float(round(terminal_growth_auto, 3)), 0.005, key=f"term_{ticker}")
            if override_terminal else terminal_growth_auto
        )

        analyst_note = f"{growth_analyst:.1%} analyst estimate" if growth_analyst is not None else "no analyst estimate found"
        st.caption(f"Auto revenue growth: {growth_auto:.1%}  (blend of {growth_hist:.1%} historical CAGR + {analyst_note})")
        override_growth = st.checkbox("Override revenue growth", key=f"growth_ovr_{ticker}")
        growth_rate = (
            st.slider("Revenue growth rate", -0.10, 0.40, float(round(growth_auto, 2)), 0.01, key=f"growth_{ticker}")
            if override_growth else growth_auto
        )

        buyback_note = "estimated from 3-year average buyback spend ÷ market cap" if buyback_ok else "no buyback data found, defaulted to 0%"
        st.caption(f"Auto buyback rate (illustrative only, see below): {buyback_auto:.1%}/year ({buyback_note})")
        override_buyback = st.checkbox("Override buyback rate", key=f"buyback_ovr_{ticker}")
        buyback_rate = (
            st.slider("Annual share count reduction (%)", 0.0, 8.0, float(round(buyback_auto * 100, 1)), 0.5, key=f"buyback_{ticker}") / 100
            if override_buyback else buyback_auto
        )

    st.caption(
        f"Forecast assumptions in use — growth: {growth_rate:.1%} · WACC: {discount_rate:.1%} · "
        f"terminal growth: {terminal_growth:.1%}"
    )

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
    forecast_display = pd.DataFrame({
        "Year": forecast["Year"].astype(int),
        "Revenue": forecast["Revenue"].apply(format_large_currency),
        "Operating Income": forecast["Operating Income"].apply(format_large_currency),
        "Free Cash Flow": forecast["FCF"].apply(format_large_currency),
    })
    st.table(forecast_display)

    st.header("5. DCF Valuation")
    st.caption(
        "Standard DCF: unlevered free cash flow discounted at WACC, summed with a Gordon-growth "
        "terminal value to get Enterprise Value, then bridged to Equity Value using the company's "
        "actual net cash/debt position, divided by today's share count."
    )

    shares = stock.info.get("sharesOutstanding")
    net_cash = wacc_details["cash"] - wacc_details["total_debt"]

    price_estimate = None
    if not shares:
        st.error("Shares outstanding unavailable for this ticker; can't compute per-share value.")
    else:
        price_estimate, enterprise_value, equity_value = run_dcf_standard(
            forecast, shares, net_cash, discount_rate, terminal_growth
        )
        if price_estimate is None:
            st.error("Discount rate must be greater than terminal growth rate.")

    if price_estimate is not None:
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

        with st.expander("Enterprise → Equity bridge"):
            st.write(f"Enterprise Value: {format_large_currency(enterprise_value)}")
            st.write(f"Net cash (cash − total debt): {format_large_currency(net_cash)}")
            st.write(f"Equity Value: {format_large_currency(equity_value)}")

        st.subheader("Sensitivity: intrinsic value/share by discount rate")
        rates = [discount_rate + d for d in (-0.02, -0.01, 0, 0.01, 0.02) if discount_rate + d > terminal_growth]
        sens_rows = []
        for r in rates:
            iv, _, _ = run_dcf_standard(forecast, shares, net_cash, r, terminal_growth)
            sens_rows.append({"Discount Rate": f"{r:.1%}", "Value/Share": f"${iv:,.2f}"})
        st.table(pd.DataFrame(sens_rows))

        if buyback_rate > 0:
            illustrative_price, _ = run_dcf_buyback_illustrative(
                forecast, shares, buyback_rate, discount_rate, terminal_growth
            )
            with st.expander("📈 Illustrative: value if buybacks keep shrinking the share count"):
                st.caption(
                    "Not part of the standard DCF above (that already values all future cash use "
                    "through the Enterprise → Equity bridge). This shows, for context only, what "
                    "per-share value looks like if the company keeps repurchasing shares at its "
                    "recent rate and you credit that shrinking share count directly."
                )
                st.metric("Illustrative Value/Share (buybacks continue)", f"${illustrative_price:,.2f}")

    st.header("6. AI Analyst Report")
    st.caption("Weighs the DCF alongside analyst consensus, valuation multiples, price momentum, and recent news — not the DCF alone.")
    if price_estimate is not None:
        with st.spinner("Pulling analyst consensus, multiples, and recent news..."):
            market_context = get_market_context(ticker)
        with st.spinner("Generating analyst report..."):
            report = build_analyst_report(ticker, forecast, price_estimate, current_price, market_context)
        remaining_report = render_verdict_banner(report)
        st.markdown(remaining_report)
    else:
        st.info("Analyst report skipped — DCF valuation was not computable above.")
