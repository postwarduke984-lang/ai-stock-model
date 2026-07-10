import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import numpy as np
import datetime
import json

# -----------------------------
# FREE AI (Groq Llama-3)
# -----------------------------
import requests

GROQ_API_KEY = "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"


def ai_summary(text):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": f"Summarize this 10-K section and extract growth drivers and risks:\n\n{text}"}
        ],
        "temperature": 0.2
    }

    r = requests.post(url, headers=headers, json=data)

    if "error" in r.json():
        return f"Groq API Error: {r.json()['error']['message']}"

    return r.json()["choices"][0]["message"]["content"]




# -----------------------------
# SEC 10-K FETCHER (FREE)
# -----------------------------

def get_cik_from_ticker(ticker):
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    r = requests.get(url, headers=headers)
    data = r.json()

    ticker = ticker.upper()
    for entry in data.values():
        if entry["ticker"].upper() == ticker:
            cik_str = str(entry["cik_str"])
            return cik_str.zfill(10)  # SEC requires 10-digit CIK

    raise ValueError(f"CIK not found for ticker {ticker}")


def get_10k(ticker):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    # 1) Convert ticker → CIK
    cik = get_cik_from_ticker(ticker)

    # 2) Get all filings for this company
    subs_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(subs_url, headers=headers)
    data = r.json()

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    primaries = recent.get("primaryDocument", [])

    # 3) Find the latest 10-K
    for i, form in enumerate(forms):
        if form == "10-K":
            accession = accessions[i].replace("-", "")
            primary_doc = primaries[i]

            # 4) Download the actual filing document
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{primary_doc}"
            doc_resp = requests.get(doc_url, headers=headers)

            text = doc_resp.text

            # 5) Trim to avoid sending 300k+ characters to Groq
            return text[:20000]

    return "No 10-K filing found for this ticker."



# -----------------------------
# SIMPLE 5-YEAR FORECAST MODEL
# -----------------------------
def build_forecast(financials):
    rev = financials["Revenue"].iloc[-1]
    op_margin = financials["Operating Margin"].iloc[-1]

    years = []
    revenues = []
    op_incomes = []
    fcfs = []

    for i in range(1, 6):
        growth = 0.05  # 5% baseline growth
        rev = rev * (1 + growth)
        op_income = rev * op_margin
        fcf = op_income * 0.7

        years.append(datetime.datetime.now().year + i)
        revenues.append(rev)
        op_incomes.append(op_income)
        fcfs.append(fcf)

    df = pd.DataFrame({
        "Year": years,
        "Revenue": revenues,
        "Operating Income": op_incomes,
        "FCF": fcfs
    })

    return df


# -----------------------------
# DCF VALUATION
# -----------------------------
def run_dcf(forecast, discount_rate=0.10, terminal_growth=0.02):
    fcfs = forecast["FCF"].values
    discounted = []

    for i, fcf in enumerate(fcfs):
        discounted.append(fcf / ((1 + discount_rate) ** (i + 1)))

    terminal_value = fcfs[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
    terminal_value_discounted = terminal_value / ((1 + discount_rate) ** len(fcfs))

    intrinsic_value = sum(discounted) + terminal_value_discounted
    return intrinsic_value


def get_financials(ticker):
    stock = yf.Ticker(ticker)
    try:
        income = stock.financials

        if income is None or income.empty:
            raise ValueError("No financial data found")

        if "Total Revenue" not in income.index or "Operating Income" not in income.index:
            raise ValueError("Missing required financial fields")

        revenue = income.loc["Total Revenue"].iloc[:3].values
        op_income = income.loc["Operating Income"].iloc[:3].values
        op_margin = np.divide(op_income, revenue, out=np.zeros_like(op_income), where=revenue!=0)

        df = pd.DataFrame({
            "Revenue": revenue,
            "Operating Margin": op_margin
        })
        return df

    except Exception as e:
        st.warning(f"Financial data unavailable: {e}")
        return pd.DataFrame({
            "Revenue": [100e9, 110e9, 120e9],
            "Operating Margin": [0.25, 0.26, 0.27]
        })


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("AI Stock Valuation Model (Free Version)")
ticker = st.text_input("Enter a stock ticker (AAPL, MSFT, TSLA):")

if ticker:
    st.header("1. Market Data")
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5y")

    st.line_chart(hist["Close"])

    st.header("2. Financial Data (Real Data)")
    financials = get_financials(ticker)
    st.write(financials)


    st.header("3. AI Summary of 10-K")
    tenk = get_10k(ticker)
    summary = ai_summary(tenk)
    st.write(summary)

    st.header("4. 5-Year Forecast")
    forecast = build_forecast(financials)
    st.write(forecast)

    st.header("5. DCF Valuation")
    intrinsic = run_dcf(forecast)
    shares = stock.info.get("sharesOutstanding", 1)
    price_estimate = intrinsic / shares

    st.metric("Estimated Intrinsic Value per Share", f"${price_estimate:,.2f}")

    st.header("6. AI Analyst Report")
    report = ai_summary(
        f"Here is the forecast: {forecast.to_string()} and the DCF value: {price_estimate}"
    )
    st.write(report)
