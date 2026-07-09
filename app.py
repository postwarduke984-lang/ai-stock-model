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
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": f"Summarize this 10-K section and extract growth drivers and risks:\n\n{text}"}
        ],
        "temperature": 0.2
    }

    r = requests.post(url, headers=headers, json=data)

    # If Groq returns an error, show it instead of crashing
    if "error" in r.json():
        return f"Groq API Error: {r.json()['error']['message']}"

    # Normal successful response
    return r.json()["choices"][0]["message"]["content"]



# -----------------------------
# SEC 10-K FETCHER (FREE)
# -----------------------------
def get_10k(ticker):
    cik_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{ticker}.json"
    # For simplicity, use a placeholder (real EDGAR parsing is longer)
    return "10-K text placeholder for demo. Replace with EDGAR fetcher."


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

    st.header("2. Financial Data (Simplified)")
    # Placeholder financials (you can expand this)
    financials = pd.DataFrame({
        "Revenue": [100e9, 110e9, 120e9],
        "Operating Margin": [0.25, 0.26, 0.27]
    })

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
