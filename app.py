import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date

st.set_page_config(page_title="JP | AI Trading Risk Management", page_icon="📈", layout="wide")

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: radial-gradient(circle at 15% 0%, #14233d 0%, #080d18 45%, #070b12 100%); }
.block-container { padding-top: 1.4rem; }
.jp-logo {
 display:inline-flex; align-items:center; justify-content:center;
 width:58px;height:58px;border-radius:16px;
 background:linear-gradient(135deg,#21d4fd,#2157d5);
 color:#fff;font-size:25px;font-weight:900;letter-spacing:-2px;
 box-shadow:0 8px 30px rgba(33,140,230,.28); margin-right:14px;
}
.brand { display:flex;align-items:center;margin-bottom:8px; }
.brand-title { font-size:28px;font-weight:800;color:#f3f7ff; }
.brand-sub { color:#9aabc5;font-size:12px;letter-spacing:2px; }
.metric-card {
 background:rgba(19,31,51,.78); border:1px solid rgba(125,160,210,.18);
 border-radius:16px;padding:18px 20px; min-height:112px;
}
.metric-label {color:#9aabc5;font-size:13px;}
.metric-value {color:#f5f8ff;font-size:27px;font-weight:750;margin-top:7px;}
.small-note {color:#9aabc5;font-size:12px;}
div[data-testid="stAlert"] { border-radius:12px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="brand">
  <div class="jp-logo">JP</div>
  <div>
    <div class="brand-title">AI Automatic Trading Risk Management</div>
    <div class="brand-sub">INTELLIGENT MARKET ANALYSIS • RISK AWARENESS</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.caption("Educational decision-support dashboard — signals are estimates, not guaranteed predictions or personal investment advice.")

with st.sidebar:
    st.header("⚙️ Analysis Settings")
    ticker = st.text_input("Stock ticker", value="RELIANCE.NS", help="Examples: RELIANCE.NS, TCS.NS, INFY.NS, AAPL")
    period = st.selectbox("Historical data period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
    interval = st.selectbox("Chart interval", ["1d", "1wk"], index=0)
    st.divider()
    st.markdown("**JP Project**")
    st.caption("Version 1.0 • Paper analysis only")

@st.cache_data(ttl=300, show_spinner=False)
def load_data(symbol, selected_period, selected_interval):
    data = yf.download(symbol, period=selected_period, interval=selected_interval,
                       auto_adjust=True, progress=False)
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.dropna()

def calculate_indicators(df):
    out = df.copy()
    close = out["Close"]
    out["SMA20"] = close.rolling(20).mean()
    out["SMA50"] = close.rolling(50).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))
    out["Daily Return"] = close.pct_change()
    out["Volatility"] = out["Daily Return"].rolling(20).std() * np.sqrt(252) * 100
    return out

if not ticker.strip():
    st.info("Enter a stock ticker in the sidebar to begin.")
    st.stop()

try:
    with st.spinner(f"Loading market data for {ticker.upper()}..."):
        raw = load_data(ticker.strip().upper(), period, interval)
    if raw.empty or "Close" not in raw:
        st.error("No stock data was found. Check the ticker symbol and try again.")
        st.stop()

    data = calculate_indicators(raw)
    latest = data.iloc[-1]
    previous_close = data["Close"].iloc[-2] if len(data) > 1 else latest["Close"]
    price = float(latest["Close"])
    change = ((price / float(previous_close)) - 1) * 100 if previous_close else 0
    rsi = float(latest["RSI"]) if pd.notna(latest["RSI"]) else 50.0
    sma20 = float(latest["SMA20"]) if pd.notna(latest["SMA20"]) else price
    sma50 = float(latest["SMA50"]) if pd.notna(latest["SMA50"]) else price

    # Transparent, rule-based educational signal (not a trained predictive model).
    score = 0
    score += 1 if price > sma20 else -1
    score += 1 if sma20 > sma50 else -1
    score += 1 if 45 <= rsi <= 65 else (-1 if rsi > 70 or rsi < 30 else 0)
    if score >= 2:
        signal, signal_color, explanation = "BUY WATCH", "🟢", "Trend and momentum conditions are relatively positive."
    elif score <= -2:
        signal, signal_color, explanation = "SELL / REDUCE WATCH", "🔴", "Trend conditions are relatively weak; review risk before acting."
    else:
        signal, signal_color, explanation = "HOLD / WAIT", "🟡", "Signals are mixed. Waiting for clearer conditions may reduce impulsive decisions."

    returns = data["Daily Return"].dropna()
    annual_vol = float(returns.std() * np.sqrt(252) * 100) if len(returns) > 2 else 0
    if annual_vol < 20:
        risk = "Lower historical volatility"
    elif annual_vol < 40:
        risk = "Moderate historical volatility"
    else:
        risk = "Higher historical volatility"

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("LATEST PRICE", f"₹{price:,.2f}", f"{change:+.2f}% vs previous period"),
        ("RULE-BASED SIGNAL", f"{signal_color} {signal}", "Educational indicator only"),
        ("RSI (14)", f"{rsi:.1f}", "Below 30: oversold zone • Above 70: overbought zone"),
        ("VOLATILITY", f"{annual_vol:.1f}%", risk),
    ]
    for col, (label, value, note) in zip([c1,c2,c3,c4], cards):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="small-note">{note}</div></div>', unsafe_allow_html=True)

    st.subheader(f"📊 {ticker.upper()} — Price & Trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Price", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=data.index, y=data["SMA20"], name="SMA 20", line=dict(width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=data.index, y=data["SMA50"], name="SMA 50", line=dict(width=1.5, dash="dash")))
    fig.update_layout(template="plotly_dark", height=440, margin=dict(l=10,r=10,t=30,b=10),
                      legend=dict(orientation="h", y=1.08), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", xaxis_title="", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([1,1])
    with left:
        st.subheader("🤖 Signal Explanation")
        st.markdown(f"### {signal_color} {signal}")
        st.write(explanation)
        st.write(f"- Price vs SMA 20: **{'Above' if price > sma20 else 'Below'}**")
        st.write(f"- SMA 20 vs SMA 50: **{'Bullish alignment' if sma20 > sma50 else 'Bearish alignment'}**")
        st.write(f"- RSI: **{rsi:.1f}**")
        st.caption("The signal uses simple moving averages and RSI rules. It is not a machine-learning forecast and does not know future prices.")
    with right:
        st.subheader("🛡️ Risk Planning Calculator")
        st.caption("Illustrative planning levels based on a user-selected risk percentage.")
        entry = st.number_input("Example entry price (₹)", min_value=0.01, value=round(price, 2), step=1.0)
        risk_pct = st.slider("Illustrative stop distance (%)", 1.0, 15.0, 5.0, 0.5)
        reward_ratio = st.slider("Illustrative reward-to-risk ratio", 1.0, 4.0, 2.0, 0.5)
        stop = entry * (1 - risk_pct / 100)
        target = entry * (1 + (risk_pct * reward_ratio) / 100)
        st.metric("Illustrative stop level", f"₹{stop:,.2f}")
        st.metric("Illustrative target level", f"₹{target:,.2f}")
        st.caption("These are calculator outputs only. Gaps, fees, liquidity and market conditions can cause actual outcomes to differ.")

    with st.expander("View latest market data"):
        st.dataframe(data.tail(20).round(2), use_container_width=True)
        csv = data.to_csv().encode("utf-8")
        st.download_button("Download analysis CSV", csv, file_name=f"{ticker.upper()}_analysis.csv", mime="text/csv")

except Exception as exc:
    st.error("The data could not be loaded. Check your internet connection and ticker symbol.")
    st.caption(f"Technical detail: {exc}")

st.divider()
st.markdown("**JP — AI Automatic Trading Risk Management**  |  Student project prototype")
st.caption("Market data may be delayed, incomplete, or unavailable. This application does not execute trades. Always verify information independently.")
