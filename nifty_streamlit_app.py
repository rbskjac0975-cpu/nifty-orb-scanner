import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="NIFTY 50 Multi-Confirmation Scanner", page_icon="📈", layout="wide")

st.title("📈 NIFTY 50 Multi-Confirmation Trade Scanner")
st.caption("Market structure • VWAP • EMA • Volume • Momentum • Breakout • ORB • Final Validator")

# ---------------- Dashboard Alerts ----------------
if enable_alerts:
    st.subheader("🔔 Live Dashboard Alerts")
    if alert_messages:
        for msg in alert_messages:
            if "BUY ALERT" in msg or "BULLISH" in msg:
                st.success(msg)
            elif "SELL ALERT" in msg or "BEARISH" in msg:
                st.error(msg)
            else:
                st.warning(msg)
    else:
        st.info("No new high-confidence alert. Scanner is monitoring for confirmation.")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Scanner Settings")
    symbol = st.selectbox("Instrument", ["NIFTY 50", "BANK NIFTY", "FINNIFTY"])
    mode = st.selectbox("Mode", ["All Confirmations", "Momentum Scalper", "Breakout", "Reversal", "Opening Range"])
    min_conf = st.slider("Minimum confidence", 1, 10, 7)
    min_rr = st.slider("Minimum R:R", 1.0, 5.0, 2.0, 0.5)
    risk_points = st.number_input("Default risk (points)", min_value=1.0, value=30.0, step=1.0)
    st.divider()
    st.subheader("🔔 Alerts")
    enable_alerts = st.toggle("Enable dashboard alerts", value=True)
    alert_conf = st.slider("Alert confidence", 1, 10, 8)
    alert_volume = st.toggle("Require volume confirmation", value=True)
    st.info("Alerts are dashboard/browser-session alerts. For Telegram/WhatsApp/email, connect a notification service.")
    if st.button("🔄 Refresh data now"):
        st.cache_data.clear()
        st.rerun()

# ---------------- Demo data layer ----------------
@st.cache_data(ttl=15)
def get_market_data():
    """
    Fetch live/recent NIFTY 50 1-minute data from Yahoo Finance.
    Yahoo Finance symbol for NIFTY 50: ^NSEI

    Note: Yahoo's 1-minute history is generally limited to recent data.
    """
    ticker = "^NSEI"

    try:
        data = yf.download(
            ticker,
            period="1d",
            interval="1m",
            auto_adjust=False,
            progress=False,
            prepost=False,
            threads=False,
        )

        if data.empty:
            return pd.DataFrame()

        # yfinance can return MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            if ticker in data.columns.get_level_values(-1):
                data = data.xs(ticker, axis=1, level=-1)
            else:
                data.columns = data.columns.get_level_values(0)

        data = data.reset_index()

        # Normalize column names
        data.columns = [str(c).lower().replace(" ", "_") for c in data.columns]

        # Yahoo may call the timestamp column Datetime
        timestamp_col = "datetime" if "datetime" in data.columns else "date"
        if timestamp_col not in data.columns:
            timestamp_col = data.columns[0]

        data = data.rename(columns={timestamp_col: "timestamp"})

        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in data.columns]
        if missing:
            st.error(f"Yahoo Finance response is missing columns: {missing}")
            return pd.DataFrame()

        data = data[required].copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"])

        # Make numeric columns numeric
        for col in ["open", "high", "low", "close", "volume"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        data = data.dropna(subset=["open", "high", "low", "close"])
        data["volume"] = data["volume"].fillna(0)
        return data.sort_values("timestamp").reset_index(drop=True)

    except Exception as e:
        st.error(f"Yahoo Finance data fetch failed: {e}")
        return pd.DataFrame()

# ---------------- Indicators ----------------
def ema(s, length):
    return s.ewm(span=length, adjust=False).mean()

def rsi(s, length=14):
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

df["ema9"] = ema(df["close"], 9)
df["ema21"] = ema(df["close"], 21)
df["rsi"] = rsi(df["close"])
df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
df["vol_ma"] = df["volume"].rolling(20).mean()

# Higher-timeframe approximations from the minute data
m5 = df.set_index("timestamp").resample("5min").agg(
    open=("open","first"), high=("high","max"), low=("low","min"),
    close=("close","last"), volume=("volume","sum")
).dropna()

m15 = df.set_index("timestamp").resample("15min").agg(
    open=("open","first"), high=("high","max"), low=("low","min"),
    close=("close","last"), volume=("volume","sum")
).dropna()

def structure(frame):
    if len(frame) < 5:
        return "Neutral"
    fast = ema(frame["close"], 9).iloc[-1]
    slow = ema(frame["close"], 21).iloc[-1]
    last = frame["close"].iloc[-1]
    if last > fast > slow:
        return "Bullish"
    if last < fast < slow:
        return "Bearish"
    return "Neutral"

trend15 = structure(m15)
trend5 = structure(m5)

last = df.iloc[-1]
price = float(last["close"])

# ---------------- Signal engine ----------------
checks = {
    "15M Trend": 0,
    "5M Trend": 0,
    "VWAP": 0,
    "EMA 9/21": 0,
    "Volume": 0,
    "Momentum / RSI": 0,
    "Price Action": 0,
    "Breakout / Retest": 0,
}

bull = bear = 0

def add(signal, weight=1):
    global bull, bear
    if signal == "bull":
        bull += weight
    elif signal == "bear":
        bear += weight

if trend15 == "Bullish":
    checks["15M Trend"] = 1; add("bull", 2)
elif trend15 == "Bearish":
    checks["15M Trend"] = -1; add("bear", 2)

if trend5 == "Bullish":
    checks["5M Trend"] = 1; add("bull", 2)
elif trend5 == "Bearish":
    checks["5M Trend"] = -1; add("bear", 2)

if price > last["vwap"]:
    checks["VWAP"] = 1; add("bull")
else:
    checks["VWAP"] = -1; add("bear")

if last["ema9"] > last["ema21"]:
    checks["EMA 9/21"] = 1; add("bull")
else:
    checks["EMA 9/21"] = -1; add("bear")

if last["volume"] > last["vol_ma"]:
    checks["Volume"] = 1
    # volume confirms direction rather than creating direction
    if bull > bear: add("bull")
    elif bear > bull: add("bear")

if last["rsi"] >= 55:
    checks["Momentum / RSI"] = 1; add("bull")
elif last["rsi"] <= 45:
    checks["Momentum / RSI"] = -1; add("bear")

body = last["close"] - last["open"]
if body > 0:
    checks["Price Action"] = 1; add("bull")
elif body < 0:
    checks["Price Action"] = -1; add("bear")

recent_high = df["high"].rolling(20).max().iloc[-2]
recent_low = df["low"].rolling(20).min().iloc[-2]
if price > recent_high and last["volume"] > last["vol_ma"]:
    checks["Breakout / Retest"] = 1; add("bull", 2)
elif price < recent_low and last["volume"] > last["vol_ma"]:
    checks["Breakout / Retest"] = -1; add("bear", 2)

total = bull + bear
if bull > bear:
    direction = "BUY"
    strength = bull / max(total, 1)
elif bear > bull:
    direction = "SELL"
    strength = bear / max(total, 1)
else:
    direction = "NO TRADE"
    strength = 0

confidence = int(round(10 * strength))

# Conflict filter
conflicts = abs(bull - bear) <= 2
if confidence < min_conf or conflicts:
    direction = "NO TRADE"

# ---------------- Risk model ----------------
if direction == "BUY":
    entry = price
    sl = entry - risk_points
    target1 = entry + risk_points * min_rr
    target2 = entry + risk_points * 3
elif direction == "SELL":
    entry = price
    sl = entry + risk_points
    target1 = entry - risk_points * min_rr
    target2 = entry - risk_points * 3
else:
    entry = sl = target1 = target2 = None

# ---------------- Alert engine ----------------
alert_messages = []

if enable_alerts:
    # High-confidence final signal
    if direction in ("BUY", "SELL") and confidence >= alert_conf:
        if not alert_volume or last["volume"] > last["vol_ma"]:
            alert_messages.append(
                f"🚨 {direction} ALERT — {symbol} @ {price:,.2f} | "
                f"Confidence {confidence}/10 | SL {sl:,.2f} | T1 {target1:,.2f}"
            )

    # VWAP cross
    prev = df.iloc[-2]
    if prev["close"] <= prev["vwap"] and last["close"] > last["vwap"]:
        alert_messages.append("📈 VWAP BULLISH CROSS — Price moved above VWAP.")
    elif prev["close"] >= prev["vwap"] and last["close"] < last["vwap"]:
        alert_messages.append("📉 VWAP BEARISH CROSS — Price moved below VWAP.")

    # EMA cross
    if prev["ema9"] <= prev["ema21"] and last["ema9"] > last["ema21"]:
        alert_messages.append("🟢 EMA BULLISH CROSS — 9 EMA crossed above 21 EMA.")
    elif prev["ema9"] >= prev["ema21"] and last["ema9"] < last["ema21"]:
        alert_messages.append("🔴 EMA BEARISH CROSS — 9 EMA crossed below 21 EMA.")

    # Volume expansion
    if last["volume"] > last["vol_ma"] * 1.5:
        alert_messages.append("🔥 VOLUME SPIKE — Current volume is >1.5× 20-period average.")

# ---------------- UI ----------------

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Price", f"{price:,.2f}")
c2.metric("15M Trend", trend15)
c3.metric("5M Trend", trend5)
c4.metric("RSI", f"{last['rsi']:.1f}")
c5.metric("VWAP", f"{last['vwap']:,.2f}")

st.divider()

left, right = st.columns([1, 1.4])

with left:
    st.subheader("🎯 Final Trade Validator")
    if direction == "BUY":
        st.success(f"### 🟢 BUY\nConfidence: {confidence}/10")
    elif direction == "SELL":
        st.error(f"### 🔴 SELL\nConfidence: {confidence}/10")
    else:
        st.warning(f"### ⚪ NO TRADE\nConfidence: {confidence}/10")

    if entry is not None:
        st.write(f"**Entry:** {entry:,.2f}")
        st.write(f"**Stop Loss:** {sl:,.2f}")
        st.write(f"**Target 1:** {target1:,.2f}")
        st.write(f"**Target 2:** {target2:,.2f}")
        st.write(f"**R:R:** 1:{min_rr:.1f}")
        st.write("**Invalidation:** Price closes beyond the setup's structural invalidation level or the confirmation stack breaks.")
    else:
        st.write("Signals are conflicting or below the required confidence threshold.")

with right:
    st.subheader("🔎 Confirmation Matrix")
    rows = []
    for name, value in checks.items():
        status = "🟢 Bullish" if value == 1 else "🔴 Bearish" if value == -1 else "⚪ Neutral"
        rows.append({"Factor": name, "Signal": status})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()

# ---------------- Price chart ----------------
st.subheader("📊 Recent Price Action")
chart = df.set_index("timestamp")[["close", "ema9", "ema21", "vwap"]].tail(150)
st.line_chart(chart, height=420)

# ---------------- Market levels ----------------
st.subheader("📍 Key Levels")
level_cols = st.columns(4)
levels = [
    ("20-Bar Resistance", recent_high),
    ("20-Bar Support", recent_low),
    ("VWAP", last["vwap"]),
    ("Current Price", price),
]
for col, (name, value) in zip(level_cols, levels):
    col.metric(name, f"{value:,.2f}")

st.divider()
st.caption(
    "Educational trading tool. Demo data is synthetic until a live broker/data feed is connected. "
    "Signals should not be treated as guaranteed or personalized investment advice."
)
