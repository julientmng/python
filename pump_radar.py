"""
🚨 Pump Radar — Streamlit app

Install:
    pip install streamlit yfinance pandas numpy plotly

Run:
    streamlit run pump_radar.py

This is a research/alerting scanner, not an automated trading system.
Yahoo Finance data may be delayed or unavailable for some symbols.
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="Pump Radar",
    page_icon="🚨",
    layout="wide",
)

st.title("🚨 Pump Radar")
st.caption(
    "Detect abnormal momentum, volume acceleration, low-float conditions, "
    "breakouts and potential exhaustion."
)


# ============================================================
# DATA
# ============================================================

@st.cache_data(ttl=30, show_spinner=False)
def load_history(ticker, period, interval):
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(c).title() for c in df.columns]

        required = ["Open", "High", "Low", "Close", "Volume"]
        df = df[[c for c in required if c in df.columns]].copy()
        df = df.dropna(subset=["Close"])

        return df

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_info(ticker):
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=180, show_spinner=False)
def load_news(ticker):
    try:
        news = yf.Ticker(ticker).news or []
        output = []

        for item in news[:10]:
            content = item.get("content", item)

            title = content.get("title", "")
            publisher = content.get("provider", {}).get("displayName", "")

            link = (
                content.get("canonicalUrl", {}).get("url")
                or content.get("clickThroughUrl", {}).get("url")
                or ""
            )

            if title:
                output.append(
                    {
                        "title": title,
                        "publisher": publisher,
                        "link": link,
                    }
                )

        return output

    except Exception:
        return []


# ============================================================
# INDICATORS
# ============================================================

def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def calculate_vwap(df):
    typical_price = (
        df["High"] +
        df["Low"] +
        df["Close"]
    ) / 3

    volume = df["Volume"].replace(0, np.nan)

    return (
        typical_price * volume
    ).cumsum() / volume.cumsum()


def num(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def format_number(value):
    if pd.isna(value):
        return "N/A"

    value = float(value)

    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"

    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"

    return f"{value:,.0f}"


def format_price(value):
    if pd.isna(value):
        return "N/A"

    return f"${value:,.2f}"


# ============================================================
# SCORING ENGINE
# ============================================================

def analyze(ticker, df, info):

    if len(df) < 10:
        return None

    close = df["Close"].astype(float)
    volume = df["Volume"].fillna(0).astype(float)

    price = float(close.iloc[-1])

    def pct_change(bars):
        if len(close) <= bars:
            return np.nan

        previous = close.iloc[-bars - 1]

        if previous == 0:
            return np.nan

        return (price / previous - 1) * 100

    move_1 = pct_change(1)
    move_5 = pct_change(5)
    move_15 = pct_change(15)

    # --------------------------------------------------------
    # Relative volume
    # --------------------------------------------------------

    baseline_length = min(60, len(volume) - 1)

    if baseline_length > 5:
        baseline = volume.iloc[-baseline_length:-1].median()
    else:
        baseline = np.nan

    current_volume = volume.iloc[-1]

    if baseline > 0:
        rvol = current_volume / baseline
    else:
        rvol = np.nan

    # --------------------------------------------------------
    # Volume acceleration
    # --------------------------------------------------------

    if len(volume) >= 15:

        recent_volume = volume.iloc[-3:].mean()
        previous_volume = volume.iloc[-15:-3].mean()

        if previous_volume > 0:
            volume_acceleration = (
                recent_volume / previous_volume
            )
        else:
            volume_acceleration = np.nan

    else:
        volume_acceleration = np.nan

    # --------------------------------------------------------
    # VWAP
    # --------------------------------------------------------

    temp = df.copy()

    temp["VWAP"] = calculate_vwap(temp)

    vwap = float(temp["VWAP"].iloc[-1])

    above_vwap = price > vwap

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    current_rsi = float(rsi(close).iloc[-1])

    # --------------------------------------------------------
    # Breakout
    # --------------------------------------------------------

    if len(close) >= 21:

        previous_high = float(
            close.iloc[-21:-1].max()
        )

        breakout = price > previous_high

    else:
        breakout = False

    # --------------------------------------------------------
    # Fundamental metadata
    # --------------------------------------------------------

    market_cap = num(info.get("marketCap"))
    float_shares = num(info.get("floatShares"))
    shares_outstanding = num(info.get("sharesOutstanding"))

    short_ratio = num(info.get("shortRatio"))
    average_volume = num(info.get("averageVolume"))

    sector = info.get("sector", "")
    industry = info.get("industry", "")
    exchange = info.get("exchange", "")

    effective_float = float_shares

    if pd.isna(effective_float):
        effective_float = shares_outstanding

    # ========================================================
    # MOMENTUM SCORE
    # ========================================================

    momentum = 0

    if not pd.isna(move_1):

        if move_1 >= 5:
            momentum += 12

        elif move_1 >= 2:
            momentum += 7

    if not pd.isna(move_5):

        if move_5 >= 15:
            momentum += 20

        elif move_5 >= 8:
            momentum += 15

        elif move_5 >= 4:
            momentum += 8

    if not pd.isna(move_15):

        if move_15 >= 30:
            momentum += 15

        elif move_15 >= 15:
            momentum += 10

        elif move_15 >= 7:
            momentum += 5

    if not pd.isna(rvol):

        if rvol >= 10:
            momentum += 20

        elif rvol >= 5:
            momentum += 15

        elif rvol >= 3:
            momentum += 10

        elif rvol >= 2:
            momentum += 5

    if not pd.isna(volume_acceleration):

        if volume_acceleration >= 5:
            momentum += 10

        elif volume_acceleration >= 2.5:
            momentum += 7

        elif volume_acceleration >= 1.5:
            momentum += 4

    if above_vwap:
        momentum += 5

    if breakout:
        momentum += 8

    momentum = min(momentum, 100)

    # ========================================================
    # LOW-FLOAT SCORE
    # ========================================================

    float_score = 0

    if not pd.isna(effective_float):

        if effective_float <= 2_000_000:
            float_score = 20

        elif effective_float <= 5_000_000:
            float_score = 16

        elif effective_float <= 10_000_000:
            float_score = 12

        elif effective_float <= 20_000_000:
            float_score = 8

        elif effective_float <= 50_000_000:
            float_score = 4

    # ========================================================
    # MARKET CAP SCORE
    # ========================================================

    market_score = 0

    if not pd.isna(market_cap):

        if market_cap <= 50_000_000:
            market_score = 15

        elif market_cap <= 150_000_000:
            market_score = 12

        elif market_cap <= 300_000_000:
            market_score = 9

        elif market_cap <= 1_000_000_000:
            market_score = 4

    # ========================================================
    # PUMP SCORE
    # ========================================================

    pump_score = round(
        momentum * 0.55
        + float_score * 1.25
        + market_score * 0.8
        + (10 if above_vwap else 0)
        + (10 if breakout else 0)
    )

    pump_score = min(100, pump_score)

    # ========================================================
    # DUMP / EXHAUSTION RISK
    # ========================================================

    dump_risk = 0

    if not pd.isna(move_5):

        if move_5 >= 20:
            dump_risk += 20

        elif move_5 >= 10:
            dump_risk += 12

        elif move_5 >= 5:
            dump_risk += 5

    if not pd.isna(move_15):

        if move_15 >= 40:
            dump_risk += 20

        elif move_15 >= 25:
            dump_risk += 12

        elif move_15 >= 15:
            dump_risk += 7

    if not pd.isna(rvol):

        if rvol >= 15:
            dump_risk += 20

        elif rvol >= 10:
            dump_risk += 15

        elif rvol >= 5:
            dump_risk += 8

    if not pd.isna(effective_float):

        if effective_float <= 2_000_000:
            dump_risk += 15

        elif effective_float <= 10_000_000:
            dump_risk += 8

    if not pd.isna(current_rsi):

        if current_rsi >= 90:
            dump_risk += 15

        elif current_rsi >= 80:
            dump_risk += 10

        elif current_rsi >= 70:
            dump_risk += 5

    if not above_vwap:
        dump_risk += 10

    if not pd.isna(short_ratio) and short_ratio >= 10:
        dump_risk += 5

    dump_risk = min(100, dump_risk)

    # ========================================================
    # SIGNAL
    # ========================================================

    if dump_risk >= 75:
        signal = "🔴 EXTREME DUMP RISK"

    elif dump_risk >= 55:
        signal = "🟠 HIGH DUMP RISK"

    elif pump_score >= 75:
        signal = "🟢 STRONG MOMENTUM"

    elif pump_score >= 55:
        signal = "🟡 WATCH"

    else:
        signal = "⚪ LOW SIGNAL"

    return {
        "ticker": ticker.upper(),
        "price": price,
        "move_1": move_1,
        "move_5": move_5,
        "move_15": move_15,
        "rvol": rvol,
        "volume_acceleration": volume_acceleration,
        "vwap": vwap,
        "above_vwap": above_vwap,
        "rsi": current_rsi,
        "float": effective_float,
        "float_score": float_score,
        "market_cap": market_cap,
        "short_ratio": short_ratio,
        "average_volume": average_volume,
        "momentum": momentum,
        "pump_score": pump_score,
        "dump_risk": dump_risk,
        "breakout": breakout,
        "sector": sector,
        "industry": industry,
        "exchange": exchange,
        "signal": signal,
    }


# ============================================================
# CHART
# ============================================================

def create_chart(df):

    chart = df.copy()

    chart["VWAP"] = calculate_vwap(chart)

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=chart.index,
            open=chart["Open"],
            high=chart["High"],
            low=chart["Low"],
            close=chart["Close"],
            name="Price",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=chart.index,
            y=chart["VWAP"],
            mode="lines",
            name="VWAP",
        )
    )

    fig.update_layout(
        height=500,
        xaxis_rangeslider_visible=False,
        margin=dict(
            l=10,
            r=10,
            t=30,
            b=10,
        ),
    )

    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Scanner")

    ticker = st.text_input(
        "Ticker",
        "YXT",
    ).strip().upper()

    period = st.selectbox(
        "History",
        ["1d", "5d", "1mo"],
        index=1,
    )

    interval = st.selectbox(
        "Interval",
        ["1m", "5m", "15m", "30m"],
        index=1,
    )

    st.divider()

    st.subheader("🚨 Alert thresholds")

    min_move = st.number_input(
        "Minimum 5-min move (%)",
        min_value=1.0,
        max_value=100.0,
        value=5.0,
    )

    min_rvol = st.number_input(
        "Minimum RVOL",
        min_value=1.0,
        max_value=100.0,
        value=5.0,
    )

    min_pump_score = st.slider(
        "Minimum Pump Score",
        0,
        100,
        75,
    )

    st.divider()

    auto_refresh = st.checkbox(
        "Auto refresh",
        False,
    )

    refresh_seconds = st.slider(
        "Refresh seconds",
        10,
        300,
        30,
        10,
    )

    st.divider()

    st.caption(
        "Yahoo Finance may provide delayed or incomplete intraday data. "
        "For live trading alerts, replace the data layer with a real-time "
        "broker/market-data API."
    )


# ============================================================
# LOAD
# ============================================================

if not ticker:
    st.warning("Enter a ticker.")
    st.stop()

df = load_history(
    ticker,
    period,
    interval,
)

info = load_info(ticker)

if df.empty:
    st.error(
        f"No market data returned for {ticker}. "
        "Try another ticker or interval."
    )
    st.stop()

result = analyze(
    ticker,
    df,
    info,
)

if result is None:
    st.error("Not enough data to calculate the scanner.")
    st.stop()


# ============================================================
# ALERT
# ============================================================

alert = (
    not pd.isna(result["move_5"])
    and result["move_5"] >= min_move
    and not pd.isna(result["rvol"])
    and result["rvol"] >= min_rvol
    and result["pump_score"] >= min_pump_score
)

if alert:

    st.error(
        f"🚨 PUMP RADAR ALERT — {ticker} | "
        f"5m {result['move_5']:+.1f}% | "
        f"RVOL {result['rvol']:.1f}× | "
        f"Pump Score {result['pump_score']}/100"
    )


# ============================================================
# METRICS
# ============================================================

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric(
    "Price",
    format_price(result["price"]),
)

c2.metric(
    "5m",
    (
        f"{result['move_5']:+.1f}%"
        if not pd.isna(result["move_5"])
        else "N/A"
    ),
)

c3.metric(
    "RVOL",
    (
        f"{result['rvol']:.1f}×"
        if not pd.isna(result["rvol"])
        else "N/A"
    ),
)

c4.metric(
    "Pump Score",
    f"{result['pump_score']}/100",
)

c5.metric(
    "Dump Risk",
    f"{result['dump_risk']}/100",
)

c6.metric(
    "RSI",
    (
        f"{result['rsi']:.1f}"
        if not pd.isna(result["rsi"])
        else "N/A"
    ),
)

st.subheader(result["signal"])


# ============================================================
# CHART + STRUCTURE
# ============================================================

left, right = st.columns(
    [1.5, 1]
)

with left:

    st.plotly_chart(
        create_chart(df.tail(150)),
        use_container_width=True,
    )

with right:

    st.subheader("📊 Market structure")

    rows = [
        ["Market Cap", format_number(result["market_cap"])],
        ["Float", format_number(result["float"])],
        ["Average Volume", format_number(result["average_volume"])],
        ["Current Volume", format_number(df["Volume"].iloc[-1])],
        [
            "Volume Acceleration",
            (
                f"{result['volume_acceleration']:.1f}×"
                if not pd.isna(result["volume_acceleration"])
                else "N/A"
            ),
        ],
        ["VWAP", format_price(result["vwap"])],
        [
            "Above VWAP",
            "YES" if result["above_vwap"] else "NO",
        ],
        [
            "Breakout",
            "YES" if result["breakout"] else "NO",
        ],
        [
            "Short Ratio",
            (
                f"{result['short_ratio']:.1f}"
                if not pd.isna(result["short_ratio"])
                else "N/A"
            ),
        ],
        ["Exchange", result["exchange"] or "N/A"],
        ["Sector", result["sector"] or "N/A"],
        ["Industry", result["industry"] or "N/A"],
    ]

    st.dataframe(
        pd.DataFrame(
            rows,
            columns=["Metric", "Value"],
        ),
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# SCORE BREAKDOWN
# ============================================================

st.subheader("🧠 Signal breakdown")

a, b, c = st.columns(3)

with a:

    st.metric(
        "Momentum",
        f"{result['momentum']}/100",
    )

    st.write(
        "Measures short-term price acceleration, relative volume, "
        "volume acceleration, VWAP and breakout behavior."
    )

with b:

    st.metric(
        "Low-Float Score",
        f"{result['float_score']}/20",
    )

    st.write(
        "Low float can create explosive moves but also increases "
        "liquidity and reversal risk."
    )

with c:

    st.metric(
        "Dump Risk",
        f"{result['dump_risk']}/100",
    )

    st.write(
        "Higher values indicate an increasingly extended or fragile move."
    )


# ============================================================
# NEWS
# ============================================================

st.subheader("📰 Recent news")

news = load_news(ticker)

if news:

    for item in news:

        title = item["title"]
        publisher = item["publisher"]
        link = item["link"]

        if link:

            st.markdown(
                f"- [{title}]({link})"
                + (
                    f" — {publisher}"
                    if publisher
                    else ""
                )
            )

        else:

            st.markdown(
                f"- {title}"
                + (
                    f" — {publisher}"
                    if publisher
                    else ""
                )
            )

else:

    st.caption(
        "No recent Yahoo Finance news was returned."
    )


# ============================================================
# WATCHLIST
# ============================================================

with st.expander("📋 Suggested watchlist format"):

    st.markdown(
        """
Use this app to monitor candidates such as:

- AMIX
- YXT
- WETO
- Other NASDAQ/NYSE small caps
- Low-float IPOs
- Recently reverse-split stocks
- Stocks with unusual premarket volume

A strong setup is generally:

**Price acceleration + abnormal volume + low float + VWAP hold + breakout**

Do not treat a high Pump Score as an automatic buy signal.
The most dangerous condition is often:

**Pump Score HIGH + Dump Risk HIGH**
"""
    )


# ============================================================
# FOOTER / REFRESH
# ============================================================

st.caption(
    "Last update: "
    + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
)

if auto_refresh:

    time.sleep(refresh_seconds)

    st.rerun()
