"""
🚨 Pump Radar PRO — Multi-Ticker Market Scanner

Install:
    pip install streamlit yfinance pandas numpy plotly

Run:
    streamlit run pump_radar_pro.py
"""

import time
from datetime import datetime
import concurrent.futures
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="Pump Radar PRO",
    page_icon="🚨",
    layout="wide",
)

st.title("🚨 Pump Radar PRO — Multi-Ticker Scanner")
st.caption(
    "Scan the market for high-volume momentum stocks, potential pumps, "
    "and early breakout candidates."
)

# ============================================================
# PRE-DEFINED TICKER LISTS
# ============================================================

# You can customize these lists
PENNY_STOCKS = [
    "AMIX", "YXT", "WETO", "HOLO", "SVMH", "MULN", "BBIG", "DWAC",
    "GNS", "REV", "RDBX", "ATER", "PROG", "CEI", "MMTLP", "TRKA",
    "SNDL", "TLRY", "ACB", "CAN", "CRBP", "CYBN", "MNMD", "ATAI",
]

SMALL_CAPS = [
    "GME", "AMC", "NIO", "PLTR", "SOFI", "RIVN", "LCID", "MARA",
    "RIOT", "COIN", "HOOD", "ROKU", "UPST", "AFRM", "SQ", "DASH",
    "UBER", "LYFT", "SNAP", "TWTR", "PINS", "SPCE", "NKLA", "QS",
]

NASDAQ_100 = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX",
    "ADBE", "INTC", "CSCO", "CMCSA", "PEP", "COST", "TMUS", "AVGO",
]

# Define watchlist groups
WATCHLISTS = {
    "Penny Stocks": PENNY_STOCKS,
    "Small Caps": SMALL_CAPS,
    "NASDAQ 100": NASDAQ_100,
    "Custom": [],  # Will be filled by user
}

# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

@st.cache_data(ttl=30, show_spinner=False)
def load_single_ticker(ticker, interval="5m", period="5d"):
    """Load data for a single ticker with error handling"""
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
            return None, f"No data for {ticker}"

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(c).title() for c in df.columns]

        required = ["Open", "High", "Low", "Close", "Volume"]
        available_cols = [c for c in required if c in df.columns]
        
        if not available_cols:
            return None, f"No required columns for {ticker}"
            
        df = df[available_cols].copy()
        df = df.dropna(subset=["Close"])

        if len(df) < 10:
            return None, f"Insufficient data for {ticker} ({len(df)} rows)"

        return df, None

    except Exception as e:
        return None, f"Error loading {ticker}: {str(e)}"


def analyze_ticker(ticker, df):
    """Analyze a ticker and return metrics"""
    if df is None or len(df) < 10:
        return None

    try:
        close = df["Close"].astype(float)
        volume = df["Volume"].fillna(0).astype(float)

        price = float(close.iloc[-1])
        current_volume = volume.iloc[-1] if len(volume) > 0 else 0

        # Calculate price changes
        def pct_change(bars):
            if len(close) <= bars:
                return np.nan
            previous = close.iloc[-bars - 1]
            if previous == 0:
                return np.nan
            return (price / previous - 1) * 100

        # Relative volume
        rvol = np.nan
        baseline_length = min(60, len(volume) - 1)
        if baseline_length > 5 and len(volume) > 0:
            baseline = volume.iloc[-baseline_length:-1].median()
            if baseline > 0 and current_volume > 0:
                rvol = current_volume / baseline

        # Volume acceleration
        volume_acc = np.nan
        if len(volume) >= 15:
            recent_volume = volume.iloc[-3:].mean()
            previous_volume = volume.iloc[-15:-3].mean()
            if previous_volume > 0:
                volume_acc = recent_volume / previous_volume

        # RSI
        def calculate_rsi(series, period=14):
            if len(series) <= period:
                return np.nan
            delta = series.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            return 100 - (100 / (1 + rs))

        current_rsi = float(calculate_rsi(close)) if len(close) > 14 else np.nan

        # Breakout detection
        breakout = False
        if len(close) >= 21:
            previous_high = float(close.iloc[-21:-1].max())
            breakout = price > previous_high

        # Get info
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info or {}
            market_cap = info.get("marketCap", np.nan)
            float_shares = info.get("floatShares", np.nan)
            short_ratio = info.get("shortRatio", np.nan)
            avg_volume = info.get("averageVolume", np.nan)
            sector = info.get("sector", "")
        except:
            market_cap = np.nan
            float_shares = np.nan
            short_ratio = np.nan
            avg_volume = np.nan
            sector = ""

        # Calculate scores
        pump_score = 0

        # Momentum score
        move_1 = pct_change(1)
        move_5 = pct_change(5)
        move_15 = pct_change(15)
        
        if not pd.isna(move_1):
            if move_1 >= 5: pump_score += 15
            elif move_1 >= 2: pump_score += 10

        if not pd.isna(move_5):
            if move_5 >= 15: pump_score += 25
            elif move_5 >= 8: pump_score += 20
            elif move_5 >= 4: pump_score += 10

        # RVOL score
        if not pd.isna(rvol):
            if rvol >= 10: pump_score += 20
            elif rvol >= 5: pump_score += 15
            elif rvol >= 3: pump_score += 10
            elif rvol >= 2: pump_score += 5

        # Volume acceleration score
        if not pd.isna(volume_acc):
            if volume_acc >= 5: pump_score += 10
            elif volume_acc >= 2.5: pump_score += 7
            elif volume_acc >= 1.5: pump_score += 4

        # Breakout bonus
        if breakout:
            pump_score += 8

        # Float/low market cap bonus
        if not pd.isna(float_shares):
            if float_shares <= 10_000_000:
                pump_score += 15
            elif float_shares <= 50_000_000:
                pump_score += 8

        if not pd.isna(market_cap):
            if market_cap <= 100_000_000:
                pump_score += 10
            elif market_cap <= 500_000_000:
                pump_score += 5

        pump_score = min(100, pump_score)

        # Dump risk score
        dump_risk = 0
        if not pd.isna(move_5):
            if move_5 >= 20:
                dump_risk += 20
            elif move_5 >= 10:
                dump_risk += 12

        if not pd.isna(rvol):
            if rvol >= 10:
                dump_risk += 15
            elif rvol >= 5:
                dump_risk += 8

        if not pd.isna(current_rsi):
            if current_rsi >= 90:
                dump_risk += 15
            elif current_rsi >= 80:
                dump_risk += 10
            elif current_rsi >= 70:
                dump_risk += 5

        if not pd.isna(float_shares):
            if float_shares <= 2_000_000:
                dump_risk += 15
            elif float_shares <= 10_000_000:
                dump_risk += 8

        dump_risk = min(100, dump_risk)

        # Signal
        if dump_risk >= 70:
            signal = "🔴 DUMP RISK"
        elif pump_score >= 70:
            signal = "🟢 STRONG PUMP"
        elif pump_score >= 50:
            signal = "🟡 WATCH"
        else:
            signal = "⚪ LOW SIGNAL"

        return {
            "ticker": ticker,
            "price": price,
            "move_1": move_1,
            "move_5": move_5,
            "move_15": move_15,
            "rvol": rvol,
            "volume_acceleration": volume_acc,
            "rsi": current_rsi,
            "pump_score": pump_score,
            "dump_risk": dump_risk,
            "breakout": breakout,
            "market_cap": market_cap,
            "float_shares": float_shares,
            "short_ratio": short_ratio,
            "avg_volume": avg_volume,
            "sector": sector,
            "signal": signal,
        }
    except Exception as e:
        return None


def scan_tickers(tickers, interval, period):
    """Scan multiple tickers in parallel"""
    results = []
    errors = []
    
    progress_bar = st.progress(0, "Scanning tickers...")
    status_text = st.empty()

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {
            executor.submit(load_single_ticker, ticker, interval, period): ticker
            for ticker in tickers
        }

        for idx, future in enumerate(concurrent.futures.as_completed(future_to_ticker)):
            ticker = future_to_ticker[future]
            try:
                df, error = future.result(timeout=15)
                if df is not None:
                    result = analyze_ticker(ticker, df)
                    if result is not None:
                        results.append(result)
                    else:
                        errors.append(f"{ticker}: Analysis failed")
                else:
                    if error:
                        errors.append(error)
            except Exception as e:
                errors.append(f"{ticker}: Timeout or error - {str(e)}")

            progress_bar.progress((idx + 1) / len(tickers))
            status_text.text(f"Scanned {idx + 1}/{len(tickers)} tickers...")

    progress_bar.empty()
    status_text.empty()
    
    # Show errors in expandable section
    if errors:
        with st.expander(f"⚠️ {len(errors)} errors occurred during scan"):
            for error in errors[:20]:  # Show first 20 errors
                st.text(error)
            if len(errors) > 20:
                st.text(f"... and {len(errors) - 20} more errors")

    return results


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Scanner Settings")

    watchlist_choice = st.selectbox(
        "Watchlist",
        list(WATCHLISTS.keys())
    )

    if watchlist_choice == "Custom":
        custom_tickers = st.text_area(
            "Enter tickers (comma separated)",
            "AAPL,MSFT,GOOGL"
        )
        tickers = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
    else:
        tickers = WATCHLISTS[watchlist_choice]

    st.divider()

    interval = st.selectbox(
        "Interval",
        ["1m", "5m", "15m", "30m"],
        index=1,
    )

    period = st.selectbox(
        "History",
        ["1d", "5d", "1mo", "3mo"],
        index=1,
    )

    st.divider()

    st.subheader("🎯 Scan Filters")

    min_pump_score = st.slider(
        "Minimum Pump Score",
        0,
        100,
        40,  # Lowered default to show more results
    )

    min_rvol = st.slider(
        "Minimum RVOL",
        0.0,
        20.0,
        2.0,  # Lowered default
    )

    min_volume = st.number_input(
        "Min Current Volume",
        min_value=0,
        value=50000,  # Lowered default
        step=10000,
    )

    st.divider()

    st.subheader("🔄 Auto Scan")

    auto_scan = st.checkbox("Auto scan", False)
    scan_interval = st.slider(
        "Scan interval (seconds)",
        10,
        300,
        30,
        10,
    )

    st.divider()

    st.caption(
        "⚠️ Data is from Yahoo Finance and may be delayed. "
        "Not financial advice. Always do your own research."
    )

# ============================================================
# MAIN SCANNER
# ============================================================

if not tickers:
    st.warning("No tickers to scan. Please add some to your watchlist.")
    st.stop()

# Scan button
col1, col2 = st.columns([1, 4])
with col1:
    scan_button = st.button("🔍 Scan Now", use_container_width=True)
with col2:
    st.info(f"📊 {len(tickers)} tickers in watchlist")

# Initialize session state for results
if "scan_results" not in st.session_state:
    st.session_state.scan_results = None

# Run scan
if scan_button or auto_scan:
    with st.spinner(f"Scanning {len(tickers)} tickers..."):
        results = scan_tickers(tickers, interval, period)
        st.session_state.scan_results = results

# Display results
if st.session_state.scan_results is not None:
    results = st.session_state.scan_results
    
    if not results:
        st.warning("No valid data returned for any ticker. This could mean:")
        st.markdown("""
        - **Market is closed** - Try during trading hours
        - **Invalid tickers** - Check if tickers are correct
        - **Rate limiting** - Yahoo Finance may be rate limiting
        - **No volume** - Tickers might have zero volume today
        """)
        st.stop()

    # Convert to DataFrame for analysis
    df_results = pd.DataFrame(results)

    # Sort by pump score
    df_results = df_results.sort_values("pump_score", ascending=False)

    # Apply filters
    filtered = df_results[
        (df_results["pump_score"] >= min_pump_score) &
        (df_results["rvol"] >= min_rvol) &
        (df_results["avg_volume"] >= min_volume)
    ]

    # Display stats
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Total Scanned", len(df_results))
    col2.metric("🎯 Candidates Found", len(filtered))
    col3.metric("📈 Avg Pump Score", f"{df_results['pump_score'].mean():.1f}/100")

    # ============================================================
    # DISPLAY TABLE
    # ============================================================

    if not filtered.empty:
        st.success(f"🎯 Found {len(filtered)} potential pump candidates!")

        # Format the display
        display_df = filtered.copy()

        # Format columns
        display_df["price"] = display_df["price"].apply(lambda x: f"${x:.2f}")
        display_df["move_5"] = display_df["move_5"].apply(
            lambda x: f"{x:+.1f}%" if not pd.isna(x) else "N/A"
        )
        display_df["rvol"] = display_df["rvol"].apply(
            lambda x: f"{x:.1f}x" if not pd.isna(x) else "N/A"
        )
        display_df["pump_score"] = display_df["pump_score"].apply(
            lambda x: f"{x}/100"
        )
        display_df["dump_risk"] = display_df["dump_risk"].apply(
            lambda x: f"{x}/100"
        )
        display_df["rsi"] = display_df["rsi"].apply(
            lambda x: f"{x:.1f}" if not pd.isna(x) else "N/A"
        )
        display_df["breakout"] = display_df["breakout"].apply(
            lambda x: "✅" if x else "❌"
        )

        # Keep only relevant columns
        columns = [
            "ticker", "price", "move_5", "rvol",
            "pump_score", "dump_risk", "signal",
            "rsi", "breakout", "sector"
        ]

        # Display as interactive table
        st.dataframe(
            display_df[columns],
            hide_index=True,
            use_container_width=True,
            height=400,
        )

        # ============================================================
        # TOP CANDIDATES CHART
        # ============================================================

        st.subheader("📈 Top Candidates Chart")

        top_n = min(10, len(filtered))
        top_tickers = filtered.head(top_n)["ticker"].tolist()

        # Create bar chart
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=top_tickers,
                y=filtered.head(top_n)["pump_score"],
                name="Pump Score",
                marker_color="green",
            )
        )

        fig.add_trace(
            go.Bar(
                x=top_tickers,
                y=filtered.head(top_n)["dump_risk"],
                name="Dump Risk",
                marker_color="red",
            )
        )

        fig.update_layout(
            title="Top Candidates: Pump Score vs Dump Risk",
            xaxis_title="Ticker",
            yaxis_title="Score",
            barmode="group",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

        # ============================================================
        # DETAILED VIEW FOR TOP TICKER
        # ============================================================

        st.subheader("🔍 Detailed Analysis")

        selected_ticker = st.selectbox(
            "Select ticker for details",
            top_tickers
        )

        if selected_ticker:
            selected_data = filtered[filtered["ticker"] == selected_ticker].iloc[0]

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Price", f"${selected_data['price']:.2f}")
            col2.metric("5m Move", selected_data["move_5"])
            col3.metric("RVOL", selected_data["rvol"])
            col4.metric("Signal", selected_data["signal"])

            col1, col2, col3 = st.columns(3)

            col1.metric("Pump Score", selected_data["pump_score"])
            col2.metric("Dump Risk", selected_data["dump_risk"])
            col3.metric("RSI", selected_data["rsi"])

            # Show more details
            with st.expander("📊 More Details"):
                detail_data = {
                    "Ticker": selected_data["ticker"],
                    "Sector": selected_data["sector"],
                    "Market Cap": f"${selected_data['market_cap']:,.0f}" if not pd.isna(selected_data['market_cap']) else "N/A",
                    "Float Shares": f"{selected_data['float_shares']:,.0f}" if not pd.isna(selected_data['float_shares']) else "N/A",
                    "Short Ratio": f"{selected_data['short_ratio']:.2f}" if not pd.isna(selected_data['short_ratio']) else "N/A",
                    "Avg Volume": f"{selected_data['avg_volume']:,.0f}" if not pd.isna(selected_data['avg_volume']) else "N/A",
                    "Breakout": "Yes" if selected_data["breakout"] else "No",
                }
                st.json(detail_data)

    else:
        st.warning("No candidates match your filter criteria. Try adjusting the thresholds:")
        st.markdown(f"""
        - Lower **Minimum Pump Score** (currently {min_pump_score})
        - Lower **Minimum RVOL** (currently {min_rvol})
        - Lower **Min Current Volume** (currently {min_volume})
        """)
        
        # Show top 5 tickers regardless of filters
        st.subheader("📊 Top 5 tickers by Pump Score (regardless of filters)")
        top_5 = df_results.head(5)[["ticker", "price", "move_5", "rvol", "pump_score"]]
        top_5["price"] = top_5["price"].apply(lambda x: f"${x:.2f}")
        top_5["move_5"] = top_5["move_5"].apply(lambda x: f"{x:+.1f}%" if not pd.isna(x) else "N/A")
        top_5["rvol"] = top_5["rvol"].apply(lambda x: f"{x:.1f}x" if not pd.isna(x) else "N/A")
        st.dataframe(top_5, hide_index=True, use_container_width=True)

    # ============================================================
    # AUTO REFRESH
    # ============================================================

    if auto_scan:
        st.caption(f"🔄 Auto-refreshing in {scan_interval} seconds...")
        time.sleep(scan_interval)
        st.rerun()

else:
    st.info("👆 Click 'Scan Now' to start scanning the watchlist.")

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    f"🕐 Last scan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
    f"📊 Watchlist: {len(tickers)} tickers"
)