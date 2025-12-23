import streamlit as st
from login import login

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

with st.sidebar:
    st.write(f"Logged in as {st.session_state.username}")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()


"""Minimalist NSE Stock Analysis Tool."""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta
from dataclasses import dataclass
from typing import Optional, Dict
from sklearn.preprocessing import MinMaxScaler

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="NSE Stock Analyzer", layout="wide", page_icon="📈")

CACHE_TTL = 3600  # 1 hour


# ─────────────────────────────────────────────────────────────────────────────
# Data Layer
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=CACHE_TTL)
def load_symbols() -> list[str]:
    """Load NSE symbols from CSV."""
    try:
        df = pd.read_csv("symbols.csv")
        return [f"{s}.NS" for s in df["Symbol"].tolist()]
    except Exception:
        return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]


@st.cache_data(ttl=CACHE_TTL)
def fetch_info(ticker: str) -> dict:
    """Fetch company info."""
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=CACHE_TTL)
def fetch_history(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Fetch OHLCV history."""
    try:
        df = yf.download(ticker, start, end, auto_adjust=True, progress=False)
        if df.empty:
            return df
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL)
def fetch_financials(ticker: str) -> Dict[str, pd.DataFrame]:
    """Fetch all financial statements."""
    try:
        t = yf.Ticker(ticker)
        return {
            "quarterly": t.quarterly_financials,
            "annual": t.financials,
            "balance": t.balance_sheet,
            "cashflow": t.cashflow,
        }
    except Exception:
        return {"quarterly": pd.DataFrame(), "annual": pd.DataFrame(), 
                "balance": pd.DataFrame(), "cashflow": pd.DataFrame()}


# ─────────────────────────────────────────────────────────────────────────────
# Indicators
# ─────────────────────────────────────────────────────────────────────────────

def calc_sma(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window).mean()


def calc_ema(series: pd.Series, window: int = 20) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calc_bollinger(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    sma = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()
    return pd.DataFrame({
        "middle": sma,
        "upper": sma + 2 * std,
        "lower": sma - 2 * std,
    }, index=df.index)


def calc_macd(series: pd.Series) -> pd.DataFrame:
    ema12 = series.ewm(span=12).mean()
    ema26 = series.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return pd.DataFrame({"macd": macd, "signal": signal, "hist": macd - signal})


# ─────────────────────────────────────────────────────────────────────────────
# Forecasting (Simplified)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Forecast:
    next_price: float
    direction: str
    confidence: float


def simple_forecast(prices: pd.Series) -> Optional[Forecast]:
    """Simple momentum-based forecast (no heavy ML)."""
    if len(prices) < 30:
        return None
    
    # Use recent momentum
    recent = prices.tail(20)
    sma_5 = prices.tail(5).mean()
    sma_20 = recent.mean()
    
    momentum = (sma_5 - sma_20) / sma_20 * 100
    last_price = float(prices.iloc[-1])
    
    # Simple projection
    next_price = last_price * (1 + momentum / 100)
    direction = "📈 Bullish" if momentum > 0 else "📉 Bearish"
    confidence = min(abs(momentum) * 10, 95)  # Cap at 95%
    
    return Forecast(next_price=next_price, direction=direction, confidence=confidence)


# ─────────────────────────────────────────────────────────────────────────────
# UI Components
# ─────────────────────────────────────────────────────────────────────────────

def fmt(val, prefix: str = "") -> str:
    """Format number for display."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if isinstance(val, (int, float)):
        if abs(val) >= 1e9:
            return f"{prefix}{val/1e9:.2f}B"
        if abs(val) >= 1e6:
            return f"{prefix}{val/1e6:.2f}M"
        if abs(val) >= 1e3:
            return f"{prefix}{val/1e3:.2f}K"
        return f"{prefix}{val:,.2f}"
    return str(val)


def render_price_chart(df: pd.DataFrame, chart_type: str) -> None:
    """Render interactive price chart."""
    if df.empty:
        st.warning("No price data available.")
        return
    
    fig = go.Figure()
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="OHLC"
        ))
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="Close", line=dict(color="#2196F3")))
    
    fig.update_layout(
        height=400, template="plotly_white", 
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)


def render_indicators(df: pd.DataFrame) -> None:
    """Render technical indicators."""
    if df.empty or len(df) < 30:
        st.info("Need at least 30 days of data for indicators.")
        return
    
    close = df["close"]
    
    # Moving Averages + Bollinger
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Moving Averages")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=close, name="Close", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=calc_sma(close, 20), name="SMA 20", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=df.index, y=calc_ema(close, 20), name="EMA 20", line=dict(dash="dash")))
        fig.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0), template="plotly_white", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### Bollinger Bands")
        bb = calc_bollinger(df)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=close, name="Close", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=bb.index, y=bb["upper"], name="Upper", line=dict(dash="dot", color="gray")))
        fig.add_trace(go.Scatter(x=bb.index, y=bb["lower"], name="Lower", line=dict(dash="dot", color="gray"), fill="tonexty"))
        fig.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0), template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    # RSI + MACD
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### RSI (14)")
        rsi = calc_rsi(close)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI", line=dict(color="#9C27B0")))
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0), template="plotly_white", yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### MACD")
        macd = calc_macd(close)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=macd["macd"], name="MACD", line=dict(color="#2196F3")))
        fig.add_trace(go.Scatter(x=df.index, y=macd["signal"], name="Signal", line=dict(color="#FF5722")))
        fig.add_trace(go.Bar(x=df.index, y=macd["hist"], name="Histogram", marker_color="gray", opacity=0.5))
        fig.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0), template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)


def render_financials(data: Dict[str, pd.DataFrame]) -> None:
    """Render financial tables."""
    tabs = st.tabs(["Quarterly", "Annual", "Balance Sheet", "Cash Flow"])
    
    for tab, (name, df) in zip(tabs, data.items()):
        with tab:
            if df.empty:
                st.info(f"No {name} data available.")
            else:
                # Clean up the dataframe
                df = df.dropna(how="all")
                if hasattr(df.columns, "date"):
                    df.columns = [c.strftime("%Y-%m") if hasattr(c, "strftime") else str(c) for c in df.columns]
                st.dataframe(df.head(15), use_container_width=True)


def render_screener(df: pd.DataFrame, info: dict) -> None:
    """Render screener metrics."""
    if df.empty:
        st.info("No data for screening.")
        return
    
    close = df["close"]
    last = float(close.iloc[-1])
    sma20 = float(calc_sma(close, 20).iloc[-1]) if len(close) >= 20 else None
    rsi_val = float(calc_rsi(close).iloc[-1]) if len(close) >= 14 else None
    
    # Price metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Last Close", fmt(last, "₹"))
    col2.metric("SMA 20", fmt(sma20, "₹"))
    col3.metric("RSI", fmt(rsi_val))
    col4.metric("Volume", fmt(df["volume"].iloc[-1] if "volume" in df else None))
    
    # 52-week
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("52W Low", fmt(info.get("fiftyTwoWeekLow"), "₹"))
    col2.metric("52W High", fmt(info.get("fiftyTwoWeekHigh"), "₹"))
    col3.metric("Market Cap", fmt(info.get("marketCap"), "₹"))
    col4.metric("PE Ratio", fmt(info.get("trailingPE")))
    
    # Margins
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Profit Margin", fmt(info.get("profitMargins"), "") + "%" if info.get("profitMargins") else "N/A")
    col2.metric("ROE", fmt(info.get("returnOnEquity"), "") + "%" if info.get("returnOnEquity") else "N/A")
    col3.metric("Debt/Equity", fmt(info.get("debtToEquity")))
    col4.metric("Current Ratio", fmt(info.get("currentRatio")))


def render_forecast(df: pd.DataFrame) -> None:
    """Render forecast section."""
    if df.empty or len(df) < 30:
        st.info("Need more historical data for forecasting.")
        return
    
    result = simple_forecast(df["close"])
    if result is None:
        st.warning("Unable to generate forecast.")
        return
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Next Close", fmt(result.next_price, "₹"))
    col2.metric("Direction", result.direction)
    col3.metric("Momentum Confidence", f"{result.confidence:.1f}%")
    
    st.caption("⚠️ This is a simple momentum-based projection, not financial advice.")


# ─────────────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("## 📈 NSE Stock Analyzer")
    st.caption("Analyze, screen, and forecast National Stock Exchange stocks")
    
    # Stock selector
    symbols = load_symbols()
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        ticker = st.selectbox("Select Stock", symbols, index=symbols.index("TCS.NS") if "TCS.NS" in symbols else 0)
    
    today = date.today()
    with col2:
        start = st.date_input("From", today - timedelta(days=365), max_value=today)
    with col3:
        end = st.date_input("To", today, min_value=start, max_value=today)
    
    # Load data
    with st.spinner("Loading..."):
        info = fetch_info(ticker)
        history = fetch_history(ticker, start, end)
        financials = fetch_financials(ticker)
    
    # Company header
    if info:
        st.markdown(f"### {info.get('longName', ticker)}")
        st.caption(f"{info.get('sector', '')} • {info.get('industry', '')}")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Price", "📈 Indicators", "💰 Financials", "🎯 Screener"])
    
    with tab1:
        chart_type = st.radio("Chart", ["Candlestick", "Line"], horizontal=True, label_visibility="collapsed")
        render_price_chart(history, chart_type)
        
        if not history.empty:
            csv = history.reset_index().to_csv(index=False).encode()
            st.download_button("📥 Download CSV", csv, f"{ticker}_history.csv", "text/csv")
        
        with st.expander("Company Info"):
            if info:
                st.write(info.get("longBusinessSummary", "No description available."))
            else:
                st.info("No company info available.")
    
    with tab2:
        render_indicators(history)
    
    with tab3:
        render_financials(financials)
    
    with tab4:
        st.markdown("##### Quick Metrics")
        render_screener(history, info)
        
        st.divider()
        st.markdown("##### Momentum Forecast")
        render_forecast(history)


if __name__ == "__main__":
    main()
