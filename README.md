# 📈 NSE Stock Analyzer

A minimalist, production-ready stock analysis tool for National Stock Exchange (NSE) listed companies. Built with Streamlit for an intuitive web interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=for-the-badge&logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📊 **Price Charts** | Interactive candlestick and line charts with Plotly |
| 📈 **Technical Indicators** | SMA, EMA, RSI, MACD, Bollinger Bands |
| 💰 **Financials** | Quarterly/Annual statements, Balance Sheet, Cash Flow |
| 🎯 **Stock Screener** | Key metrics, 52-week range, margins, ratios |
| 🔮 **Momentum Forecast** | Simple trend-based next-day prediction |
| 📥 **Data Export** | Download historical data as CSV |

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Build the image
docker build -t nse-stock-analyzer .

# Run the container
docker run -d -p 8501:8501 --name stock-app nse-stock-analyzer

# Open in browser
open http://localhost:8501
```

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/sumittttttt/Stock-market-prediction-and-screener.git
cd Stock-market-prediction-and-screener

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app using the provided script
python run.py

# Or run directly with Streamlit
streamlit run Home.py
```

---

## 📁 Project Structure

```
Stock-market-prediction-and-screener/
├── Home.py              # Main Streamlit application
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker configuration
├── .dockerignore        # Docker build exclusions
├── symbols.csv          # NSE stock symbols database
├── LICENSE              # MIT License
└── README.md            # This file
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **Charts** | Plotly |
| **Data Source** | Yahoo Finance (yfinance) |
| **Data Processing** | Pandas, NumPy |
| **ML** | Scikit-learn |
| **Containerization** | Docker |

---

## 📊 Technical Indicators

The app calculates the following indicators in real-time:

- **Moving Averages**: SMA (20-period), EMA (20-period)
- **Bollinger Bands**: 20-period with 2 standard deviations
- **RSI**: 14-period Relative Strength Index
- **MACD**: 12/26/9 configuration

---

## 🎯 Screener Metrics

| Category | Metrics |
|----------|---------|
| **Price** | Last Close, SMA 20, RSI, Volume |
| **Range** | 52-Week Low/High, Day Low/High |
| **Fundamentals** | Market Cap, PE Ratio |
| **Margins** | Profit Margin, ROE, Debt/Equity, Current Ratio |

---

## 🔮 Forecasting

The app uses a **momentum-based forecasting** approach:

1. Calculates 5-day and 20-day moving averages
2. Measures momentum as the percentage difference
3. Projects next-day price based on current momentum
4. Provides directional signal (Bullish/Bearish) with confidence score

> ⚠️ **Disclaimer**: This is for educational purposes only. Not financial advice.

---

## 🐳 Docker Commands

```bash
# Build image
docker build -t nse-stock-analyzer .

# Run container
docker run -d -p 8501:8501 --name stock-app nse-stock-analyzer

# View logs
docker logs -f stock-app

# Stop container
docker stop stock-app

# Remove container
docker rm stock-app

# Run with custom port
docker run -d -p 3000:8501 nse-stock-analyzer
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAMLIT_SERVER_PORT` | 8501 | Server port |
| `STREAMLIT_SERVER_HEADLESS` | true | Run without browser |
| `STREAMLIT_BROWSER_GATHERUSAGESTATS` | false | Disable telemetry |

### Streamlit Config

Create `.streamlit/config.toml` for custom settings:

```toml
[server]
port = 8501
headless = true

[theme]
primaryColor = "#2196F3"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

---

## 📈 Data Sources

- **Stock Data**: [Yahoo Finance](https://finance.yahoo.com/) via `yfinance`
- **Symbols**: 1,700+ NSE-listed companies from `symbols.csv`

---

## 🧪 Development

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run with auto-reload
streamlit run Home.py --server.runOnSave true

# Check syntax
python -m py_compile Home.py
```

---

## 🚢 Deployment Options

### Streamlit Community Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set main file: `Home.py`
5. Deploy

### Docker on Cloud

```bash
# AWS/GCP/Azure - Push to container registry
docker tag nse-stock-analyzer:latest your-registry/nse-stock-analyzer:latest
docker push your-registry/nse-stock-analyzer:latest

# Deploy to cloud container service
```

### Heroku

```bash
heroku container:push web -a your-app-name
heroku container:release web -a your-app-name
```

---

## 📝 API Reference

### Data Functions

| Function | Description |
|----------|-------------|
| `load_symbols()` | Load NSE stock symbols from CSV |
| `fetch_info(ticker)` | Fetch company information |
| `fetch_history(ticker, start, end)` | Fetch OHLCV price history |
| `fetch_financials(ticker)` | Fetch financial statements |

### Indicator Functions

| Function | Description |
|----------|-------------|
| `calc_sma(series, window)` | Simple Moving Average |
| `calc_ema(series, window)` | Exponential Moving Average |
| `calc_rsi(series, period)` | Relative Strength Index |
| `calc_bollinger(df, window)` | Bollinger Bands |
| `calc_macd(series)` | MACD with Signal line |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing framework
- [Yahoo Finance](https://finance.yahoo.com/) for market data
- [Plotly](https://plotly.com/) for interactive charts

---

## 📞 Support

- 📧 Create an [Issue](https://github.com/sumittttttt/Stock-market-prediction-and-screener/issues)
- ⭐ Star the repository if you find it useful!

---

<p align="center">
  Made with ❤️ for retail investors
</p>
