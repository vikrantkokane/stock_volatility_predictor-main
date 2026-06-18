# VOLTAI — Stock Market Volatility Prediction

A production-grade Flask web app that trains **four deep learning models**
on 5 years of Yahoo Finance data and compares their volatility forecasting performance.

## Models
| Model | Architecture |
|-------|-------------|
| GARCH-LSTM | Hybrid variance-aware LSTM (GARCH-inspired branch + LSTM branch) |
| GRU | Stacked Gated Recurrent Units |
| RNN | Stacked Simple Recurrent Network |
| BiRNN | Bidirectional LSTM (reads sequences in both directions) |

## Features
- **5 engineered features**: log returns, log volume, RSI-14, Bollinger Band width, ATR-14
- **60-day look-back window** for sequence modelling
- **Metrics**: Direction Accuracy %, RMSE, MAE
- **Forecasts**: 3-day, 5-day, 7-day iterative predictions
- **Charts**: Matplotlib dark-theme charts embedded as base64

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## Usage
1. Enter any valid stock ticker (e.g. `AAPL`, `TSLA`, `RELIANCE.NS`)
2. Click **Analyse**
3. All four models train automatically — watch terminal for progress
4. Results page shows metrics, comparison charts, and forecasts

## Project Structure
```
voltai/
├── app.py                      # Flask entry point
├── train_models.py             # Pipeline orchestrator
├── requirements.txt
├── models/
│   ├── __init__.py
│   ├── model_definitions.py    # Keras model builders
│   └── saved/                  # Trained .keras files (auto-created)
├── utils/
│   ├── __init__.py
│   ├── data_pipeline.py        # Download + feature engineering + sequences
│   ├── metrics.py              # RMSE / MAE / direction accuracy
│   └── charts.py               # Matplotlib chart generators
├── templates/
│   ├── base.html
│   ├── index.html              # Landing page
│   └── results.html            # Results page
└── static/
    ├── css/main.css
    └── js/main.js
```

## Notes
- Training takes ~2–5 minutes per ticker on CPU (30 epochs, early stopping)
- Models are saved to `models/saved/` and can be reused
- Supports US tickers (AAPL, TSLA…) and NSE tickers (RELIANCE.NS, TCS.NS…)
- **Not financial advice** — for academic / research use only
