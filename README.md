# 🛡️ Crypto Swing Sentinel (CSS)

**A Machine Learning CLI for Bitcoin Swing Trading**

Crypto Swing Sentinel is a Python-based market intelligence tool designed for the "set and forget" trader. Instead of staring at charts all day, CSS runs once daily, analyzes macro trends and technical indicators using an **XGBoost** machine learning model, and outputs a concise "Morning Intelligence Report" directly in your terminal.

---

## 🚀 Features

*   **Machine Learning Prediction:** Uses XGBoost (Gradient Boosting) to predict high-probability directional moves (>2.5%) for the next 24 hours.
*   **Macro & Technical Context:** Analyzes Price vs. 200 SMA, RSI Momentum, and Volatility Squeezes (Bollinger Bands).
*   **Risk Management:** Automatically calculates Dynamic Stop Losses based on ATR (Average True Range) to protect capital.
*   **Terminal Dashboard:** A clean, zero-distraction CLI interface with color-coded signals (Red/Green/Neutral).
*   **Dynamic Reasoning:** The system explains *why* it is bullish or bearish based on current data.

---

## 🛠️ Tech Stack

*   **Language:** Python 3.11
*   **Data Source:** `yfinance` (Yahoo Finance API)
*   **ML Engine:** `xgboost` (Extreme Gradient Boosting)
*   **Technical Analysis:** `ta` library
*   **Data Manipulation:** `pandas`, `scikit-learn`

---

## ⚙️ Installation Guide

### Prerequisites
*   Python 3.11 (Recommended)
*   MacOS Users: Homebrew installed

### 1. Clone & Setup
```bash
# Create project directory
mkdir crypto_sentinel
cd crypto_sentinel

# Create Virtual Environment (Python 3.11)
python3.11 -m venv venv

# Activate Environment
source venv/bin/activate
```

### 2. Install System Dependencies (MacOS Only)
XGBoost requires the OpenMP runtime, which is not installed by default on Macs.
```bash
brew install libomp
```

### 3. Install Python Libraries
```bash
pip install --upgrade pip
pip install yfinance pandas ta xgboost scikit-learn requests joblib
```

---

## 🖥️ Usage

### Step 1: Train the Model
You must train the model before making predictions. The model learns from historical data (2014–Present).
```bash
python train.py
```
*Output: Saves a `model.pkl` file to your directory.*

### Step 2: Run the Daily Dashboard
Run this command once a day (preferably after the Daily Close at 00:00 UTC).
```bash
python main.py
```

### Example Output:
```text
===============================================
   BTC SWING STRATEGIST (Daily)
===============================================
LAST UPDATE: 2023-12-10 14:30 UTC
NEXT UPDATE: Next Daily Close
-----------------------------------------------
--- MARKET CONTEXT ----------------------------
PRICE:       $98,230.50
TREND (SMA): BULLISH (Price > 200 SMA)
VOLATILITY:  LOW (Squeeze Detected)
-----------------------------------------------
--- FORECAST FOR TODAY ------------------------
DIRECTION:   HOLD / NEUTRAL ⚪
CONFIDENCE:  45.2%

REASONING:   Model is indecisive. CAUTION: Trend is
             up, but RSI is Overbought. Volatility
             squeeze active; expect a violent move.
-----------------------------------------------
```

---

## 📂 Project Structure

```text
crypto_sentinel/
├── data_loader.py    # Fetches data from Yahoo & calculates indicators (RSI, ATR, etc.)
├── train.py          # Trains the XGBoost model and saves 'model.pkl'
├── main.py           # Runs the daily inference and generates the Terminal Dashboard
├── model.pkl         # The saved Machine Learning model (generated after training)
└── venv/             # Virtual Environment folder
```

---

## 🧠 Strategy & Logic

### The Target
The model does not try to predict every small wiggle. It looks for **Volatility Expansion**.
*   **Target:** `1` (Buy Signal) only if tomorrow's price moves **> +2.5%**.
*   This filters out choppy/ranging days and focuses on capturing swings.

### The Input Features
1.  **Trend:** Distance from SMA 50 & SMA 200.
2.  **Momentum:** RSI (Relative Strength Index) 14-day.
3.  **Volatility:** ATR (Average True Range) & Bollinger Band Width.
4.  **Volume:** Raw volume data.

### Risk Management
The dashboard provides a **Stop Loss** calculated as:
`Current Price - (2.0 * ATR)`
*   **High Volatility:** Stop loss widens to avoid fake-outs.
*   **Low Volatility:** Stop loss tightens to preserve capital.
