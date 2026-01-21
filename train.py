import argparse
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import precision_score
import joblib
from data_loader import fetch_data, prepare_targets

DEFAULT_TICKER = "BTC-USD"


def train_model(ticker):
    print("--- 1. Loading Data ---")
    df = fetch_data(ticker=ticker)
    df = prepare_targets(df)
    
    # Define features (Must match data_loader.py)
    features = [
        'ATR', 'SMA_50', 'SMA_200', 'RSI', 'BB_Width', 
        'Volume', 'Open', 'High', 'Low', 'Close'
    ]
    
    X = df[features]
    y = df['Target']
    
    # Split Data (Time-series split)
    split_point = int(len(df) * 0.85)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    print(f"--- 2. Training XGBoost on {len(X_train)} days ---")
    model = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=5, 
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    precision = precision_score(y_test, preds)
    print(f"--- Model Precision: {precision:.2f} ---")
    
    # Save Model
    joblib.dump(model, 'model.pkl')
    print("--- Model Saved to model.pkl ---")

def parse_args():
    parser = argparse.ArgumentParser(description="Train the crypto swing strategist model.")
    parser.add_argument("--ticker", default=DEFAULT_TICKER, help="Ticker symbol to fetch data for.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args.ticker)
