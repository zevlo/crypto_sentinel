import joblib
import pandas as pd
import sys
from datetime import datetime, timezone
from data_loader import fetch_data

# --- CONFIG ---
CONFIDENCE_THRESHOLD = 0.65

# --- COLORS (ANSI Escape Codes) ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_market_context(current_row):
    """Analyzes the current row to generate context strings."""
    price = current_row['Close'].values[0]
    sma200 = current_row['SMA_200'].values[0]
    bb_width = current_row['BB_Width'].values[0]
    rsi = current_row['RSI'].values[0]

    # Trend Context
    if price > sma200:
        trend_str = f"{Colors.GREEN}BULLISH (Price > 200 SMA){Colors.ENDC}"
        is_bullish = True
    else:
        trend_str = f"{Colors.RED}BEARISH (Price < 200 SMA){Colors.ENDC}"
        is_bullish = False

    # Volatility Context (Bollinger Band Width)
    if bb_width < 0.10:
        vol_str = "LOW (Squeeze Detected)"
    elif bb_width > 0.30:
        vol_str = "HIGH (Volatile)"
    else:
        vol_str = "NORMAL"

    return price, rsi, trend_str, vol_str, is_bullish

def generate_reasoning(prob, rsi, is_bullish, bb_width):
    """Generates a dynamic explanation based on indicators."""
    reasons = []
    
    # 1. Model Confidence Logic
    if prob > CONFIDENCE_THRESHOLD:
        reasons.append("Model detects high-probability upside patterns.")
    elif prob < 0.40:
        reasons.append("Model sees weak structure/downside risk.")
    else:
        reasons.append("Model is indecisive (probabilities are mixed).")

    # 2. Technical Logic
    if is_bullish and rsi > 70:
        reasons.append("CAUTION: Trend is up, but RSI is Overbought (Pullback risk).")
    elif is_bullish and rsi < 40:
        reasons.append("OPPORTUNITY: Bull trend with oversold RSI (Dip Buy).")
    elif not is_bullish and rsi < 30:
        reasons.append("Market is in a downtrend and oversold.")
    
    if bb_width < 0.10:
        reasons.append("Volatility squeeze active; expect a violent move soon.")

    return " ".join(reasons)

def run_dashboard():
    # 1. Load Model
    try:
        model = joblib.load('model.pkl')
    except FileNotFoundError:
        print(f"{Colors.RED}Error: model.pkl not found. Run 'python train.py' first.{Colors.ENDC}")
        return

    # 2. Fetch Data
    # print("Fetching market data...") # Commented out to keep dashboard clean
    df = fetch_data()
    current_data = df.iloc[[-1]].copy()
    
    # Features must match training
    features = [
        'ATR', 'SMA_50', 'SMA_200', 'RSI', 'BB_Width', 
        'Volume', 'Open', 'High', 'Low', 'Close'
    ]
    
    # 3. Predict
    try:
        probs = model.predict_proba(current_data[features])[0]
        bullish_prob = probs[1]
    except Exception as e:
        print(f"Prediction Error: {e}")
        return

    # 4. Prepare Dashboard Data
    price, rsi, trend_str, vol_str, is_bullish = get_market_context(current_data)
    
    # Determine Signal
    if bullish_prob > CONFIDENCE_THRESHOLD:
        direction = f"{Colors.GREEN}LONG / BUY{Colors.ENDC} 🟢"
    elif bullish_prob < 0.40:
        direction = f"{Colors.RED}CASH / AVOID{Colors.ENDC} 🔴"
    else:
        direction = f"{Colors.YELLOW}HOLD / NEUTRAL{Colors.ENDC} ⚪"

    reasoning = generate_reasoning(bullish_prob, rsi, is_bullish, current_data['BB_Width'].values[0])
    
    # Format Reasoning for width (simple wrap)
    import textwrap
    wrapped_reasoning = textwrap.fill(reasoning, width=45, initial_indent='', subsequent_indent='             ')

    # Timestamps
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # 5. RENDER DASHBOARD
    print("\n" + "="*47)
    print(f"   {Colors.BOLD}BTC SWING STRATEGIST (Daily){Colors.ENDC}")
    print("="*47)
    print(f"LAST UPDATE: {now_utc}")
    print(f"NEXT UPDATE: Next Daily Close")
    print("-" * 47)
    
    print(f"{Colors.BOLD}--- MARKET CONTEXT ----------------------------{Colors.ENDC}")
    print(f"PRICE:       ${price:,.2f}")
    print(f"TREND (SMA): {trend_str}")
    print(f"VOLATILITY:  {vol_str}")
    print("-" * 47)

    print(f"{Colors.BOLD}--- FORECAST FOR TODAY ------------------------{Colors.ENDC}")
    print(f"DIRECTION:   {direction}")
    print(f"CONFIDENCE:  {bullish_prob*100:.1f}%")
    print("")
    print(f"REASONING:   {wrapped_reasoning}")
    print("-" * 47)

    # Placeholder for Portfolio (Since we don't have a database yet)
    print(f"{Colors.BOLD}--- RECENT PERFORMANCE (Last 30 Days) ---------{Colors.ENDC}")
    print(f"TRADES TAKEN: 0 (Simulation Mode)")
    print(f"WIN RATE:     N/A")
    print(f"P&L:          $0.00")
    print("="*47 + "\n")

if __name__ == "__main__":
    run_dashboard()