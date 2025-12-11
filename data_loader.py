import yfinance as yf
import pandas as pd
import ta

def fetch_data(ticker="BTC-USD", period="max", interval="1d"):
    """Fetches data and calculates technical indicators using 'ta' library."""
    # 1. Fetch Crypto Data
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    
    # Fix for MultiIndex columns (yfinance specific)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    # Ensure we have valid data
    if df.empty:
        raise ValueError("No data fetched. Check your internet or ticker symbol.")

    # 2. Feature Engineering (Using 'ta' library)
    
    # Trend: SMA 50 and 200
    df['SMA_50'] = ta.trend.SMAIndicator(close=df['Close'], window=50).sma_indicator()
    df['SMA_200'] = ta.trend.SMAIndicator(close=df['Close'], window=200).sma_indicator()
    
    # Momentum: RSI
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    
    # Volatility: ATR (Average True Range)
    df['ATR'] = ta.volatility.AverageTrueRange(
        high=df['High'], low=df['Low'], close=df['Close'], window=14
    ).average_true_range()
    
    # Volatility: Bollinger Bands Width
    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['BB_Mid'] = bb.bollinger_mavg()
    # Width = (High - Low) / Mid
    df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['BB_Mid']
    
    # 3. Clean up (Drop rows with NaN created by indicators)
    df.dropna(inplace=True)
    
    return df

def prepare_targets(df, target_pct=0.025):
    """
    Creates the Target column.
    Target = 1 if (Tomorrow's High - Today's Close) / Today's Close > 2.5%
    """
    # Shift 'High' back by 1 to get "Tomorrow's High" on "Today's row"
    df['Next_High'] = df['High'].shift(-1)
    
    # Calculate potential max return for the next day
    df['Next_Day_Return'] = (df['Next_High'] - df['Close']) / df['Close']
    
    # Binary Target: Did it swing 2.5%?
    df['Target'] = (df['Next_Day_Return'] > target_pct).astype(int)
    
    # Drop the last row (NaN because we don't know tomorrow yet)
    df.dropna(inplace=True)
    
    return df

if __name__ == "__main__":
    # Test run
    try:
        data = fetch_data()
        data = prepare_targets(data)
        print("Data loaded successfully!")
        print(data.tail())
    except Exception as e:
        print(f"Error: {e}")
