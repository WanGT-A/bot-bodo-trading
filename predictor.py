# predictor.py

import pandas as pd
import numpy as np
import requests
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def fetch_gold_data():
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": "XAUUSD",
        "interval": "5min",
        "apikey": os.getenv("ALPHA_VANTAGE_KEY"),
        "outputsize": "compact"
    }
    response = requests.get(url, params=params)
    data = response.json()

    if "Time Series (5min)" not in data:
        print("Error fetching data:", data)
        return None

    df = pd.DataFrame.from_dict(data["Time Series (5min)"], orient="index", dtype=float)
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

def add_indicators(df):
    df["return"] = df["close"].pct_change()
    df["sma"] = df["close"].rolling(window=5).mean()
    df["rsi"] = compute_rsi(df["close"])
    df = df.dropna()
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain = pd.Series(gain).rolling(window=period).mean()
    loss = pd.Series(loss).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def train_model(df):
    df["future_return"] = df["close"].shift(-1) / df["close"] - 1
    df["target"] = (df["future_return"] > 0).astype(int)
    df = df.dropna()

    X = df[["return", "sma", "rsi"]]
    y = df["target"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model, scaler, df

def make_prediction(model, scaler, df):
    X_latest = df[["return", "sma", "rsi"]].iloc[-1:]
    X_scaled = scaler.transform(X_latest)
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][pred]
    return "UP" if pred == 1 else "DOWN", prob
