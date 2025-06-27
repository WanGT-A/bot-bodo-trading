from flask import Flask
import time
import requests
import os
from predictor import fetch_gold_data, add_indicators, train_model, make_prediction

app = Flask(__name__)

# Your credentials
BOT_TOKEN = "8128557967:AAHYbQbzxUnmsp9wTgylMJ7BqF4tuNdfhsM"
CHAT_ID = "729778363"

def send_message(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            print("❌ Failed to send message:", response.text)
    except Exception as e:
        print(f"❌ Telegram error: {e}")

def check_gold():
    print("🔁 Checking gold...")
    try:
        print("📡 Fetching gold data from Alpha Vantage...")
        df = fetch_gold_data()
        if df is None:
            raise Exception("Failed to fetch gold data.")

        df = add_indicators(df)
        model, scaler, df = train_model(df)
        prediction, confidence = make_prediction(model, scaler, df)

        latest_price = df["close"].iloc[-1]
        message = (
            f"📈 Gold Price Prediction\n"
            f"Current Price: ${latest_price:.2f}\n"
            f"Prediction: {prediction} ({confidence * 100:.2f}%)"
        )

        send_message(BOT_TOKEN, CHAT_ID, message)
        print("✅ Prediction sent.")

    except Exception as e:
        print(f"❌ Error during prediction: {e}")

@app.route('/')
def index():
    return "✅ Gold Trading Bot is Running."

if __name__ == '__main__':
    check_gold()  # Only run once when called by GitHub Action
