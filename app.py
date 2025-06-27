from predictor import fetch_gold_data, add_indicators, train_model, make_prediction
import requests

# Telegram
BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"

def send_message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Telegram Error: {e}")

def main():
    try:
        df = fetch_gold_data()
        df = add_indicators(df)
        model, scaler, df = train_model(df)
        prediction, confidence = make_prediction(model, scaler, df)
        latest_price = df["close"].iloc[-1]

        message = (
            f"📈 Gold Price Prediction\n"
            f"Current Price: ${latest_price:.2f}\n"
            f"Prediction: {prediction} ({confidence * 100:.2f}%)"
        )
        send_message(message)
    except Exception as e:
        send_message(f"❌ Error: {e}")
        print("Error:", e)

if __name__ == "__main__":
    main()
