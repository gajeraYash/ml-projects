import torch
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from model import StockPredictor
import os

def load_model(stock_ticker, device):
    model = StockPredictor().to(device)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_filename = os.path.join(current_dir, f"{stock_ticker}_model.th")
    try:
        model.load_state_dict(torch.load(model_filename, map_location=device))
        model.eval()
        print(f"Loaded model: {model_filename}")
    except FileNotFoundError:
        print(f"Error: Model file {model_filename} not found. Train the model first.")
        exit()

    return model

def predict_future_prices(stock_ticker, device):
    # Load trained model
    model = load_model(stock_ticker, device)

    # Fetch 2025 stock prices
    data_2025 = yf.download(stock_ticker, start="2025-01-01", end="2025-12-31")
    
    actual_prices = data_2025["Close"].values

    last_known_price = yf.download(stock_ticker, start="2024-12-30", end="2024-12-31")["Close"].values[-1]

    # **Fix: Ensure min and max are from actual 2025 prices to avoid division by zero**
    price_min, price_max = actual_prices.min(), actual_prices.max()
    
    # **Fix: Handle edge case where min and max are the same
    if price_min == price_max:
        print("Warning: price_min and price_max are equal. Adjusting scaling.")
        price_min -= 1  # Slight adjustment to avoid division by zero
        price_max += 1

    # Normalize the last known price
    last_known_price_scaled = (last_known_price - price_min) / (price_max - price_min)
    
    # Predict stock prices sequentially for 2025
    predicted_prices_scaled = []
    input_feature = torch.tensor(np.array([[last_known_price_scaled]]), dtype=torch.float32).to(device)

    for _ in range(len(actual_prices)):  
        with torch.no_grad():
            prediction_scaled = model(input_feature).cpu().numpy()
        predicted_prices_scaled.append(prediction_scaled[0][0])

        # Use predicted price as input for the next day
        input_feature = torch.tensor(np.array([[prediction_scaled[0][0]]]), dtype=torch.float32).to(device)

    # **Convert predictions back to original scale**
    predicted_prices = np.array(predicted_prices_scaled) * (price_max - price_min) + price_min

    # **Plot actual vs predicted prices**
    plt.figure(figsize=(12, 5))
    plt.plot(actual_prices, label="Actual Prices (2025)", alpha=0.7, color="blue")
    plt.plot(predicted_prices, label="Predicted Prices (2025)", linestyle="dashed", linewidth=2, color="red")
    plt.xlabel("Days")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.title(f"{stock_ticker} 2025 Stock Price Prediction vs Actual")
    plt.show()

    return actual_prices, predicted_prices
if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using torch.device: {device}")


    # Predict and compare prices
    stock_ticker = "AAPL"
    actual_prices, predicted_prices = predict_future_prices(stock_ticker, device)
