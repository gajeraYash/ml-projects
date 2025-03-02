import yfinance as yf
import torch
import numpy as np

def load_data(stock_ticker, start="2024-01-01", end="2024-12-31"):
    data = yf.download(stock_ticker, start=start, end=end)

    # Get 'Close' prices
    prices = data["Close"].values

    return prices

def preprocess_data(prices, device):

    # Create dataset: Use previous day's price to predict next day
    features = prices[:-1].reshape(-1, 1)  # Previous day's price
    labels = prices[1:].reshape(-1, 1)   # Next day's price

    # Normalize data (Min-Max Scaling)
    features_min, features_max = features.min(), features.max()
    labels_min, labels_max = labels.min(), labels.max()
    features = (features - features_min) / (features_max - features_min)
    labels = (labels - labels_min) / (labels_max - labels_min)

    # Split into training (80%) and validation (20%)
    split_idx = int(len(features) * 0.8)
    train_features, val_features = features[:split_idx], features[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    # Convert to PyTorch tensors
    train_features = torch.tensor(train_features, dtype=torch.float32).to(device)
    train_labels = torch.tensor(train_labels, dtype=torch.float32).to(device)
    val_features = torch.tensor(val_features, dtype=torch.float32).to(device)
    val_labels = torch.tensor(val_labels, dtype=torch.float32).to(device)

    return train_features, train_labels, val_features, val_labels, features_min, features_max, labels_min, labels_max
