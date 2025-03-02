import torch
from preprocessor import load_data, preprocess_data
from model import StockPredictor
from train import train_model, plot_results

# Select device (MPS for Apple Silicon, CUDA for Nvidia, else CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using torch.device: {device}")

# Fetch and preprocess data for 2024
stock_ticker = "AAPL"
prices = load_data(stock_ticker, start="2024-01-01", end="2024-12-31")
train_features, train_labels, val_features, val_labels, features_min, features_max, labels_min, labels_max = preprocess_data(prices, device)

# Initialize model and move to device
model = StockPredictor().to(device)

# Train model and save it
train_losses, val_accuracies = train_model(model, train_features, train_labels, val_features, val_labels, device, stock_ticker)

# Plot results
plot_results(train_losses, val_accuracies, model, train_features, train_labels, val_features, val_labels, features_min, features_max, labels_min, labels_max, stock_ticker)