import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.functional import r2_score
import os

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

def train_model(model, train_features, train_labels, val_features, val_labels, device, stock_ticker, epochs=500, lr=0.01):
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()

        # Forward pass
        predictions = model(train_features.to(device))
        loss = criterion(predictions, train_labels.to(device))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store training loss
        train_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(val_features.to(device))
            val_acc = r2_score(val_labels.cpu(), val_predictions.cpu())  # R² Score
            val_accuracies.append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}, Val R²: {val_acc:.4f}")

    # Save the trained model in the current directory of this file execution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_filename = os.path.join(current_dir, f"{stock_ticker}_model.th")
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as {model_filename}")

    return train_losses, val_accuracies

def plot_results(train_losses, val_accuracies, model, train_features, train_labels, val_features, val_labels, features_min, features_max, labels_min, labels_max, stock_ticker):
    
    with torch.no_grad():
        train_predictions = model(train_features).cpu().numpy() * (labels_max - labels_min) + labels_min
        val_predictions = model(val_features).cpu().numpy() * (labels_max - labels_min) + labels_min
        train_labels = train_labels.cpu().numpy() * (labels_max - labels_min) + labels_min
        val_labels = val_labels.cpu().numpy() * (labels_max - labels_min) + labels_min
        train_features = train_features.cpu().numpy() * (features_max - features_min) + features_min
        val_features = val_features.cpu().numpy() * (features_max - features_min) + features_min

    # Plot Training Loss
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.show()

    # Plot Validation Accuracy
    plt.figure(figsize=(10, 4))
    plt.plot(val_accuracies, label="Validation R² Score", color="green")
    plt.xlabel("Epochs")
    plt.ylabel("R² Score")
    plt.title("Validation Accuracy (R²) over Epochs")
    plt.legend()
    plt.show()

    # Plot Actual vs Predicted Prices
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(train_labels)), train_labels, label="Actual (Train)", alpha=0.6)
    plt.plot(range(len(train_predictions)), train_predictions, label="Predicted (Train)", linestyle="dashed", color="red")
    plt.plot(range(len(train_labels), len(train_labels) + len(val_labels)), val_labels, label="Actual (Val)", alpha=0.6, color="orange")
    plt.plot(range(len(train_labels), len(train_labels) + len(val_predictions)), val_predictions, label="Predicted (Val)", linestyle="dashed", color="purple")
    plt.xlabel("Days")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.title(f"{stock_ticker} Stock Price Prediction using PyTorch Linear Regression")
    plt.show()
