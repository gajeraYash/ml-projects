import torch.nn as nn

class StockPredictor(nn.Module):
    """Linear Regression model for stock price prediction"""
    def __init__(self):
        super(StockPredictor, self).__init__()
        self.linear = nn.Linear(1, 1)  # One input feature → One output

    def forward(self, x):
        return self.linear(x)
