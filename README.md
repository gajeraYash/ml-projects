# Traditional Machine Learning Projects

## Description
This repository contains various traditional machine learning projects implemented using Python and relevant libraries such as `scikit-learn`, `pandas`, `numpy`, `matplotlib`, and `PyTorch`. The projects focus on solving real-world problems using classical machine learning techniques like regression, classification, and time series forecasting.

---

## **Environment Setup**
All dependencies required to run the projects are listed in **`requirements.txt`**.

### **Installation Instructions**
1. **Clone the repository**:
   ```
   git clone https://github.com/gajeraYash/ml-projects.git
   cd ml-projects
   ```

2. **Create a virtual environment** (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Mac/Linux
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

---

## **Projects**

### **1. Stock Price Prediction**
#### **Overview**
This project predicts future stock prices using **Linear Regression** with `PyTorch`. The model is trained on historical stock data and makes **future predictions** based on learned patterns.

#### **Features**
✅ Fetches real-time stock data from **Yahoo Finance (`yfinance`)**.  
✅ Uses **Min-Max Scaling** for data normalization.  
✅ Implements **Linear Regression in PyTorch** for forecasting.  
✅ Saves the trained model for future use.  
✅ **Separates training (`main.py`) and prediction (`predict.py`)** for modularity.  
✅ **Plots actual vs. predicted stock prices** for easy comparison.

#### **Files**
- `preprocessor.py` – Handles **data fetching & preprocessing**.
- `model.py` – Defines **PyTorch Linear Regression model**.
- `train.py` – Trains & saves the **stock price prediction model**.
- `main.py` – **Runs training & saves model** under the stock ticker name.
- `predict.py` – Loads **trained model**, predicts **future stock prices**, and compares them with actual values.

#### **How to Run**
1. **Train the model** on past stock data (e.g., 2024):
   ```
   python main.py
   ```
   This saves the trained model as `{TICKER}_model.pth` (e.g., `AAPL_model.pth`).

2. **Predict & Compare Future Prices (e.g., 2025)**:
   ```
   python predict.py
   ```
   This will **load the trained model, predict 2025 prices, and compare them with actual stock prices**.

---

## **License**
This repository is licensed under the **MIT License**.

---
