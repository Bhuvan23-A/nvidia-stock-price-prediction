import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("📈 NVIDIA Stock Price Predictor")
st.markdown("A Random Forest model to predict NVIDIA stock closing prices.")

# ── Load Data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    col_names = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    df = pd.read_csv("NVIDIA_STOCK.csv", skiprows=2, names=col_names)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.set_index('Date', inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)
    df.ffill(inplace=True)
    return df

df = load_data()

st.subheader("Raw Data (Last 5 Rows)")
st.dataframe(df.tail())

st.subheader("Missing Values")
st.write(df.isnull().sum())

# ── Feature Engineering ────────────────────────────────────────────────────────
@st.cache_data
def engineer_features(df):
    df = df.copy()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=10).std()
    df.dropna(inplace=True)
    return df

df = engineer_features(df)

st.subheader("Data After Feature Engineering (Last 5 Rows)")
st.dataframe(df.tail())

# ── Train / Test Split ─────────────────────────────────────────────────────────
features = ['Open', 'High', 'Low', 'Volume', 'MA10', 'MA50', 'Daily_Return', 'Volatility']
target = 'Close'

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.write(f"**Training Set:** {X_train.shape[0]} samples")
st.write(f"**Testing Set:** {X_test.shape[0]} samples")

# ── Train Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

with st.spinner("Training model... please wait ⏳"):
    model = train_model(X_train, y_train)

st.success("Model trained successfully!")

# ── Predictions & Metrics ──────────────────────────────────────────────────────
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

st.subheader("📊 Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("MAE",    f"{mae:.2f}")
col2.metric("RMSE",   f"{rmse:.2f}")
col3.metric("R² Score", f"{r2:.4f}")

# ── Plot 1: Actual vs Predicted Line Chart ─────────────────────────────────────
st.subheader("Actual vs Predicted Stock Prices")
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(y_test.values, label='Actual Prices',    color='blue')
ax1.plot(y_pred,        label='Predicted Prices', color='red', linestyle='dashed')
ax1.set_title('NVIDIA Stock Price Prediction')
ax1.set_xlabel('Time')
ax1.set_ylabel('Stock Price (USD)')
ax1.legend()
st.pyplot(fig1)

# ── Plot 2: Scatter Plot ───────────────────────────────────────────────────────
st.subheader("Actual vs Predicted — Scatter Plot")
fig2, ax2 = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, ax=ax2)
ax2.set_xlabel("Actual Prices")
ax2.set_ylabel("Predicted Prices")
ax2.set_title("Actual vs. Predicted Stock Prices")
st.pyplot(fig2)