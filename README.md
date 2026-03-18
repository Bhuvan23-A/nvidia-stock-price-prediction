# NVIDIA Stock Price Prediction using Random Forest

This project builds an end-to-end machine learning pipeline to predict **NVIDIA (NVDA) stock closing prices** using historical stock market data. It includes data cleaning, feature engineering, model training with a Random Forest Regressor, and evaluation using regression metrics and visualizations.

## 🚀 Live Demo
👉 [Click here to view the deployed Streamlit app](https://nvidia-stock-price-prediction-b2zghcwkhtluvzyetqdtqy.streamlit.app/)

---

## 📊 Dataset
- Source file: `NVIDIA_STOCK.csv`
- Columns used:
  - `Date`
  - `Adj Close`
  - `Close`
  - `High`
  - `Low`
  - `Open`
  - `Volume`
- Date range (after cleaning): **2018-01-02 to 2024-09-30**
- Total records after preprocessing: **1697 rows**

Preprocessing steps:
- Skipped extra CSV headers using `skiprows`
- Parsed `Date` as a datetime column and set it as the DataFrame index
- Converted all price/volume columns to numeric
- Dropped or forward-filled missing values

---

## 🧩 Feature Engineering
The following features are created from the raw stock data:
- **MA10** – 10-day Moving Average of `Close`
- **MA50** – 50-day Moving Average of `Close`
- **Daily_Return** – Percentage change of the closing price (`Close.pct_change()`)
- **Volatility** – 10-day rolling standard deviation of `Daily_Return`

After feature creation, rows with NaN values are dropped.

**Features used for training:**
- `Open`
- `High`
- `Low`
- `Volume`
- `MA10`
- `MA50`
- `Daily_Return`
- `Volatility`

**Target variable:**
- `Close` (NVIDIA closing price)

---

## 🧠 Model
The project uses a **RandomForestRegressor** from Scikit-learn.
- Algorithm: `RandomForestRegressor`
- Number of trees (estimators): `n_estimators = 100`
- `random_state = 42`
- Train–test split: `80%` training, `20%` testing
  - Training samples: **1318**
  - Testing samples: **330**

---

## 📈 Results
The model performance on the test set:

| Metric | Score |
|--------|-------|
| Mean Absolute Error (MAE) | `0.26` |
| Root Mean Squared Error (RMSE) | `0.50` |
| R² Score | `0.9997` |

A higher R² (close to 1) indicates that the model explains almost all the variance in the target (closing price).

---

## 📉 Visualizations
The notebook generates:
1. **Actual vs Predicted Line Plot**
   - Compares true closing prices with model predictions over time.
2. **Actual vs Predicted Scatter Plot**
   - Shows how closely predicted values align with actual prices.

These plots help visually inspect the quality of predictions.

---

## 🛠️ Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Jupyter Notebook / Google Colab

---

## 📂 Project Structure
```text
nvidia-stock-price-predictor/
│
├── Nvidiastockpricepredictor.ipynb   # Main notebook
├── app.py                            # Streamlit web app
├── NVIDIA_STOCK.csv                  # Raw historical stock data
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```
