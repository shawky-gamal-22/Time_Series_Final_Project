# 📈 Bitcoin Financial Time-Series Forecaster

An interactive web application built with Streamlit for analyzing and forecasting Bitcoin (BTC) price trends. This tool is designed to handle Kaggle-style historical datasets and provides a comparative analysis using multiple forecasting algorithms.

## 🚀 Features

-   **Flexible Data Ingestion**: Supports standard Kaggle BTC historical CSV files with automatic parsing.
-   **Dynamic Configuration**: 
    -   Select between multiple models: **Prophet**, **AutoARIMA**, and **Random Forest Regressor**.
    -   Adjustable Forecast Horizon (7, 30, or 90 days).
    -   Customizable Confidence Intervals (80%, 90%, 95%).
-   **Forecasting Engine**:
    -   Automatic data stationarity checks using the **Augmented Dickey-Fuller (ADF) Test**.
    -   Log transformation and Differencing for non-stationary data.
    -   Automated backtesting with performance metrics (**MAE** & **RMSE** in USD terms).
-   **Interactive Visualizations**:
    -   High-fidelity Plotly charts showing historical trends vs. projected forecasts.
    -   Uncertainty zones representing confidence intervals.
    -   Technical indicators (SMA/EMA) toggles for market context.

## 🛠️ Tech Stack

-   **UI**: [Streamlit](https://streamlit.io/)
-   **Visualization**: [Plotly](https://plotly.com/python/)
-   **Forecasting Models**: 
    -   [Prophet](https://facebook.github.io/prophet/) (Facebook's time-series model)
    -   [StatsForecast](https://nixtla.github.io/statsforecast/) (AutoARIMA)
    -   [Scikit-learn](https://scikit-learn.org/) (Random Forest)
-   **Data Processing**: Pandas, NumPy, Scipy, Statsmodels.

## 📋 Setup & Installation

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/your-username/btc-forecaster.git](https://github.com/your-username/btc-forecaster.git)
    cd btc-forecaster
    ```

2.  **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Handle Large Datasets**:
    If your dataset is larger than 200MB, create a `.streamlit/config.toml` file in your project directory:
    ```toml
    [server]
    maxUploadSize = 600
    ```

5.  **Run the Application**:
    ```bash
    streamlit run app.py
    ```

## 📉 Methodology: Handling Volatility

The Bitcoin market is characterized by extreme volatility and non-linear trends. To ensure forecast accuracy, this application implements several strategies:

1.  **Stationarity Handling**: Since BTC prices are non-stationary, the **ARIMA** engine applies log-transformation to stabilize variance and first-order differencing to remove trends before training.
2.  **Inverse Scaling**: Forecasts are mathematically reversed (Cumulative Sum + Exponential) to return predictions to their original USD scale accurately.
3.  **Hybrid Modeling (Random Forest)**: For the Machine Learning Regressor, we use a Linear Model to capture the global trend and Random Forest to model the residual volatility and seasonality.
4.  **Uncertainty Modeling**: We use dynamic Z-scores (e.g., 1.96 for 95% CI) based on the standard deviation of historical residuals to visualize the uncertainty zone.

## 📊 Dataset Link
Testing was performed using the [Bitcoin Historical Data (2014-2024) with name BTCUSD_1m_Binance](https://www.kaggle.com/datasets/imranbukhari/comprehensive-btcusd-1m-data) dataset from Kaggle.
