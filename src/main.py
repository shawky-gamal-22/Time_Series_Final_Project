import streamlit as st
import pandas as pd

# my own files and imports
from ploting import plot_with_plotly
from metrics import calculate_metrics
from prophet_model import run_prophet
from arema import apply_arima
from random_forest import apply_random_forest
from clean_big_data import clean_big_data

st.set_page_config(page_title="BTC Forecasting Tool", layout="wide")

st.title("📈 Bitcoin Time-Series Analysis Tool")
st.markdown("Upload your Bitcoin historical data.")

# ------------------------ File Upload & Data Validation ------------------------

uploaded_file = st.file_uploader("Upload BTC Historical CSV", type=["csv"])

if uploaded_file is not None:
    try:

        df = clean_big_data(uploaded_file)

        st.success("File uploaded successfully!")
        st.subheader("Raw Data Preview")
        st.write(df.head())

        col1, col2 = st.columns(2)

        with col1:
            date_col = st.selectbox("Select Timestamp Column:", df.columns)

        with col2:
            price_col = st.selectbox(
                "Select Price Value (Target):", df.columns, index=4
            )

        if date_col != "Open time":
            st.error("Please select the correct timestamp column ('Open time').")
            st.stop()

        new_df = df[[date_col, price_col]].copy()

        new_df[date_col] = pd.to_datetime(new_df[date_col])

        new_df = new_df.sort_values(by=date_col)

        st.subheader("Data Validation & Cleaning")

        df = new_df.set_index(date_col)

        missing_values = df[price_col].isnull().sum()
        if missing_values > 0:
            st.warning(
                f"Found {missing_values} missing values. Filling them using linear interpolation..."
            )
            df[price_col] = df[price_col].interpolate(method="linear")
        else:
            st.info("No missing values detected.")

        st.write("Processed Data (Tail):")
        st.write(df[[price_col]].tail())

        st.session_state["df"] = df
        st.session_state["price_col"] = price_col

    except Exception as e:
        st.error(f"Error parsing file: {e}")
else:
    st.info("Waiting for CSV file upload...")

# ------------------------ End File Upload & Data Validation ------------------------

# ------------------------ Sidebar for Forecasting Settings ------------------------
st.sidebar.header("🛠️ Forecasting Settings")

model_choice = st.sidebar.selectbox(
    "Select Model:", ["Prophet", "ARIMA", "Random Forest Regressor"]
)

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (Days):", min_value=7, max_value=90, value=30, step=7
)

confidence_level = st.sidebar.select_slider(
    "Confidence Interval:",
    options=[0.80, 0.90, 0.95],
    value=0.95,
    help="Adjust the uncertainty bands for the forecast.",
)

st.sidebar.subheader("Technical Indicators")
show_sma = st.sidebar.checkbox("Show Simple Moving Average (SMA)")
show_ema = st.sidebar.checkbox("Show Exponential Moving Average (EMA)")
sma_period = None
if show_sma:
    sma_period = st.sidebar.number_input(
        "SMA | EMA Period", min_value=5, max_value=100, value=20
    )

generate_btn = st.sidebar.button("🚀 Generate Forecast")

st.info(
    f"**Target Model:** {model_choice} | **Horizon:** {forecast_horizon} Days | **Confidence:** {int(confidence_level*100)}%"
)


# ------------------------ End Sidebar for Forecasting Settings ------------------------

# ------------------------ Forecast Generation & Visualization -------------------------
if generate_btn and "df" not in st.session_state:
    st.error("Please upload and process your data before generating a forecast.")

if generate_btn and "df" in st.session_state and model_choice == "Prophet":

    df_plot = st.session_state["df"].copy()
    target = st.session_state["price_col"]

    st.subheader("🔮 Generating Forecast with Prophet...")
    forecast = run_prophet(df_plot, target, forecast_horizon, confidence_level)

    plot_with_plotly(df_plot, forecast, target, show_sma, show_ema, sma_period)
    calculate_metrics(forecast["y_test"], forecast["yhat"])

elif generate_btn and "df" in st.session_state and model_choice == "ARIMA":

    df_plot = st.session_state["df"].copy()
    target = st.session_state["price_col"]

    st.subheader("🔮 Generating Forecast with ARIMA...")
    forecast = apply_arima(df_plot[target], forecast_horizon, confidence_level)

    plot_with_plotly(df_plot, forecast, target, show_sma, show_ema, sma_period)
    calculate_metrics(forecast["y_test"], forecast["yhat"])

elif (
    generate_btn
    and "df" in st.session_state
    and model_choice == "Random Forest Regressor"
):

    df_plot = st.session_state["df"].copy()
    target = st.session_state["price_col"]

    st.subheader("🔮 Generating Forecast with Random Forest Regressor...")
    forecast = apply_random_forest(df_plot[target], forecast_horizon, confidence_level)

    plot_with_plotly(df_plot, forecast, target, show_sma, show_ema, sma_period)
    calculate_metrics(forecast["y_test"], forecast["yhat"])
