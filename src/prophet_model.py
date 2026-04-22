from prophet import Prophet
import pandas as pd
import streamlit as st


def run_prophet(df, price_col, forecast_horizon, confidence_level):

    prophet_df = pd.DataFrame({"ds": df.index, "y": df[price_col].values})

    # split into train and test for plotting actual future data points
    train_prophet_df = prophet_df[:-forecast_horizon]
    test_prophet_df = prophet_df[-forecast_horizon:]

    model = Prophet(interval_width=confidence_level)
    model.fit(train_prophet_df)

    future = model.make_future_dataframe(periods=len(test_prophet_df), freq="D")

    forecast = model.predict(future)

    return {
        "ds_train": train_prophet_df["ds"],
        "y_train": train_prophet_df["y"],
        "yhat_train_set": forecast["yhat"][: len(train_prophet_df)],
        "ds": test_prophet_df["ds"],
        "y_test": test_prophet_df["y"],
        "yhat": forecast["yhat"][-len(test_prophet_df) :],
        "yhat_upper": forecast["yhat_upper"][-len(test_prophet_df) :],
        "yhat_lower": forecast["yhat_lower"][-len(test_prophet_df) :],
    }
