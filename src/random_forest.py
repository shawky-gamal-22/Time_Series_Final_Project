import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from scipy import stats


def add_lags(df, lags, target_col: str = "y"):
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    return df.dropna()


def add_rolling_windows(df, window_size: int, type_roll: str, target_col: str = "y"):
    if type_roll == "mean":
        df[f"rolling_mean_{window_size}"] = (
            df[target_col].rolling(window=window_size).mean()
        )
    if type_roll == "std":
        df[f"rolling_std_{window_size}"] = (
            df[target_col].rolling(window=window_size).std()
        )

    return df.dropna()


def apply_random_forest(series, forecast_horizon, confidence_level):

    st.write("Fitting Random Forest model...")

    # Time Embeddings
    df = pd.DataFrame({"ds": series.index, "y": series.values})
    df["dayofweek"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    df["dayofyear"] = df["ds"].dt.dayofyear
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # lags and Roling Windows
    df = add_lags(df, lags=[1, 3, 7], target_col="y")
    df = add_rolling_windows(df, window_size=7, type_roll="mean", target_col="y")
    df = add_rolling_windows(df, window_size=7, type_roll="std", target_col="y")

    # split into train and test
    full_copy = df.copy()
    df_train = full_copy[:-forecast_horizon]
    df_test = full_copy[-forecast_horizon:]

    # prepare training data
    x_train = df_train.drop(columns=["ds", "y"])
    y_train = df_train["y"]
    x_test = df_test.drop(columns=["ds", "y"])
    y_test = df_test["y"]

    # linear model for trend
    trend_model = LinearRegression()
    trend_model.fit(x_train, y_train)

    y_pred_trend = trend_model.predict(x_train)

    y_train_detrended = y_train - y_pred_trend

    # Random Forest for residuals
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(x_train, y_train_detrended)

    y_pred_over_all = y_pred_trend + rf_model.predict(x_train)

    #  testing the model on the test set
    trend_pred_test = trend_model.predict(x_test)
    rf_pred_test = rf_model.predict(x_test)
    y_pred_test = trend_pred_test + rf_pred_test

    # create forecast dataframe
    confidence_level = confidence_level if confidence_level else 0.95
    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)

    forcast = {
        # test set predictions and actuals
        "ds": df_test["ds"],
        "yhat": y_pred_test,
        "yhat_lower": y_pred_test - z_score * np.std(y_test - y_pred_test),
        "yhat_upper": y_pred_test + z_score * np.std(y_test - y_pred_test),
        "yhat_train_set": y_pred_over_all,
        "y_test": y_test,
        # training set actuals for plotting
        "y_train": y_train,
        "ds_train": df_train["ds"],
    }

    return forcast
