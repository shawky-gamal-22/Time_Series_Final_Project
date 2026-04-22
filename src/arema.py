from statsmodels.tsa.stattools import adfuller
from statsforecast.models import AutoARIMA
from statsforecast import StatsForecast
from statsforecast.arima import arima_string
import streamlit as st
import numpy as np
import pandas as pd


def test_stationary(series):
    result = adfuller(series)
    st.write("ADF Statistic: %f" % result[0])
    st.write("p-value: %f" % result[1])
    st.write("Critical Values:")
    for key, value in result[4].items():
        st.write("\t%s: %.3f" % (key, value))

    if result[1] < 0.05 and result[0] < result[4]["5%"]:
        st.write("The series is stationary. (Reject H0)")
        return True
    else:
        st.write("The series is not stationary. (Fail to reject H0)")
        return False


def make_stationary(series):
    st.write("Applying log transformation to stabilize variance...")
    log_transformed = np.log(series.replace(0, np.nan)).dropna()
    st.write("Applying first-order differencing to make the series stationary...")
    log_differenced = log_transformed.diff().dropna()
    return log_differenced


def inverse_stationary(forecast_diff, original_log_series):

    last_log_value = original_log_series.iloc[-1]

    inverted_diff = last_log_value + forecast_diff

    final_forecast = np.exp(inverted_diff)

    return final_forecast


def apply_arima(series, forecast_horizon, confidence_level):
    st.write("Fitting AutoARIMA model...")

    log_reference_full = np.log(series.replace(0, np.nan)).dropna()

    is_stationary = test_stationary(series)
    if not is_stationary:
        stationary_series = make_stationary(series)
    else:
        stationary_series = series

    train_set = stationary_series[:-forecast_horizon]
    test_set = stationary_series[-forecast_horizon:]

    log_ref_train = log_reference_full[: len(train_set)]

    train_df = pd.DataFrame(
        {"ds": train_set.index, "y": train_set.values, "unique_id": "btc"}
    )

    model = AutoARIMA(seasonal=False)
    sf = StatsForecast(models=[model], freq="D")
    sf.fit(train_df)

    level_val = int(confidence_level * 100)
    forecast = sf.predict(h=forecast_horizon, level=[level_val])

    low_col = f"AutoARIMA-lo-{level_val}"
    hi_col = f"AutoARIMA-hi-{level_val}"
    forecast = forecast.rename(
        columns={"AutoARIMA": "yhat", low_col: "yhat_lower", hi_col: "yhat_upper"}
    )

    yhat_usd = inverse_stationary(forecast["yhat"], log_ref_train)
    yhat_lower_usd = inverse_stationary(forecast["yhat_lower"], log_ref_train)
    yhat_upper_usd = inverse_stationary(forecast["yhat_upper"], log_ref_train)

    y_test_usd = inverse_stationary(test_set, log_ref_train)

    return {
        "ds_train": train_set.index,
        "y_train": np.exp(log_ref_train),
        "yhat_train_set": np.exp(log_ref_train),
        "ds": pd.Series(test_set.index),
        "y_test": y_test_usd,
        "yhat": yhat_usd,
        "yhat_upper": yhat_upper_usd,
        "yhat_lower": yhat_lower_usd,
    }
