import streamlit as st

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    st.subheader("Forecast Performance Metrics")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f} USD")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f} USD")
    return mae, rmse
