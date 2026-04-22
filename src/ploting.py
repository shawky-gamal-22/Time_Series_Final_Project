import pandas as pd

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_with_plotly(
    df, forecast, target_col, show_sma=False, show_ema=False, sma_period=20
):

    fig = make_subplots(rows=1, cols=1)

    # Actual data
    fig.add_trace(
        go.Scatter(
            x=forecast["ds_train"],
            y=forecast["y_train"],
            mode="lines",
            name="Actual Training",
            line=dict(color="red"),
        )
    )

    # Forecasted training data
    fig.add_trace(
        go.Scatter(
            x=forecast["ds_train"],
            y=forecast["yhat_train_set"],
            mode="lines",
            name="Forecasted Training",
            line=dict(color="orange"),
        )
    )

    # line for the last actual data point
    fig.add_trace(
        go.Scatter(
            x=[forecast["ds"].iloc[0], forecast["ds"].iloc[0]],
            y=[forecast["y_train"].min(), forecast["y_train"].max() + 90000],
            line_width=2,
            mode="lines",
            name="Last Actual",
            line=dict(color="green", dash="dot"),
        )
    )

    # Future forecast
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            mode="lines",
            name="Forecasted Future",
            line=dict(color="blue"),
        )
    )

    # actual future data points
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["y_test"],
            mode="lines",
            name="Actual Future",
            line=dict(color="black", width=2),
        )
    )
    # Confidence intervals
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_upper"],
            mode="lines",
            name="Upper Confidence",
            line=dict(color="lightgray"),
            fill=None,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_lower"],
            mode="lines",
            name="Lower Confidence",
            line=dict(color="lightgray"),
            fill="tonexty",
        )
    )

    # SMA
    if show_sma:
        sma = pd.Series(forecast["yhat_train_set"]).rolling(window=sma_period).mean()
        forecast_value_future = sma.iloc[-1]
        forecast_sma_series = [forecast_value_future] * len(forecast["ds"])
        sma = pd.concat(
            [sma, pd.Series(forecast_sma_series, index=forecast["ds"])]
        ).dropna()
        fig.add_trace(
            go.Scatter(
                x=pd.concat([forecast["ds_train"], pd.Series(forecast["ds"])]),
                y=sma,
                mode="lines",
                name=f"SMA ({sma_period})",
                line=dict(color="purple", dash="dash"),
            )
        )

    # EMA
    if show_ema:
        ema = (
            pd.Series(forecast["yhat_train_set"])
            .ewm(span=sma_period, adjust=False)
            .mean()
        )
        forecast_value_future = ema.iloc[-1]
        forecast_ema_series = [forecast_value_future] * len(forecast["ds"])
        ema = pd.concat(
            [ema, pd.Series(forecast_ema_series, index=forecast["ds"])]
        ).dropna()
        fig.add_trace(
            go.Scatter(
                x=pd.concat([forecast["ds_train"], pd.Series(forecast["ds"])]),
                y=ema,
                mode="lines",
                name=f"EMA ({sma_period})",
                line=dict(color="blue", dash="dash"),
            )
        )

    fig.update_layout(
        title=dict(
            text="BTC Price Forecast with Prophet", font=dict(color="black", size=20)
        ),
        xaxis_title=dict(text="Date", font=dict(color="black", size=14)),
        yaxis_title=dict(text="Price", font=dict(color="black", size=14)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black"),
        xaxis=dict(
            showgrid=True,
            gridcolor="lightgray",
            linecolor="black",
            showline=True,
            zeroline=True,
            zerolinecolor="black",
            tickfont=dict(color="black"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="lightgray",
            linecolor="black",
            showline=True,
            zeroline=True,
            zerolinecolor="black",
            tickfont=dict(color="black"),
        ),
        legend=dict(
            bgcolor="rgba(255, 255, 255, 0.5)",
            bordercolor="black",
            borderwidth=1,
            font=dict(color="black"),
        ),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
