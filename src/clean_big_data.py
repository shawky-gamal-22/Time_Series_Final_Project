import pandas as pd


def clean_big_data(file_path):
    df = pd.read_csv(
        file_path,
        usecols=["Open time", "Open", "High", "Low", "Close", "Volume"],
    )

    df["Open time"] = pd.to_datetime(df["Open time"])
    df = df.sort_values(by="Open time")

    df_daily = df.resample("D", on="Open time").mean().reset_index()

    return df_daily
