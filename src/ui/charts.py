import pandas as pd
import plotly.express as px


def _resolve_close_column(df: pd.DataFrame):
    close_col_name = "Close"
    if isinstance(df.columns, pd.MultiIndex):
        for col in df.columns:
            if col[0] == "Close":
                close_col_name = col
                break
    return close_col_name


def build_price_chart(df: pd.DataFrame, ticker: str):
    close_col_name = _resolve_close_column(df)
    y_data = df[close_col_name]
    if hasattr(y_data, "values") and len(y_data) == len(df.index):
        return px.line(x=df.index, y=y_data, title=f"{ticker} Closing Price"), y_data
    return px.line(df, x=df.index, y="Close", title=f"{ticker} Closing Price"), df["Close"]


def build_anomaly_chart(df: pd.DataFrame, pts: pd.DataFrame, y_data):
    scatter_close_col = _resolve_close_column(pts)
    y_pts = pts[scatter_close_col]
    if hasattr(y_pts, "values") and len(y_pts) == len(pts.index):
        fig_final = px.scatter(
            x=pts.index, y=y_pts, color=pts["Method"], title="Anomalies Detected"
        )
    else:
        fig_final = px.scatter(
            pts, x=pts.index, y="Close", color="Method", title="Anomalies Detected"
        )
    fig_final.add_scatter(x=df.index, y=y_data, mode="lines", name="Price", opacity=0.3)
    return fig_final
