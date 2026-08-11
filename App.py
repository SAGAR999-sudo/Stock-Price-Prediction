import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from tensorflow.keras.models import load_model


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = os.path.join(
    "model",
    "stock_model.keras"
)

SCALER_PATH = os.path.join(
    "model",
    "scaler.pkl"
)


# ============================================================
# CHECK MODEL AND SCALER
# ============================================================

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file not found: {MODEL_PATH}"
    )
    st.stop()

if not os.path.exists(SCALER_PATH):
    st.error(
        f"Scaler file not found: {SCALER_PATH}"
    )
    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_prediction_model():

    return load_model(
        MODEL_PATH
    )


# ============================================================
# LOAD SCALER
# ============================================================

@st.cache_resource
def load_prediction_scaler():

    return joblib.load(
        SCALER_PATH
    )


model = load_prediction_model()

scaler = load_prediction_scaler()


# ============================================================
# STOCK SETTINGS
# ============================================================

ticker = st.sidebar.text_input(
    "Enter Stock Ticker",
    value="AAPL"
).upper().strip()

lookback = 60


# ============================================================
# TITLE
# ============================================================

st.title(
    "📈 Stock Price Prediction"
)

st.markdown(
    """
    ### LSTM-Based Stock Price Prediction

    This application uses a trained **Long Short-Term Memory (LSTM)**
    neural network to predict the next stock closing price using the
    previous **60 days** of AAPL closing prices.
    """
)


# ============================================================
# DOWNLOAD CURRENT STOCK DATA
# ============================================================

import yfinance as yf


@st.cache_data
def download_stock_data():

    data = yf.download(
        ticker,
        period="5y",
        auto_adjust=False
    )

    return data


stock_data = download_stock_data()


# ============================================================
# CLEAN DATA
# ============================================================

if stock_data.empty:

    st.error(
        "Unable to download stock data."
    )

    st.stop()


# Handle yfinance multi-level columns
if isinstance(
    stock_data.columns,
    pd.MultiIndex
):

    stock_data.columns = (
        stock_data.columns.get_level_values(0)
    )


stock_data = stock_data.dropna(
    subset=["Close"]
)


# ============================================================
# CLOSE PRICE
# ============================================================

close_prices = stock_data[
    "Close"
].values


# Make sure Close is one-dimensional
close_prices = np.array(
    close_prices
).reshape(-1, 1)


# ============================================================
# CURRENT PRICE
# ============================================================

current_price = float(
    close_prices[-1][0]
)


# ============================================================
# DISPLAY CURRENT INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Current Price",
        f"${current_price:.2f}"
    )


with col2:

    st.metric(
        "Stock",
        ticker
    )


with col3:

    st.metric(
        "Lookback Period",
        "60 Days"
    )


# ============================================================
# HISTORICAL PRICE CHART
# ============================================================

st.subheader(
    "📊 Historical Closing Price"
)


historical_fig = go.Figure()


historical_fig.add_trace(
    go.Scatter(
        x=stock_data.index,
        y=close_prices.flatten(),
        mode="lines",
        name="AAPL Closing Price"
    )
)


historical_fig.update_layout(
    title="AAPL Historical Closing Price",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    hovermode="x unified"
)


st.plotly_chart(
    historical_fig,
    use_container_width=True
)


# ============================================================
# PREDICTION SECTION
# ============================================================

st.subheader(
    "🔮 Predict Next Closing Price"
)


st.write(
    "The model uses the previous 60 trading days "
    "to predict the next closing price."
)


predict_button = st.button(
    "🚀 Predict Next Price",
    type="primary"
)


if predict_button:

    # --------------------------------------------------------
    # Check sufficient data
    # --------------------------------------------------------

    if len(close_prices) < lookback:

        st.error(
            "Not enough historical data for prediction."
        )

        st.stop()


    # --------------------------------------------------------
    # Get last 60 days
    # --------------------------------------------------------

    last_60_days = close_prices[
        -lookback:
    ]


    # --------------------------------------------------------
    # Scale the last 60 days
    # --------------------------------------------------------

    scaled_last_60 = scaler.transform(
        last_60_days
    )


    # --------------------------------------------------------
    # Reshape for LSTM
    # --------------------------------------------------------

    X_input = scaled_last_60.reshape(
        1,
        lookback,
        1
    )


    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    prediction_scaled = model.predict(
        X_input,
        verbose=0
    )


    # --------------------------------------------------------
    # Convert prediction back to actual price
    # --------------------------------------------------------

    predicted_price = scaler.inverse_transform(
        prediction_scaled
    )[0][0]


    predicted_price = float(
        predicted_price
    )


    # --------------------------------------------------------
    # Calculate price change
    # --------------------------------------------------------

    price_difference = (
        predicted_price -
        current_price
    )


    percentage_change = (
        price_difference /
        current_price
    ) * 100


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    st.success(
        "Prediction completed successfully!"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Current Price",
            f"${current_price:.2f}"
        )


    with col2:

        st.metric(
            "Predicted Price",
            f"${predicted_price:.2f}"
        )


    with col3:

        st.metric(
            "Expected Change",
            f"{percentage_change:.2f}%"
        )


    # ========================================================
    # PREDICTION CHART
    # ========================================================

    st.subheader(
        "📈 Prediction Visualization"
    )


    recent_dates = stock_data.index[
        -lookback:
    ]

    recent_prices = close_prices[
        -lookback:
    ].flatten()


    prediction_fig = go.Figure()


    prediction_fig.add_trace(
        go.Scatter(
            x=recent_dates,
            y=recent_prices,
            mode="lines",
            name="Historical Price"
        )
    )


    # Add prediction point
    prediction_date = (
        stock_data.index[-1]
        + pd.Timedelta(days=1)
    )


    prediction_fig.add_trace(
        go.Scatter(
            x=[prediction_date],
            y=[predicted_price],
            mode="markers",
            marker=dict(
                size=12
            ),
            name="Predicted Price"
        )
    )


    prediction_fig.update_layout(
        title="Last 60 Days + Predicted Price",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified"
    )


    st.plotly_chart(
        prediction_fig,
        use_container_width=True
    )


    # ========================================================
    # PREDICTION DETAILS
    # ========================================================

    st.subheader(
        "📋 Prediction Details"
    )


    prediction_data = pd.DataFrame(
        {
            "Metric": [
                "Stock",
                "Current Price",
                "Predicted Price",
                "Price Difference",
                "Percentage Change",
                "Lookback Period"
            ],

            "Value": [
                ticker,
                f"${current_price:.2f}",
                f"${predicted_price:.2f}",
                f"${price_difference:.2f}",
                f"{percentage_change:.2f}%",
                "60 trading days"
            ]
        }
    )


    st.dataframe(
        prediction_data,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.subheader(
    "🤖 Model Information"
)


model_info = pd.DataFrame(
    {
        "Parameter": [
            "Model",
            "Algorithm",
            "Input Window",
            "Features",
            "Optimizer",
            "Loss Function"
        ],

        "Value": [
            "LSTM",
            "Long Short-Term Memory",
            "60 days",
            "Closing Price",
            "Adam",
            "Mean Squared Error"
        ]
    }
)


st.dataframe(
    model_info,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown("---")

st.warning(
    """
    **Disclaimer:** This application is for educational and
    demonstration purposes only. Stock-price predictions are
    estimates and should not be considered financial advice.
    """
)


st.caption(
    "Stock Price Prediction | LSTM | Streamlit"
)
