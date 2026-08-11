import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

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
# STOCK SELECTION
# ============================================================

st.sidebar.header("📊 Stock Selection")

ticker = st.sidebar.text_input(
    "Enter Stock Ticker",
    value="AAPL",
    placeholder="Example: AAPL"
).upper().strip()

st.sidebar.caption(
    "Examples: AAPL, TSLA, MSFT, GOOGL, AMZN"
)


# ============================================================
# MODEL FILE PATHS
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


try:

    model = load_prediction_model()

    scaler = load_prediction_scaler()

except Exception as e:

    st.error(
        f"Error loading model or scaler: {e}"
    )

    st.stop()


# ============================================================
# MODEL SETTINGS
# ============================================================

lookback = 60


# ============================================================
# TITLE
# ============================================================

st.title(
    "📈 Stock Price Prediction Using LSTM"
)

st.markdown(
    f"""
    ### LSTM-Based Stock Price Prediction

    This application uses a trained
    **Long Short-Term Memory (LSTM)** neural network
    to analyze historical stock prices.

    **Selected Stock:** `{ticker}`

    **Prediction Window:** Previous 60 trading days
    """
)


# ============================================================
# DOWNLOAD STOCK DATA
# ============================================================

@st.cache_data
def download_stock_data(symbol):

    try:

        stock = yf.Ticker(symbol)

        data = stock.history(
            period="5y",
            auto_adjust=False
        )

        return data

    except Exception:

        return None


# Download data for selected ticker
stock_data = download_stock_data(
    ticker
)


# ============================================================
# CHECK DOWNLOADED DATA
# ============================================================

if stock_data is None:

    st.error(
        f"Unable to download stock data for {ticker}."
    )

    st.info(
        "Please check that the ticker is valid. "
        "Examples: AAPL, TSLA, MSFT, GOOGL, AMZN."
    )

    st.stop()


if stock_data.empty:

    st.error(
        f"No stock data was found for {ticker}."
    )

    st.info(
        "Please enter a valid stock ticker."
    )

    st.stop()


# ============================================================
# HANDLE YFINANCE MULTI-LEVEL COLUMNS
# ============================================================

if isinstance(
    stock_data.columns,
    pd.MultiIndex
):

    stock_data.columns = (
        stock_data.columns
        .get_level_values(0)
    )


# ============================================================
# CHECK CLOSE COLUMN
# ============================================================

if "Close" not in stock_data.columns:

    st.error(
        "The downloaded data does not contain "
        "a 'Close' price column."
    )

    st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

stock_data = stock_data.dropna(
    subset=["Close"]
)


# ============================================================
# CLOSE PRICES
# ============================================================

close_prices = stock_data[
    "Close"
].values


close_prices = np.array(
    close_prices,
    dtype=float
).reshape(-1, 1)


# ============================================================
# CHECK SUFFICIENT DATA
# ============================================================

if len(close_prices) < lookback:

    st.error(
        f"Only {len(close_prices)} trading days "
        f"were downloaded for {ticker}. "
        f"At least {lookback} days are required."
    )

    st.stop()


# ============================================================
# CURRENT PRICE
# ============================================================

current_price = float(
    close_prices[-1][0]
)


# ============================================================
# SUCCESS MESSAGE
# ============================================================

st.success(
    f"Successfully loaded "
    f"{len(close_prices)} trading days "
    f"for {ticker}."
)


# ============================================================
# CURRENT STOCK INFORMATION
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
    f"📊 {ticker} Historical Closing Price"
)


historical_fig = go.Figure()


historical_fig.add_trace(
    go.Scatter(
        x=stock_data.index,
        y=close_prices.flatten(),
        mode="lines",
        name=f"{ticker} Closing Price"
    )
)


historical_fig.update_layout(
    title=f"{ticker} Historical Closing Price",
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
    f"The model uses the previous {lookback} "
    f"trading days to predict the next closing price."
)


predict_button = st.button(
    "🚀 Predict Next Price",
    type="primary"
)


# ============================================================
# MAKE PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # IMPORTANT MODEL WARNING
    # --------------------------------------------------------

    if ticker != "AAPL":

        st.warning(
            "⚠️ Your saved LSTM model was trained on "
            "AAPL data. The prediction for another ticker "
            "is only a demonstration and is not a properly "
            "trained model for that stock."
        )


    # --------------------------------------------------------
    # Get last 60 days
    # --------------------------------------------------------

    last_60_days = close_prices[
        -lookback:
    ]


    # --------------------------------------------------------
    # Scale data
    # --------------------------------------------------------

    try:

        scaled_last_60 = scaler.transform(
            last_60_days
        )

    except Exception as e:

        st.error(
            f"Error scaling stock data: {e}"
        )

        st.stop()


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

    try:

        prediction_scaled = model.predict(
            X_input,
            verbose=0
        )

    except Exception as e:

        st.error(
            f"Error making prediction: {e}"
        )

        st.stop()


    # --------------------------------------------------------
    # Convert prediction to original price
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
    # PREDICTION VISUALIZATION
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


    # --------------------------------------------------------
    # Prediction date
    # --------------------------------------------------------

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
        title=f"{ticker} - Last 60 Days + Prediction",
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
    **Disclaimer:** This application is for educational
    and demonstration purposes only. Stock-price predictions
    are estimates and should not be considered financial advice.
    """
)


st.caption(
    "Stock Price Prediction | LSTM | Streamlit"
)
