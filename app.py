import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

# Import our custom modules
from datafetcher import DataFetcher, DataValidationException
from model import StockPredictor


# Configure Streamlit page
st.set_page_config(
    page_title="🔮 ML Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional dark theme
st.markdown("""
<style>
    /* --- General App & Text Styling --- */
    body {
        color: #fafafa;
    }
    .main {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        color: #fafafa;
    }
    .st-emotion-cache-16txtl3 {
        color: #fafafa;
    }
    .st-emotion-cache-1avcm0n {
        color: #fafafa;
    }

    /* --- Button Styles --- */
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #ff4b4b;
        transition: transform 0.2s ease, background-color 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.4);
        background-color: #ff4b4b;
        color: white;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 10px rgba(255, 75, 75, 0.3);
        background-color: #ff6a6a;
    }
    
    /* --- Investment Signal Cards (Dark Theme) --- */
    .signal-card {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-left: 5px solid;
    }
    .buy-signal {
        background-color: #1c3d2f;
        border-color: #28a745;
        color: #29b56d;
    }
    .sell-signal {
        background-color: #4a272e;
        border-color: #dc3545;
        color: #e55353;
    }
    .hold-signal {
        background-color: #493f2b;
        border-color: #ffc107;
        color: #ffc107;
    }
    
    /* --- Performance Metric Cards (Dark Theme) --- */
    .metric-card {
        background-color: #2a2a2e;
        color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .metric-label {
        font-size: 0.9rem;
        color: #a0a0a0;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: bold;
    }
    .metric-value-green { color: #29b56d; }
    .metric-value-orange { color: #ffc107; }
    .metric-value-red { color: #e55353; }

    /* --- Confidence Badge --- */
    .confidence-badge {
        display: inline-block;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        font-weight: bold;
        color: white;
    }
    .confidence-green { background-color: #28a745; }
    .confidence-yellow { background-color: #ffc107; color: #333; }
    .confidence-red { background-color: #dc3545; }

    /* --- Animations --- */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .results-container {
        animation: fadeIn 0.5s ease-out;
    }

    /* --- Custom Animated Progress Bar --- */
    .loading-text {
        text-align: center;
        margin-bottom: 8px;
        color: #a0a0a0;
        animation: pulse 1.8s infinite ease-in-out;
    }
    .custom-progress-bar-container {
        width: 100%;
        height: 10px;
        background-color: #2a2a2e;
        border-radius: 10px;
        overflow: hidden;
    }
    .custom-progress-bar-filler {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(270deg, #00f2fe, #4facfe, #00f2fe);
        background-size: 200% 200%;
        animation: animateGradientFlow 2.5s ease-in-out infinite;
        transition: width 0.4s ease-in-out;
        box-shadow: 0 0 10px rgba(79, 172, 254, 0.4);
    }

    @keyframes pulse {
        0%, 100% { opacity: 0.7; }
        50% { opacity: 1; }
    }
    @keyframes animateGradientFlow {
        0% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* --- Responsive Design --- */
    @media (max-width: 768px) {
        .stButton > button {
            width: 100%;
        }
        .stColumns {
            flex-direction: column;
        }
    }
</style>
""", unsafe_allow_html=True)


def display_custom_progress_bar(container, progress: int, text: str):
    """
    Displays a custom animated progress bar.
    
    Args:
        container: The st.empty() container to draw in.
        progress: The progress percentage (0-100).
        text: The text to display above the bar.
    """
    bar_html = f"""
        <div>
            <p class="loading-text">{text}</p>
            <div class="custom-progress-bar-container">
                <div class="custom-progress-bar-filler" style="width: {progress}%;"></div>
            </div>
        </div>
    """
    container.markdown(bar_html, unsafe_allow_html=True)


def get_styled_metric_html(label: str, value: str, color_class: str = ""):
    """Generates HTML for a styled metric card."""
    return f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {color_class}">{value}</div>
        </div>
    """


def initialize_session_state():
    """Initialize session state variables."""
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None
    if 'training_results' not in st.session_state:
        st.session_state.training_results = None
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = None
    if 'data_source' not in st.session_state:
        st.session_state.data_source = None
    if 'best_model_type' not in st.session_state:
        st.session_state.best_model_type = None


def get_currency_symbol(ticker: str) -> str:
    """Get currency symbol based on ticker."""
    if ticker and (ticker.endswith('.NS') or ticker.endswith('.BO')):
        return "₹"
    return "$"


def create_price_chart(data: pd.DataFrame, ticker: str, currency_symbol: str, 
                         prediction_result: dict = None):
    """
    Create an interactive price chart using Plotly, with a single prediction.
    """
    chart_data = data.tail(100).copy()
    
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=chart_data['Date'],
        open=chart_data['Open'],
        high=chart_data['High'],
        low=chart_data['Low'],
        close=chart_data['Close'],
        name=f"{ticker} Price",
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))
    
    if prediction_result and prediction_result.get('success'):
        last_date = chart_data['Date'].iloc[-1]
        next_date = last_date + timedelta(days=1)
        predicted_price = prediction_result['predicted_price']
        
        fig.add_trace(go.Scatter(
            x=[last_date, next_date],
            y=[chart_data['Close'].iloc[-1], predicted_price],
            mode='lines+markers',
            name='Prediction',
            line=dict(color='#ff7f0e', width=3, dash='dash'),
            marker=dict(size=10, color='#ff7f0e')
        ))
    
    fig.update_layout(
        title=f"{ticker} Stock Price Chart (Last 100 Days)",
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_symbol})",
        template="plotly_dark",
        height=500,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig


def create_volume_chart(data: pd.DataFrame, ticker: str):
    """Create volume chart if volume data is available."""
    if 'Volume' not in data.columns or data['Volume'].isna().all():
        return None
    
    chart_data = data.tail(100).copy()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=chart_data['Date'],
        y=chart_data['Volume'],
        name='Volume',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title=f"{ticker} Trading Volume",
        xaxis_title="Date",
        yaxis_title="Volume",
        template="plotly_white",
        height=300
    )
    
    return fig


def display_signal(signal_info: dict):
    """Display the investment signal with appropriate styling."""
    action = signal_info['action']
    reason = signal_info['reason']
    
    if action == 'BUY':
        st.markdown(f'<div class="signal-card buy-signal">🚀 BUY: {reason}</div>', unsafe_allow_html=True)
    elif action == 'SELL':
        st.markdown(f'<div class="signal-card sell-signal">📉 SELL: {reason}</div>', unsafe_allow_html=True)
    else:  # HOLD
        st.markdown(f'<div class="signal-card hold-signal">⏸️ HOLD: {reason}</div>', unsafe_allow_html=True)


def display_model_metrics(metrics: dict, currency_symbol: str):
    """Display model performance metrics with color coding."""
    st.subheader("📊 Model Performance")

    r2 = metrics.get('test_r2', 0)
    rmse = metrics.get('test_rmse', 0)
    train_samples = metrics.get('train_samples', 0)

    # Determine colors for metrics
    if r2 > 0.9:
        r2_color_class = "metric-value-green"
    elif r2 > 0.7:
        r2_color_class = "metric-value-orange"
    else:
        r2_color_class = "metric-value-red"
    
    # For RMSE, a value less than 1 is excellent, otherwise, it's a point of caution.
    rmse_color_class = "metric-value-green" if rmse < 1.0 else "metric-value-red"

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(get_styled_metric_html("Test R² Score", f"{r2:.3f}", r2_color_class), unsafe_allow_html=True)
    
    with col2:
        st.markdown(get_styled_metric_html("Test RMSE", f"{currency_symbol}{rmse:.2f}", rmse_color_class), unsafe_allow_html=True)
        
    with col3:
        st.markdown(get_styled_metric_html("Train Samples", f"{train_samples}", ""), unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Main header
    st.markdown('<h1 style="text-align: center;">🔮 ML Stock Price Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("📝 Configuration")
        
        with st.form(key="prediction_form"):
            # Stock ticker input
            ticker = st.text_input(
                "Stock Ticker Symbol",
                value="AAPL",
                help="Enter stock ticker (e.g., AAPL, TCS.NS)",
                placeholder="AAPL, MSFT, TCS.NS, RELIANCE.NS"
            ).upper().strip()
            
            # Prediction threshold
            st.markdown("---")
            threshold = st.slider(
                "Signal Threshold (in stock's currency)",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="Minimum price difference to trigger buy/sell signal"
            )
            
            # The submit button for the form
            predict_button = st.form_submit_button(
                "🔮 Predict Stock Price",
                type="primary",
                use_container_width=True
            )

    # This is now outside the form, so it gets the correct currency for the submitted ticker
    currency_symbol = get_currency_symbol(ticker)
    
    if predict_button:
        if not ticker:
            st.error("❌ Please enter a stock ticker symbol")
            return
        
        # Create a container for the custom progress bar
        progress_container = st.empty()
        
        try:
            display_custom_progress_bar(progress_container, 5, "🔄 Initializing...")
            data_fetcher = DataFetcher()
            
            display_custom_progress_bar(progress_container, 10, "📊 Fetching & validating stock data...")
            stock_data, data_source = data_fetcher.fetch_and_validate_data(ticker)
            st.session_state.stock_data = stock_data
            st.session_state.data_source = data_source
            
            # --- Model Training and Selection ---
            display_custom_progress_bar(progress_container, 40, "🤖 Training primary model (XGBoost)...")
            
            predictor_xgb = StockPredictor(threshold=threshold, model_type='XGBoost')
            training_results_xgb = predictor_xgb.train(stock_data)
            prediction_results_xgb = predictor_xgb.predict_next_price(stock_data, currency_symbol)

            display_custom_progress_bar(progress_container, 70, "📱 Training comparison model (RandomForest)...")
            predictor_rf = StockPredictor(threshold=threshold, model_type='RandomForest')
            training_results_rf = predictor_rf.train(stock_data)
            prediction_results_rf = predictor_rf.predict_next_price(stock_data, currency_symbol)

            # --- Automatically select the best model based on R² ---
            display_custom_progress_bar(progress_container, 95, "✅ Finalizing prediction...")
            xgb_r2 = training_results_xgb.get('metrics', {}).get('test_r2', 0)
            rf_r2 = training_results_rf.get('metrics', {}).get('test_r2', 0)

            if rf_r2 > xgb_r2:
                st.session_state.best_model_type = 'RandomForest'
                st.session_state.training_results = training_results_rf
                st.session_state.prediction_results = prediction_results_rf
            else:
                st.session_state.best_model_type = 'XGBoost'
                st.session_state.training_results = training_results_xgb
                st.session_state.prediction_results = prediction_results_xgb

            display_custom_progress_bar(progress_container, 100, "✨ Done!")
            time.sleep(1)
            progress_container.empty()
            
        except DataValidationException as e:
            progress_container.empty()
            st.error("❌ Invalid or unsupported stock symbol. Please check the ticker and try again.")
            st.info("💡 **Tip:** For Indian stocks, use the full ticker symbol ending in `.NS` or `.BO` (e.g., 'TCS.NS', 'RELIANCE.BO'). For US stocks, use the symbol directly (e.g., 'AAPL', 'GOOGL').")
            # Log the detailed error for debugging, but don't show it to the user
            print(f"Data Validation Error for ticker '{ticker}': {e}")
            return
        except Exception as e:
            progress_container.empty()
            st.error(f"❌ An unexpected error occurred. Please try again.")
            print(f"Unexpected Error: {e}") # Log for debugging
            return
    
    # Display results if available
    if st.session_state.prediction_results and st.session_state.prediction_results['success']:
        st.markdown("<div class='results-container'>", unsafe_allow_html=True)
        prediction = st.session_state.prediction_results
        
        st.success(f"✅ Best Model Used: **{st.session_state.best_model_type}**")
        
        # Low confidence warning
        CONFIDENCE_WARNING_THRESHOLD = 0.15
        confidence = prediction.get('confidence', 0.5)
        if confidence < CONFIDENCE_WARNING_THRESHOLD:
            st.warning("⚠️ Prediction confidence is low, likely due to high market volatility or limited historical data for this stock.")

        # Key metrics display
        result_cols = st.columns(4)
        
        with result_cols[0]:
            st.metric(label="Current Price", value=f"{currency_symbol}{prediction['current_price']:.2f}")
        with result_cols[1]:
            st.metric(label="Predicted Price", value=f"{currency_symbol}{prediction['predicted_price']:.2f}", delta=f"{currency_symbol}{prediction['price_difference']:.2f}")
        with result_cols[2]:
            change_percent = (prediction['price_difference'] / prediction['current_price']) * 100
            st.metric(label="Expected Change", value=f"{change_percent:.2f}%")
        with result_cols[3]:
            if confidence > 0.8:
                badge_class, help_text = "confidence-green", "High confidence (>80%)"
            elif confidence >= 0.4:
                badge_class, help_text = "confidence-yellow", "Moderate confidence (40-80%)"
            else:
                badge_class, help_text = "confidence-red", "Low confidence (<40%)"

            st.markdown(f"""
                <div class="metric-card" style="background:none; box-shadow:none; padding:0; margin:0;">
                    <div class="metric-label">Confidence</div>
                    <div class="confidence-badge {badge_class}" title="{help_text}">{confidence:.1%}</div>
                </div>
            """, unsafe_allow_html=True)

        # Investment signal
        st.subheader("🎯 Investment Signal")
        display_signal(prediction['signal'])
        
        st.subheader("📊 Price Chart")
        if st.session_state.stock_data is not None:
            price_chart = create_price_chart(
                st.session_state.stock_data, 
                ticker,
                currency_symbol,
                st.session_state.prediction_results
            )
            st.plotly_chart(price_chart, use_container_width=True)
            
            volume_chart = create_volume_chart(st.session_state.stock_data, ticker)
            if volume_chart:
                st.plotly_chart(volume_chart, use_container_width=True)
        
        if st.session_state.training_results and st.session_state.training_results['success']:
            display_model_metrics(st.session_state.training_results['metrics'], currency_symbol)
        
        if st.session_state.training_results and 'feature_importance' in st.session_state.training_results:
            with st.expander(f"🔍 Feature Importance Analysis ({st.session_state.best_model_type})"):
                feature_importance = st.session_state.training_results['feature_importance']
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                if sorted_features:
                    features, importance = zip(*sorted_features)
                    fig = px.bar(x=list(importance), y=list(features), orientation='h', title="Top 10 Most Important Features", labels={'x': 'Importance Score', 'y': 'Features'})
                    fig.update_layout(height=400, template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"❌ Prediction failed or data is unrealistic.")

        st.markdown("</div>", unsafe_allow_html=True)
    
    # Information section
    st.markdown("---")
    st.header("ℹ️ How It Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔄 Data Sources")
        st.markdown("""
        - **Indian Stocks** (.NS, .BO): Uses yfinance library
        - **US Stocks**: Uses Twelve Data API
        - **Data Period**: Last 2 years of daily data
        """)
        
        st.subheader("🤖 Machine Learning Model")
        st.markdown("""
        - **Algorithms**: XGBoost & Random Forest
        - **Selection**: Best model is chosen automatically based on R² score
        - **Features**: 15+ technical indicators
        - **Validation**: Time-series cross-validation
        """)
    
    with col2:
        st.subheader("📊 Technical Indicators")
        st.markdown("""
        - Moving Averages (SMA, EMA)
        - MACD and Signal Lines
        - RSI (Relative Strength Index)
        - Bollinger Bands & ATR
        - ADX (Average Directional Index)
        """)
        
        st.subheader("🎯 Signal Generation")
        st.markdown(f"""
        - **Buy**: Predicted price > Current price + {currency_symbol}{threshold}
        - **Sell**: Predicted price < Current price - {currency_symbol}{threshold}
        - **Hold**: Price change within threshold range
        - **Confidence**: Based on recent volatility vs. historical average
        """)
    
    # Disclaimer
    st.markdown("---")
    st.warning("""
    ⚠️ **Important Disclaimer**: This application is for educational and research purposes only. 
    Stock market predictions are inherently uncertain and past performance does not guarantee future results. 
    Always conduct your own research and consider consulting with financial advisors before making investment decisions.
    """)
    
    # Footer
    st.markdown("""
    ---
    <div style="text-align: center; color: #888;">
    <p>Smart Stock Predictor | Built with Streamlit, scikit-learn, and XGBoost</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()