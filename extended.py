import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import warnings
warnings.filterwarnings('ignore')

# Model imports
try:
    from pmdarima import auto_arima
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError as e:
    st.error(f"Required library missing: {e}. Install with: pip install pmdarima scikit-learn tensorflow")
    st.stop()

# Page configuration
st.set_page_config(page_title="LME Copper Price Forecaster", layout="wide", page_icon="📊")

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'arima_model' not in st.session_state:
    st.session_state.arima_model = None
if 'lstm_model' not in st.session_state:
    st.session_state.lstm_model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'best_metrics' not in st.session_state:
    st.session_state.best_metrics = {'rmse_7': float('inf'), 'rmse_30': float('inf')}
if 'data' not in st.session_state:
    st.session_state.data = None

# Title and description
st.title("📊 LME Copper Price Forecasting System")
st.markdown("**Hybrid ARIMA-LSTM Model for Short-term Copper Price Prediction**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    data_source = st.radio("Data Source", ["Upload CSV", "Sample Data"])
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload LME Copper Data", type=['csv'])
        st.info("📋 CSV should have **'Date'** and **'Price'** columns\n\n(Case-insensitive: date/Date/DATE and price/Price/PRICE all work)")
    
    st.divider()
    
    months_back = st.slider("Training Data Window (months)", 6, 12, 9)
    st.caption(f"⏱️ Trains on last {months_back} months only")
    
    lstm_epochs = st.slider("LSTM Training Epochs", 20, 100, 50)
    lookback_days = st.slider("LSTM Lookback Window", 5, 30, 14)
    
    st.divider()
    
    arima_weight = st.slider("ARIMA Weight in Blend", 0.0, 1.0, 0.5, 0.1)
    lstm_weight = 1.0 - arima_weight
    st.caption(f"LSTM Weight: {lstm_weight:.1f}")
    
    anchoring_factor = st.slider("Anchoring Factor", 0.0, 0.5, 0.2, 0.05)
    
    add_volatility = st.checkbox("Add Volatility & Shocks", value=True, 
                                  help="Include realistic price volatility and sudden movements")
    
    if add_volatility:
        volatility_level = st.slider("Volatility Intensity", 0.5, 2.0, 1.0, 0.1,
                                     help="Higher values = more dramatic price swings")


def generate_sample_data():
    """Generate realistic sample LME copper price data"""
    np.random.seed(42)
    # Generate 3 years of data
    dates = pd.date_range(end=datetime.now(), periods=1095, freq='D')
    
    # Base price with trend and seasonality
    base_price = 8500
    trend = np.linspace(0, 1500, len(dates))
    # Multiple seasonal patterns
    seasonality = (500 * np.sin(np.linspace(0, 12*np.pi, len(dates))) + 
                   300 * np.sin(np.linspace(0, 6*np.pi, len(dates))))
    noise = np.random.normal(0, 200, len(dates))
    
    # Add occasional shocks
    shock_indices = np.random.choice(len(dates), size=int(len(dates)*0.05), replace=False)
    shocks = np.zeros(len(dates))
    shocks[shock_indices] = np.random.choice([-1, 1], size=len(shock_indices)) * np.random.uniform(300, 800, size=len(shock_indices))
    
    prices = base_price + trend + seasonality + noise + shocks
    prices = np.maximum(prices, 7000)  # Floor price
    
    df = pd.DataFrame({
        'date': dates,
        'price': prices
    })
    return df


def load_and_preprocess_data(df, months):
    """Load and preprocess data, keeping only recent months"""
    df = df.copy()
    
    # Normalize column names to lowercase
    df.columns = df.columns.str.lower().str.strip()
    
    # Check for required columns
    if 'date' not in df.columns or 'price' not in df.columns:
        # Try alternative names
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        price_cols = [col for col in df.columns if 'price' in col.lower() or 'close' in col.lower() or 'value' in col.lower()]
        
        if date_cols:
            df = df.rename(columns={date_cols[0]: 'date'})
        if price_cols:
            df = df.rename(columns={price_cols[0]: 'price'})
        
        # Final check
        if 'date' not in df.columns or 'price' not in df.columns:
            raise ValueError(f"CSV must have 'date' and 'price' columns. Found columns: {list(df.columns)}")
    
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # Remove any rows with missing values
    df = df.dropna(subset=['date', 'price'])
    
    df = df.sort_values('date').reset_index(drop=True)
    
    # Filter to recent months
    cutoff_date = df['date'].max() - timedelta(days=months*30)
    df = df[df['date'] >= cutoff_date].reset_index(drop=True)
    
    # Calculate daily returns
    df['return'] = df['price'].pct_change()
    df = df.dropna().reset_index(drop=True)
    
    return df


def create_lstm_sequences(data, lookback):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def build_lstm_model(lookback):
    """Build LSTM model architecture"""
    model = Sequential([
        LSTM(50, activation='tanh', return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(50, activation='tanh'),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_arima_model(returns):
    """Train ARIMA model on returns"""
    with st.spinner("Training ARIMA model..."):
        model = auto_arima(
            returns,
            start_p=1, start_q=1,
            max_p=5, max_q=5,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore'
        )
    return model


def train_lstm_model(returns, lookback, epochs):
    """Train LSTM model on returns"""
    # Scale data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    returns_scaled = scaler.fit_transform(returns.values.reshape(-1, 1))
    
    # Create sequences
    X, y = create_lstm_sequences(returns_scaled, lookback)
    
    # Train-test split (80-20)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build and train model
    with st.spinner("Training LSTM model..."):
        model = build_lstm_model(lookback)
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=0
        )
    
    return model, scaler


def forecast_arima(model, steps):
    """Forecast using ARIMA model"""
    forecast = model.predict(n_periods=steps)
    return forecast


def forecast_lstm(model, scaler, recent_data, lookback, steps):
    """Forecast using LSTM model with Monte Carlo simulation for uncertainty"""
    predictions = []
    current_sequence = recent_data[-lookback:].copy()
    
    for _ in range(steps):
        # Prepare input
        X_input = current_sequence.reshape(1, lookback, 1)
        
        # Predict next value
        pred_scaled = model.predict(X_input, verbose=0)[0, 0]
        pred = scaler.inverse_transform([[pred_scaled]])[0, 0]
        predictions.append(pred)
        
        # Update sequence
        current_sequence = np.append(current_sequence[1:], [[pred_scaled]], axis=0)
    
    return np.array(predictions)


def add_volatility_component(return_forecasts, historical_returns, confidence_level=0.8):
    """Add realistic volatility to forecasts using historical volatility patterns"""
    # Ensure return_forecasts is a numpy array
    return_forecasts = np.array(return_forecasts)
    
    # Calculate historical volatility metrics
    historical_std = np.std(historical_returns)
    historical_skew = pd.Series(historical_returns).skew()
    
    # Identify extreme movements
    extreme_threshold = np.percentile(np.abs(historical_returns), 95)
    extreme_prob = np.mean(np.abs(historical_returns) > extreme_threshold)
    
    # Add volatility with increasing uncertainty over time
    adjusted_forecasts = return_forecasts.copy()
    
    for i in range(len(return_forecasts)):
        # Uncertainty grows with forecast horizon
        time_factor = (i + 1) / len(return_forecasts)
        volatility = historical_std * time_factor * (1 - confidence_level)
        
        # Add noise with skewness
        noise = np.random.normal(0, volatility)
        if historical_skew != 0:
            noise += np.random.choice([-1, 1]) * abs(historical_skew) * volatility * 0.5
        
        # Occasionally add extreme movements (shocks)
        if np.random.random() < extreme_prob * time_factor:
            shock = np.random.choice([-1, 1]) * extreme_threshold * np.random.uniform(0.5, 1.0)
            adjusted_forecasts[i] += shock
        else:
            adjusted_forecasts[i] += noise
    
    return adjusted_forecasts


def blend_forecasts(arima_forecast, lstm_forecast, arima_weight):
    """Blend ARIMA and LSTM forecasts"""
    return arima_weight * arima_forecast + (1 - arima_weight) * lstm_forecast


def reconstruct_prices(last_price, return_forecasts, anchoring_factor):
    """Reconstruct prices from returns with anchoring"""
    predicted_prices = []
    current_price = last_price
    
    for ret in return_forecasts:
        next_price = current_price * (1 + ret)
        # Apply anchoring
        anchored_price = (1 - anchoring_factor) * next_price + anchoring_factor * last_price
        predicted_prices.append(anchored_price)
        current_price = anchored_price
    
    return np.array(predicted_prices)


def generate_confidence_intervals(last_price, return_forecasts, historical_returns, anchoring_factor):
    """Generate confidence intervals with realistic volatility"""
    # Calculate volatility that grows with time
    base_volatility = np.std(historical_returns)
    
    lower_bounds = []
    upper_bounds = []
    
    for i, ret in enumerate(return_forecasts):
        days_ahead = i + 1
        # Volatility increases with square root of time (standard finance theory)
        time_volatility = base_volatility * np.sqrt(days_ahead)
        
        # Calculate price from return
        price = last_price * (1 + ret)
        
        # Apply anchoring
        price = (1 - anchoring_factor) * price + anchoring_factor * last_price
        
        # Calculate confidence intervals (95%)
        lower = price * (1 - 1.96 * time_volatility)
        upper = price * (1 + 1.96 * time_volatility)
        
        lower_bounds.append(lower)
        upper_bounds.append(upper)
    
    return np.array(lower_bounds), np.array(upper_bounds)


def rolling_origin_validation(df, arima_model, lstm_model, scaler, lookback, horizon, arima_weight):
    """Perform rolling-origin validation"""
    predictions = []
    actuals = []
    
    # Use last 60 days for validation
    validation_start = max(lookback + 30, len(df) - 90)
    
    for i in range(validation_start, len(df) - horizon):
        train_returns = df['return'].iloc[:i].values
        
        # ARIMA forecast
        arima_temp = auto_arima(train_returns, suppress_warnings=True, error_action='ignore')
        arima_pred = forecast_arima(arima_temp, horizon)
        
        # LSTM forecast
        scaler_temp = MinMaxScaler(feature_range=(-1, 1))
        scaled_temp = scaler_temp.fit_transform(train_returns.reshape(-1, 1))
        lstm_pred = forecast_lstm(lstm_model, scaler_temp, scaled_temp, lookback, horizon)
        
        # Blend
        blended = blend_forecasts(arima_pred, lstm_pred, arima_weight)
        
        # Reconstruct prices
        last_price = df['price'].iloc[i-1]
        pred_prices = reconstruct_prices(last_price, blended, 0.2)
        
        predictions.append(pred_prices[-1])
        actuals.append(df['price'].iloc[i + horizon - 1])
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    
    return rmse, mae


# Main app logic
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Data & Training")
    
    # Load data
    if data_source == "Sample Data":
        df_original = generate_sample_data()
        st.success(f"✅ Sample data loaded ({len(df_original)} days)")
    elif uploaded_file is not None:
        try:
            df_original = pd.read_csv(uploaded_file)
            st.success(f"✅ CSV data loaded ({len(df_original)} records)")
            
            # Show detected columns
            st.caption(f"Detected columns: {', '.join(df_original.columns.tolist())}")
        except Exception as e:
            st.error(f"❌ Error reading CSV: {e}")
            st.stop()
    else:
        st.info("⬆️ Please upload a CSV file")
        st.stop()
    
    # Preprocess
    try:
        df = load_and_preprocess_data(df_original, months_back)
        st.session_state.data = df
    except ValueError as e:
        st.error(f"❌ {str(e)}")
        st.info("💡 Please ensure your CSV has columns named 'Date' and 'Price' (or similar)")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error processing data: {e}")
        st.stop()
    
    # Display data summary
    st.write(f"**Full Dataset:** {len(df_original) if 'df_original' in locals() else 'N/A'} days")
    st.write(f"**Training Data Range:** {df['date'].min().date()} to {df['date'].max().date()}")
    st.write(f"**Training Records:** {len(df)} days ({months_back} months)")
    st.write(f"**Current Price:** ${df['price'].iloc[-1]:,.2f}")
    
    st.info(f"ℹ️ Using most recent **{months_back} months** for training (adjust in sidebar). This captures the current price regime while avoiding outdated patterns.")
    
    # Show recent data
    with st.expander("View Recent Training Data (Last 10 Days)"):
        st.dataframe(df.tail(10), use_container_width=True)
    
    with st.expander("View All Available Data"):
        st.dataframe(df_original.tail(50), use_container_width=True)
        st.caption(f"Showing last 50 of {len(df_original)} total records")

with col2:
    st.subheader("🎯 Model Training")
    
    if st.button("🚀 Train Models", type="primary", use_container_width=True):
        try:
            returns = df['return']
            
            # Train ARIMA
            arima_model = train_arima_model(returns)
            
            # Train LSTM
            lstm_model, scaler = train_lstm_model(returns, lookback_days, lstm_epochs)
            
            # Validate
            st.info("Running validation...")
            rmse_7, mae_7 = rolling_origin_validation(
                df, arima_model, lstm_model, scaler, lookback_days, 7, arima_weight
            )
            rmse_30, mae_30 = rolling_origin_validation(
                df, arima_model, lstm_model, scaler, lookback_days, 30, arima_weight
            )
            
            # Check if better than previous
            if rmse_7 < st.session_state.best_metrics['rmse_7']:
                st.session_state.arima_model = arima_model
                st.session_state.lstm_model = lstm_model
                st.session_state.scaler = scaler
                st.session_state.best_metrics = {'rmse_7': rmse_7, 'rmse_30': rmse_30}
                st.session_state.model_trained = True
                st.success("✅ Models trained and saved!")
            else:
                st.warning("⚠️ New model performance not better. Keeping previous model.")
            
            # Display metrics
            st.metric("7-Day RMSE", f"${rmse_7:,.2f}")
            st.metric("30-Day RMSE", f"${rmse_30:,.2f}")
            st.metric("7-Day MAE", f"${mae_7:,.2f}")
            st.metric("30-Day MAE", f"${mae_30:,.2f}")
            
        except Exception as e:
            st.error(f"Training error: {e}")
    
    if st.session_state.model_trained:
        st.success("✅ Model Ready")
        st.caption(f"Best 7-day RMSE: ${st.session_state.best_metrics['rmse_7']:,.2f}")

# Forecasting section
if st.session_state.model_trained:
    st.divider()
    st.subheader("🔮 Forecasts")
    
    # Generate forecasts
    returns = df['return'].values
    last_price = df['price'].iloc[-1]
    
    # ARIMA forecasts
    arima_7 = forecast_arima(st.session_state.arima_model, 7)
    arima_30 = forecast_arima(st.session_state.arima_model, 30)
    
    # LSTM forecasts
    scaler = st.session_state.scaler
    returns_scaled = scaler.transform(returns.reshape(-1, 1))
    lstm_7 = forecast_lstm(st.session_state.lstm_model, scaler, returns_scaled, lookback_days, 7)
    lstm_30 = forecast_lstm(st.session_state.lstm_model, scaler, returns_scaled, lookback_days, 30)
    
    # Blend forecasts
    blended_7 = blend_forecasts(arima_7, lstm_7, arima_weight)
    blended_30 = blend_forecasts(arima_30, lstm_30, arima_weight)
    
    # Add volatility and shocks if enabled
    if add_volatility:
        historical_returns = df['return'].values
        blended_7 = add_volatility_component(blended_7, historical_returns, 
                                            confidence_level=1.0/volatility_level)
        blended_30 = add_volatility_component(blended_30, historical_returns, 
                                             confidence_level=1.0/volatility_level)
    
    # Reconstruct prices
    prices_7 = reconstruct_prices(last_price, blended_7, anchoring_factor)
    prices_30 = reconstruct_prices(last_price, blended_30, anchoring_factor)
    
    # Generate confidence intervals
    lower_7, upper_7 = generate_confidence_intervals(last_price, blended_7, df['return'].values, anchoring_factor)
    lower_30, upper_30 = generate_confidence_intervals(last_price, blended_30, df['return'].values, anchoring_factor)
    
    # Display predictions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Current Price",
            f"${last_price:,.2f}",
            help="Latest price from data"
        )
    
    with col2:
        price_7 = prices_7[-1]
        change_7 = ((price_7 - last_price) / last_price) * 100
        st.metric(
            "7-Day Forecast",
            f"${price_7:,.2f}",
            f"{change_7:+.2f}%",
            delta_color="normal"
        )
    
    with col3:
        price_30 = prices_30[-1]
        change_30 = ((price_30 - last_price) / last_price) * 100
        st.metric(
            "30-Day Forecast",
            f"${price_30:,.2f}",
            f"{change_30:+.2f}%",
            delta_color="normal"
        )
    
    # Create forecast chart
    st.subheader("📊 Price Forecast Visualization")
    
    # Prepare data for plotting
    historical_dates = df['date'].values
    historical_prices = df['price'].values
    
    forecast_dates_7 = pd.date_range(start=df['date'].iloc[-1] + timedelta(days=1), periods=7, freq='D')
    forecast_dates_30 = pd.date_range(start=df['date'].iloc[-1] + timedelta(days=1), periods=30, freq='D')
    
    # Create figure
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_dates[-90:],
        y=historical_prices[-90:],
        mode='lines',
        name='Historical',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # 7-day forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates_7,
        y=prices_7,
        mode='lines+markers',
        name='7-Day Forecast',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    # 30-day forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates_30,
        y=prices_30,
        mode='lines+markers',
        name='30-Day Forecast',
        line=dict(color='#2ca02c', width=2, dash='dot'),
        marker=dict(size=4)
    ))
    
    # Add confidence intervals if volatility is enabled
    if add_volatility:
        # 7-day confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_dates_7,
            y=upper_7,
            mode='lines',
            name='7-Day Upper (95%)',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=forecast_dates_7,
            y=lower_7,
            mode='lines',
            name='7-Day CI',
            line=dict(width=0),
            fillcolor='rgba(255, 127, 14, 0.2)',
            fill='tonexty',
            showlegend=True
        ))
        
        # 30-day confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_dates_30,
            y=upper_30,
            mode='lines',
            name='30-Day Upper (95%)',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=forecast_dates_30,
            y=lower_30,
            mode='lines',
            name='30-Day CI',
            line=dict(width=0),
            fillcolor='rgba(44, 160, 44, 0.2)',
            fill='tonexty',
            showlegend=True
        ))
    
    fig.update_layout(
        title="LME Copper Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price (USD/ton)",
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Export functionality
    st.subheader("💾 Export Forecasts")
    
    # Prepare export data
    export_df = pd.DataFrame({
        'Date': forecast_dates_30,
        'Predicted_Price': prices_30,
        'Lower_Bound_95': lower_30,
        'Upper_Bound_95': upper_30,
        'Days_Ahead': range(1, 31)
    })
    
    # Convert to CSV
    csv = export_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv,
        file_name=f"copper_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    with st.expander("View Forecast Table"):
        st.dataframe(export_df, use_container_width=True)
    
    # Volatility analysis
    if add_volatility:
        st.subheader("📊 Volatility Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            daily_vol = np.std(df['return'].values) * 100
            st.metric("Daily Volatility", f"{daily_vol:.2f}%")
        
        with col2:
            max_gain = np.max(df['return'].values) * 100
            st.metric("Hist. Max Daily Gain", f"+{max_gain:.2f}%")
        
        with col3:
            max_loss = np.min(df['return'].values) * 100
            st.metric("Hist. Max Daily Loss", f"{max_loss:.2f}%")
        
        st.info("💡 **Note:** Forecasts include realistic volatility based on historical patterns. The confidence intervals show the 95% probability range for future prices.")

else:
    st.info("👆 Please train the models first to generate forecasts")

# Footer
st.divider()
st.caption("**Note:** Forecasts are for informational purposes only. Always conduct thorough analysis before making trading decisions.")
st.caption("Model uses hybrid ARIMA-LSTM architecture with anchoring adjustments for stability.")
