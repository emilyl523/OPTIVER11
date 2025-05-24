import streamlit as st
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, time
import ta  
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import lightgbm as lgb
import numpy as np
from tqdm import tqdm

# Enhanced Page Configuration
st.set_page_config(
    page_title="EchoVol20",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Dark Mode Option
st.markdown("""
<style>
:root {
    --primary: #1a73e8;
    --secondary: #34a853;
    --accent: #fbbc05;
    --text: #202124;
    --bg: #ffffff;
    --card-bg: #f8f9fa;
    --border: #dadce0;
}

[data-theme="dark"] {
    --primary: #8ab4f8;
    --secondary: #81c995;
    --accent: #fdd663;
    --text: #e8eaed;
    --bg: #202124;
    --card-bg: #2d2e30;
    --border: #5f6368;
}

.header-container {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 56px;
    background: linear-gradient(90deg, #1a73e8 0%, #0d47a1 100%);
    color: white;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 20px;
    z-index: 999;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
}

/* Main container styling */
.main {
    background-color: var(--card-bg);
    color: var(--text);
}

/* Card styling */
.card {
    background-color: var(--card-bg);
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    margin-bottom: 20px;
    border: 1px solid var(--border);
}

/* Metric styling */
.metric {
    font-family: 'San Francisco', sans-serif;
}

/* Custom tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    padding: 8px 16px;
    border-radius: 8px 8px 0 0;
    background-color: var(--card-bg);
    transition: all 0.3s ease;
}

.stTabs [aria-selected="true"] {
    background-color: var(--primary) !important;
    color: white !important;
}

/* Custom sidebar */
[data-testid="stSidebar"] {
    background: #fafafa;
    box-shadow: 5px 0 30px rgba(6, 103, 214, 0.3); /* Light subtle shadow */;
    border-right: 1px solid var(--border);
}

/* Plotly chart styling */
.js-plotly-plot .plotly .modebar {
    background: var(--card-bg) !important;
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: var(--card-bg);
}

::-webkit-scrollbar-thumb {
    background: var(--primary);
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)


COMPANIES = {
        "AAPL": {
                "ticker": "AAPL",
                "id": 9323,
                "name": "Apple Inc.",
                "logo": "https://1000logos.net/wp-content/uploads/2016/10/Apple-Logo.png",
                "sector": "Technology"
            },
            "AMZN": {
                "ticker": "AMZN",
                "id": 22675,
                "name": "Amazon Inc.",
                "logo": "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg",
                "sector": "Consumer Discretionary"
            },
            "FACEBOOK": {
                "ticker": "FACEBOOK",
                "id": 22951,
                "name": "Facebook",
                "logo": "https://upload.wikimedia.org/wikipedia/commons/9/93/Facebook_logo_%282023%29.svg",
                "sector": "Communication Services"
            },
            "GOOGC": {
                "ticker": "GOOGC",
                "id": 22729,
                "name": "Alphabet, Inc.",
                "logo": "https://upload.wikimedia.org/wikipedia/commons/2/2f/Google_2015_logo.svg",
                "sector": "Communication Services"
            },
            "GS": {
                "ticker": "GS",
                "id": 48219,
                "name": "Goldman Sachs Group Inc.",
                "logo": "https://upload.wikimedia.org/wikipedia/commons/e/ef/Goldman_Sachs_2022_Black.svg",
                "sector": "Financials"
            },
            "JPM": {
                "ticker": "JPM",
                "id": 22753,
                "name": "JPMorgan Chase & Co.",
                "logo": "https://upload.wikimedia.org/wikipedia/commons/c/c9/Logo_of_JPMorganChase_2024.svg",
                "sector": "Financials"
            },
            "NFLX": {
                "ticker": "NFLX",
                "id": 22771,
                "name": "Netflix Inc.",
                "logo": "https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg",
                "sector": "Communication Services"
            },
            "QQQ": {
                "ticker": "QQQ",
                "id": 104919,
                "name": "Invesco QQQ Trust",
                "logo": "https://mma.prnewswire.com/media/168499/invesco_ltd__logo.jpg",
                "sector": "ETF (tracks Nasdaq-100, mostly Tech)"
            },
            "SPY": {
                "ticker": "SPY",
                "id": 50200,
                "name": "SPDR S&P 500 ETF Trust",
                "logo": "https://www.ssga.com/library-content/images/site/logo-ssga.svg",
                "sector": "ETF (tracks S&P 500, diversified)"
            },
            "TSLA": {
                "ticker": "TSLA",
                "id": 8382,
                "name": "Tesla Inc.",
                "logo": "https://upload.wikimedia.org/wikipedia/commons/b/b8/Tesla%2C_Inc._-_Logo_%28black_script_version%29.svg",
                "sector": "Consumer Discretionary"
            }
        }


# App title and description
st.markdown(f"""
<div class="main">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
        <div>
            <h1 style="color: var(--primary); margin-bottom: 5px;">ECHOVOL20</h1>
            <p style="color: var(--text); opacity: 0.6; margin-top: 0;">Dynamic Analytics for Fast Markets</p>
        </div>
        <div style="display: flex; gap: 15px; align-items: center;">
            <div style="background: var(--card-bg); padding: 8px 15px; border-radius: 20px; font-size: 14px;">
                <span style="color: var(--primary); font-weight: 600;">LIVE</span> • {datetime.now().strftime('%H:%M:%S')}
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_stock(stock_id):
    stock_data = pd.read_csv("full_hour_combined.csv")
    stock_data['datetime'] = pd.to_datetime(stock_data['date'].astype(str) + ' ' + stock_data['time'].astype(str)) + \
                    pd.to_timedelta(stock_data['seconds_in_bucket'], unit='s')
    stock_data.set_index('datetime', inplace=True)
    stock_data.drop(columns=['date', 'time'], inplace=True)
    
    stock_data = stock_data[stock_data['stock_id'] == stock_id]
    return stock_data

@st.cache_data(ttl=3600)
def get_stock_data(stock_id, start_datetime, end_datetime):
    stock_data = load_stock(stock_id)
    stock_data.index = pd.to_datetime(stock_data.index)

    start_datetime = pd.to_datetime(start_datetime)
    end_datetime = pd.to_datetime(end_datetime)
    
    # Filter by datetime range
    mask = (stock_data.index >= start_datetime) & (stock_data.index <= end_datetime)
    filtered_data = stock_data.loc[mask]

    filtered_data['Close'] = (stock_data['bid_price1'] * stock_data['ask_size1'] + stock_data['ask_price1'] * stock_data['bid_size1']) / (stock_data['bid_size1'] + stock_data['ask_size1'])
    return filtered_data


# Sidebar for user inputs
with st.sidebar:
    st.markdown("""
        <div style="background-color: #f1f1f1; text-align: center; border-radius: 10px; padding: 2px; margin-bottom: 20px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);">
             <h3 style="color: black; margin: 0;">🔍 <span style="margin-left: 8px;">Data Configuration</span></h3>
        """, unsafe_allow_html=True)
        
    # Ticker input
    company_name_to_key = {v["name"]: k for k, v in COMPANIES.items()}
    selected_name = st.selectbox("Pick a Company", list(company_name_to_key.keys()))
    selected_key = company_name_to_key[selected_name]
    selected_id = COMPANIES[selected_key]["id"]
    
    # Date range selection
    full_stock_data = load_stock(selected_id)
    min_date = full_stock_data.index.min().to_pydatetime().date()
    max_date = full_stock_data.index.max().to_pydatetime().date()

    # Time range selection
    times_in_range = full_stock_data.between_time("11:00", "16:59").index.time
    min_time = min(times_in_range)
    max_time = max(times_in_range)

    default_start = max_date - timedelta(days=1)
    default_time = time(15, 29, 0)
        
    # Ensure default start isn't before min date
    if default_start < min_date:
        default_start = min_date

    start_date = st.date_input(
        "Start Date",
        value=default_start,
        min_value=min_date,
        max_value=max_date - timedelta(days=1)
    )
    start_time = st.time_input(
        "Start Time",
        value=default_time)
        
    # Analysis parameters
    st.header("⚙️ Analysis Parameters")
    lookback_window = st.slider("Volatility Lookback Window (secs)", 1, 60, 20)
    forecast_horizon = st.slider("Forecast Horizon (secs)", 1, 30, 20)
    start_datetime = datetime.combine(start_date, start_time)
    if start_datetime not in full_stock_data.index:
        st.error("⚠️ The selected start time does not exist in the data. Please choose another time.")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"], label_visibility="collapsed")

    def read_uploaded_file(uploaded_file):
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(uploaded_file)
                else:
                    st.error("Unsupported file type.")
                    return None

                # Same processing as load_stock
                df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str)) + \
                                 pd.to_timedelta(df['seconds_in_bucket'], unit='s')
                df.set_index('datetime', inplace=True)
                df.drop(columns=['date', 'time'], inplace=True)
                df['Close'] = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
    
                return df
            except Exception as e:
                st.error(f"⚠️ Failed to read the file: {e}")
                return None
        return None

# Function to create features for our model
def create_features(window_data):
    window_data = window_data.copy()
    
    window_data['WAP'] = (window_data['bid_price1'] * window_data['ask_size1'] + \
                            window_data['ask_price1'] * window_data['bid_size1']) / \
                           (window_data['bid_size1'] + window_data['ask_size1'])
    
    window_data['log_ret'] = window_data['WAP'].transform(lambda x: np.log(x).diff().fillna(0))
    
    for lag in range(1, 6):
        window_data[f'wap_lag_{lag}'] = window_data['WAP'].shift(lag)
        window_data[f'wap_delta_{lag}'] = window_data['WAP'] - window_data[f'wap_lag_{lag}']

    window_data['wap_trend_5s'] = window_data[[f'wap_delta_{i}' for i in range(1, 6)]].mean(axis=1)
    window_data['spread'] = (window_data['ask_price1'] - window_data['bid_price1']) / window_data['ask_price1']

    window_data['spread_lag_1'] = window_data['spread'].shift(1)
    window_data['spread_delta_1'] = window_data['spread'] - window_data[f'spread_lag_1']
    
    window_data['imbalance_velocity'] = ((window_data['bid_size1'] - window_data['ask_size1']) / (window_data['bid_size1'] + window_data['ask_size1'])).diff().rolling(3).mean()
    window_data['vol_weighted_vol'] = window_data['log_ret'].abs() * window_data['bid_size1'].rolling(10).sum()

    return window_data.iloc[-1][[
        col for col in window_data.columns 
        if col not in ['stock_id','time_id', 'bid_price1', 'ask_price1', 'bid_size1', 'ask_size1','bid_price2', 'ask_price2', 'bid_size2', 'ask_size2']
    ]]

# Function to train volatility prediction model
def simulate_real_time_forecasting(full_data, lookback_seconds=200, prediction_horizon=20):
    all_predictions = []

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.03,
        'num_leaves': 81,
        'max_depth': 5,
        'min_data_in_leaf': 20,
        'bagging_freq': 10,
        'lambda_l2': 0,
        'max_bin': 511,
        'feature_fraction': 0.8,
        'verbosity': -1
    }
    min_data_needed = lookback_seconds + prediction_horizon

    for current_pos in tqdm(range(min_data_needed, len(full_data))): 
        
        lookback_window = full_data.iloc[current_pos-lookback_seconds:current_pos]
        future_window = full_data.iloc[current_pos:current_pos+prediction_horizon]
        
        features = create_features(lookback_window)
        future_wap = (future_window['bid_price1'] * future_window['ask_size1'] + 
                     future_window['ask_price1'] * future_window['bid_size1']) / \
                    (future_window['bid_size1'] + future_window['ask_size1'])
        
        future_log_ret = np.log(future_wap).diff().dropna()
        future_vol = np.sqrt((future_log_ret ** 2).sum())
        
        
        if current_pos == min_data_needed:
            X_train = features.to_frame().T  
            y_train = pd.Series([future_vol])
        else:
            X_train = pd.concat([X_train, features.to_frame().T])
            y_train = pd.concat([y_train, pd.Series([future_vol])])
        
        if len(X_train) > 600:  
            X_train = X_train.iloc[-600:]
            y_train = y_train.iloc[-600:]


        lgb_train = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(params, lgb_train, num_boost_round=300)
        
        current_prediction = model.predict(features.to_frame().T)[0]
        
        all_predictions.append({
            'timestamp': full_data.index[current_pos],
            'actual_volatility': future_vol,
            'predicted_volatility': current_prediction
        })

    return pd.DataFrame(all_predictions)

@st.cache_data(ttl=3600)
def model_results():
    df = get_stock_data(selected_id, start_datetime, end_datetime)
    return simulate_real_time_forecasting(df, lookback_seconds=200, prediction_horizon=20)


# Main app execution
if selected_id:
    if uploaded_file:
        filtered_data = read_uploaded_file(uploaded_file)
        st.success("✅ Uploaded file successfully loaded.")
    else:
        filtered_data = get_stock_data(
            selected_id,
            start_datetime=start_datetime - timedelta(seconds=lookback_window),
            end_datetime=start_datetime
        )
        
    tab1, tab2, tab3 = st.tabs(["📈 Dashboard","🧮 Risk Calculator", "🤖 Prediction Model", ])
    
    with tab1:
        if not uploaded_file:
            with st.container():
                company_info = COMPANIES[selected_key]
                st.markdown(f"""
                <div class="card" style="display: flex; align-items: center; gap: 20px; padding: 20px;">
                    <img src="{company_info['logo']}" style="height: 60px; width: auto;"/>
                    <div>
                        <h2 style="margin: 0; color: var(--text);">{company_info['name']} <span style="color: var(--primary);">{company_info['ticker']}</span></h2>
                        <div style="display: flex; gap: 10px; margin-top: 5px;">
                            <span style="background: rgba(26, 115, 232, 0.1); color: var(--primary); padding: 3px 10px; border-radius: 12px; font-size: 12px;">{company_info['sector']}</span>
                            <span style="color: var(--text); opacity: 0.7; font-size: 12px;">NASDAQ • Real-time Data</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with st.container():
                st.markdown("""
                <div class="card" style="padding: 15px;">
                    <h3 style="margin-top: 0; color: var(--text);">Market Snapshot</h3>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
                """, unsafe_allow_html=True)
            
            # Calculate metrics
            last_price = filtered_data['Close'].iloc[-1]
            price_change = filtered_data['Close'].pct_change().iloc[-1]
            spread_pct = (filtered_data['ask_price1'].iloc[-1] - filtered_data['bid_price1'].iloc[-1]) / filtered_data['ask_price1'].iloc[-1] * 100
            liquidity = filtered_data['bid_size1'].iloc[-1] + filtered_data['ask_size1'].iloc[-1]
            vwap = (filtered_data['Close'] * (filtered_data['bid_size1'] + filtered_data['ask_size1'])).sum() / (filtered_data['bid_size1'] + filtered_data['ask_size1']).sum()
            
            # Create metric cards
            cols = st.columns(4)

            metrics = [
                ("Last Price", f"${last_price:.2f}", f"{price_change:.2%}", "Current weighted average price"),
                ("Bid-Ask Spread", f"{spread_pct:.2f}%", "-", "Percentage difference between best bid and ask"),
                ("L1 Liquidity", f"{liquidity:,.0f}", "shares", "Total shares available at top of book"),
                ("Session VWAP", f"${vwap:.2f}", "-", "Volume-weighted average price for current session")
            ]

            for i, (title, value, delta, help_text) in enumerate(metrics):
                with cols[i]:
                    st.markdown(f"""
                    <div class="metric" style="padding: 10px; border-radius: 8px; background: rgba(26, 115, 232, 0.05);">
                        <div style="font-size: 12px; color: var(--text); opacity: 0.8;">{title}</div>
                        <div style="display: flex; align-items: baseline; gap: 5px; margin-top: 5px;">
                            <span style="font-size: 20px; font-weight: 600; color: var(--text);">{value}</span>
                            {f'<span style="font-size: 14px; color: {"#0dba3b" if float(price_change) >=0 else "#f44336"};">{delta}</span>' if delta else ''}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)

        
        with st.container():
            st.markdown("""
            <div class="card" style="padding: 15px;">
                <h3 style="margin-top: 0; color: var(--text);">Market Microstructure</h3>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
            """, unsafe_allow_html=True)
            
            chart_cols = st.columns(3)
            chart_data = [
                ("Price Action", filtered_data['Close'],
                 "line", "#1a73e8"),
                ("Bid-Ask Spread", (filtered_data['ask_price1'] - filtered_data['bid_price1']),
                 "bar", ["#f44336", "#6bc9c0"]),
                ("L1 Volume", (filtered_data['bid_size1'] + filtered_data['ask_size1']),
                 "bar", "#1976D2")
            ]
            
            for i, (title, data, chart_type, color) in enumerate(chart_data):
                with chart_cols[i]:
                    fig = go.Figure()
                    
                    if chart_type == "line":
                        fig.add_trace(go.Scatter(
                            x=filtered_data.index,
                            y=data,
                            line=dict(color=color, width=2)
                        ))
                    else:
                        if isinstance(color, list):
                            # For spread (positive/negative coloring)
                            fig.add_trace(go.Bar(
                                x=filtered_data.index,
                                y=data,
                                marker_color=np.where(data > data.mean(), color[0], color[1])
                            ))
                        else:
                            # For volume
                            fig.add_trace(go.Bar(
                                x=filtered_data.index,
                                y=data,
                                marker_color=color
                            ))
                    
                    fig.update_layout(
                        title=dict(text=title, font=dict(size=14)),
                        height=200,
                        margin=dict(l=0, r=0, t=30, b=0),
                        showlegend=False,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        xaxis_showgrid=False,
                        yaxis_showgrid=True,
                        yaxis_gridcolor="rgba(0,0,0,0.1)"
                    )
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            st.markdown("</div></div>", unsafe_allow_html=True)

        with st.expander("Order Book Depth", expanded=True):
            # Ensure left-to-right: Bid (left), Ask (right)
            bid_col, ask_col = st.columns(2)

            for side, col in [("Bid", bid_col), ("Ask", ask_col)]:
                with col:
                    st.markdown(f"""
                    <div class="metric" style="padding: 10px; border-radius: 8px; background: rgba(26, 115, 232, 0.05); margin-bottom: 10px;">
                        <div style="font-size: 14px; color: var(--text); opacity: 0.8;">{side} Side</div>
                    </div>
                    """, unsafe_allow_html=True)

                    df = pd.DataFrame({
                        'Price': filtered_data[f'{side.lower()}_price1'].iloc[-5:], 
                        'Size': filtered_data[f'{side.lower()}_size1'].iloc[-5:]
                    })

                    bg_color = "#e8f5e9" if side == "Bid" else "#ffebee"

                    styled_df = df.style.format({
                        'Price': "${:.2f}",
                        'Size': "{:,.0f}"
                    }).apply(lambda x: [f'background-color: {bg_color}'] * len(x), axis=1)

                    st.dataframe(styled_df, height=220, use_container_width=True)
    
    with tab2:
        st.subheader("Volatility-Based Position Sizing Calculator")
        st.markdown("Use this tool to estimate your position size based on market volatility and your risk preferences.")

        capital = st.number_input("💰 Total Capital ($)", value=100000, help="The total amount you're allocating for trading.")
        risk_per_trade = st.slider("⚖️ Risk per Trade (%)", 0.1, 5.0, 1.0, help="How much of your capital you're willing to risk per trade.")
        forecast_vol = st.number_input(
            "📉 Forecasted Volatility (%)",
            value=0.03,  # Show as % (e.g., 0.03 = 0.0003 raw)
            format="%.4f",
            help="Enter model-predicted short-term volatility. Typical values might be below 0.1% for high-frequency data."
        )

        risk_dollar = capital * (risk_per_trade / 100)
        max_position = risk_dollar / (forecast_vol / 100)

        st.markdown(f"""
        ### 📊 Results:
        - **Maximum risk per trade:** ${risk_dollar:,.2f}  
        - **Recommended position size:** {max_position:,.0f} units (based on volatility)
        """)
        st.markdown("**Tip:** To estimate safe position sizing based on this forecast, switch to the 'Model Prediction' tab and input the predicted volatility.")
    
    with tab3:
        st.header("Volatility Prediction Model")
        n_past = lookback_window
        n_focus = forecast_horizon
        
        selected = datetime.combine(start_date, start_time)
        window_secs = 90
        start_sim = selected - timedelta(seconds=window_secs)
        end_sim = selected + timedelta(seconds=window_secs)
        
        stock_data = get_stock_data(selected_id, start_sim, end_sim)
        results = simulate_real_time_forecasting(
            stock_data,
            lookback_seconds=n_past,
            prediction_horizon=n_focus
        )

        
        forecast_start_time = pd.to_datetime(selected)
        time_diffs = np.abs(results['timestamp'] - forecast_start_time)
        closest_idx = time_diffs.idxmin()
        selected_timestamp = results.loc[closest_idx, 'timestamp']
        
        lookback_start = selected_timestamp - pd.Timedelta(seconds=n_past)
        forecast_end = selected_timestamp + pd.Timedelta(seconds=n_focus)

        
        # Get indices with boundary checks
        lookback_start_idx = max(0, results['timestamp'].searchsorted(lookback_start))
        forecast_end_idx = min(len(results), results['timestamp'].searchsorted(forecast_end))
        
        lookback_data = results.iloc[lookback_start_idx:closest_idx]
        forecast_data = results.iloc[closest_idx:forecast_end_idx]
        
        # Combine for plotting
        plot_data = pd.concat([lookback_data, forecast_data])
        fig = go.Figure()
        
        colors = {
            'actual': '#026ab5',
            'predicted': '#ba221a',
            'highlight': '#8eb5de',
            'background': '#fafdff',
            'grid': '#e0e0e0',
            'text': '#292F36',
            'safe': '#2ecc71',
            'caution': '#f39c12',
            'danger': '#e74c3c' }

        # volatility thresholds
        vol_mean = lookback_data['actual_volatility'].mean()
        vol_std = lookback_data['actual_volatility'].std()
        high_vol_threshold = vol_mean + 1.5 * vol_std
        extreme_vol_threshold = vol_mean + 2.5 * vol_std

            
        # Actual Volatility (Full) - with gradient fill to prediction line
        fig.add_trace(go.Scatter(
            x=lookback_data['timestamp'],
            y=lookback_data['actual_volatility'],
            mode='lines',
            name='Historical Volatility',
            line=dict(color=colors['actual'], width=3)
        ))
        # Predicted Volatility with confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_data['timestamp'],
            y=forecast_data['predicted_volatility'],
            mode='lines+markers',
            name='Predicted Volatility',
            line=dict(color=colors['predicted'], width=3),
            marker=dict(size=6, color=colors['predicted'])
        ))

        # Add warning emojis for high volatility points
        high_vol_points = forecast_data[forecast_data['predicted_volatility'] > high_vol_threshold]
        extreme_vol_points = forecast_data[forecast_data['predicted_volatility'] > extreme_vol_threshold]

        # Regular high volatility warnings
        fig.add_trace(go.Scatter(
            x=high_vol_points['timestamp'],
            y=high_vol_points['predicted_volatility'],
            mode='markers+text',
            marker=dict(size=8, symbol='triangle-up', color=colors['caution']),
            text="⚠️",
            textposition="top center",
            textfont=dict(size=16),
            name='High Volatility Warning',
            hoverinfo='text',
            hovertext='High Volatility Expected'
        ))
        
        # Extreme volatility warnings
        fig.add_trace(go.Scatter(
            x=extreme_vol_points['timestamp'],
            y=extreme_vol_points['predicted_volatility'],
            mode='markers+text',
            marker=dict(size=8, symbol='diamond', color=colors['danger']),
            text="🚨",
            textposition="top center",
            textfont=dict(size=18),
            name='Extreme Volatility Warning',
            hoverinfo='text',
            hovertext='EXTREME Volatility Expected'
        ))
        error = forecast_data['predicted_volatility'].std()

         # Lower Bound (start the fill)
        fig.add_trace(go.Scatter(
            x=forecast_data['timestamp'],
            y=forecast_data['predicted_volatility'] - error,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Upper Bound with fill to lower
        fig.add_trace(go.Scatter(
            x=forecast_data['timestamp'],
            y=forecast_data['predicted_volatility'] + error,
            mode='lines',
            fill='tonexty',  # fill between this and previous trace
            fillcolor='rgba(186, 34, 26, 0.2)',  # translucent red
            line=dict(width=0),
            showlegend=True,
            name='Confidence Interval',
            hoverinfo='skip'
        ))

        # Highlight forecast period
        fig.add_vrect(
            x0=selected_timestamp,
            x1=forecast_data['timestamp'].iloc[-1],
            fillcolor=colors['highlight'],
            opacity=0.1,
            layer="below",
            line_width=0,
            annotation_text=f"{n_focus}-Second Forecast Period",
            annotation_position="top left",
            annotation_font_size=12,
            annotation_font_color=colors['text']
        )

        fig.add_shape(
            type="line",
            x0=selected_timestamp,
            y0=lookback_data['actual_volatility'].min(),
            x1=selected_timestamp,
            y1=lookback_data['actual_volatility'].max(),
            line=dict(color=colors['text'], width=2, dash="dot"),
            name="Forecast Start"
        )

        # Add annotation for forecast start
        fig.add_annotation(
            x=selected_timestamp,
            y=lookback_data['actual_volatility'].max(),
            text=" Forecast Start ",
            showarrow=True,
            arrowhead=1,
            ax=-50,
            ay=-30,
            bgcolor='white',
            bordercolor=colors['text'],
            borderwidth=1
        )

        # Add horizontal line for high volatility threshold
        fig.add_hline(y=high_vol_threshold, line_dash="dot", 
                     line_color=colors['caution'], opacity=0.7,
                     annotation_text=f"High Vol Threshold ({high_vol_threshold:.6f})", 
                     annotation_position="bottom right")
        
        # Add horizontal line for extreme volatility threshold
        fig.add_hline(y=extreme_vol_threshold, line_dash="dot", 
                     line_color=colors['danger'], opacity=0.7,
                     annotation_text=f"Extreme Vol Threshold ({extreme_vol_threshold:.6f})", 
                     annotation_position="bottom right")


        # Final layout with enhanced styling
        fig.update_layout(
            title={
                'text': f'<b>Volatility Forecast with Trading Signals</b><br><span style="font-size:12px">Next {n_focus} periods | {selected_name}</span>',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=20, color=colors['text'])
            },
            xaxis_title='<b>Time</b>',
            yaxis_title='<b>Volatility</b>',
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.7)',
                bordercolor=colors['grid'],
                borderwidth=1,
                orientation="h"
            ),
            width=1000,
            height=600,
            margin=dict(t=80, l=80, r=80, b=80),
            hovermode='x unified',
            xaxis=dict(
                showgrid=True,
                gridcolor=colors['grid'],
                gridwidth=1,
                tickmode='auto',
                showline=True,
                linecolor=colors['text'],
                linewidth=1,
                nticks=40,
                mirror=True
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=colors['grid'],
                gridwidth=1,
                showline=True,
                linecolor=colors['text'],
                linewidth=1,
                mirror=True
            )
        )

        # Add custom hover template
        fig.update_traces(
            hovertemplate='<b>%{x|%Y-%m-%d %H:%M}</b><br>Volatility: %{y:.7f}<extra></extra>'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Add trader guidance based on volatility predictions
        max_pred_vol = forecast_data['predicted_volatility'].max()
        formatted_time = forecast_end.strftime("%Y-%m-%d %H:%M:%S")
        
        if max_pred_vol > extreme_vol_threshold:
            st.markdown(f"""
            <div style="
                color: #ffffff;
                background: #ffdede;
                border-left: 5px solid #ff0000;
                padding: 16px;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                margin: 12px 0;
                font-family: San Francisco, sans-serif;
            ">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                    <span style="font-size: 22px;">🚨</span>
                    <h3 style="margin: 0;">EXTREME VOLATILITY ALERT at {formatted_time}</h3>
                </div>
                <ul style="margin: 8px 0 0 20px; padding-left: 5px;">
                    <li style="margin-bottom: 6px;">Expect large, rapid price movements</li>
                    <li style="margin-bottom: 6px;">Reduce position sizes significantly</li>
                    <li style="margin-bottom: 6px;">Widen stop-loss orders</li>
                    <li style="margin-bottom: 6px;">Avoid entering new positions during spikes</li>
                    <li>Focus on liquidity provision if market-making</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        elif max_pred_vol > high_vol_threshold:
            st.markdown(f"""
            <div style="
                color: #856404;
                background: #f2eed3;
                border-left: 5px solid #ffc107;
                padding: 16px;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                margin: 12px 0;
                font-family: San Francisco, sans-serif;
            ">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                    <span style="font-size: 22px;">⚠️</span>
                    <h3 style="margin: 0;">HIGH VOLATILITY ALERT at {formatted_time}</h3>
                </div>
                <ul style="margin: 8px 0 0 20px; padding-left: 5px;">
                    <li style="margin-bottom: 6px;">Be prepared for increased price swings</li>
                    <li style="margin-bottom: 6px;">Tighten stop-loss levels</li>
                    <li style="margin-bottom: 6px;">Consider taking partial profits</li>
                    <li style="margin-bottom: 6px;">Favor limit orders over market orders</li>
                    <li>Monitor order flow closely</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="
                color: #036b20;
                background: #d3f2db;
                border-left: 5px solid #28a745;
                padding: 16px;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                margin: 12px 0;
                font-family: San Francisco, sans-serif;
            ">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                    <span style="font-size: 22px;">✅</span>
                    <h3 style="margin: 0;">NORMAL VOLATILITY CONDITIONS at {formatted_time}</h3>
                </div>
                <ul style="margin: 8px 0 0 20px; padding-left: 5px;">
                    <li style="margin-bottom: 6px;">Standard trading strategies can be employed</li>
                    <li style="margin-bottom: 6px;">Maintain normal position sizing</li>
                    <li style="margin-bottom: 6px;">Continue monitoring for volatility regime changes</li>
                    <li>Watch for order book imbalances</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

##
##        # Footer
##        st.markdown("""
##            <div style="margin-top: 40px; text-align: center; color: var(--text); opacity: 0.7; font-size: 12px;">
##                <hr style="border: 0.5px solid var(--border); margin-bottom: 10px;">
##                ECHOVOL20 • Institutional Analytics Platform • Data Source: Optiver
##                <br>© 2024 Quant Analytics LLC • All rights reserved
##            </div>
##            """, unsafe_allow_html=True)

