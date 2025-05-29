# EchoVol20 Volatility Prediction App

## 📖 Overview
The **EchoVol20** is a powerful tool designed to analyze stock market data and predict volatility using advanced machine learning models. It provides real-time forecasting, visualizations, and actionable insights for traders and analysts.

---

## 🚀 Features
- **Real-Time Volatility Forecasting**: Predict future volatility using historical data.
- **Customizable Parameters**: Adjust lookback windows and forecast horizons to suit your analysis.
- **File Upload Support**: Upload custom datasets (CSV or Excel) for personalized predictions.
- **Interactive Visualizations**: View volatility trends, confidence intervals, and trading signals.
- **Trading Plan Analysis**: Get insights into position sizing, risk assessment, and expected market moves.

---

## 🛠️ How to Use
1. **Select a Stock**:
   - Choose a stock from the sidebar dropdown menu.
2. **Set Analysis Parameters**:
   - Adjust the lookback window and forecast horizon using the sliders.
   - Select a start date and time for the analysis.
3. **Upload a File (Optional)**:
   - Upload a CSV or Excel file with custom data for predictions.
   - Ensure the file contains the required columns (e.g., `bid_price1`, `ask_price1`, etc.).
4. **Run the Analysis**:
   - View real-time forecasts, volatility thresholds, and trading signals.
5. **Export Results**:
   - Download the predictions as a CSV file for further analysis.

---

## 📂 File Requirements
If uploading a custom dataset, ensure the file meets the following requirements:
- **File Format**: CSV or Excel (`.csv`, `.xlsx`).
- Follow the same format as the `small_stock_data.csv`
- ✅ To recreate the dataset used in the app, run the script app_dataset_merge.py, which merges time_id_reference.csv and order_book_feature.csv into the required input format.
---

## 🖥️ Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/emilyl523/OPTIVER11/
    ```

2. Create a vitual environment:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  
    # On Windows: .venv\Scripts\activate
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the app:
    ```bash
    streamlit run volapp_final.py
    ```

## 📊 Example Use Cases
- Traders: Identify periods of high or extreme volatility to adjust trading strategies.
- Analysts: Analyze historical data to understand market behavior.
- Researchers: Test custom datasets to evaluate model performance.
