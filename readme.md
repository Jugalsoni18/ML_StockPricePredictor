# 🔮 Smart Stock Predictor with Actionable Signals

## 📘 Project Description
This project is a sophisticated, web-based stock price predictor designed for traders, investors, and data science enthusiasts. It leverages machine learning to forecast the next-day closing price for publicly traded stocks and generates clear, actionable **Buy, Sell, or Hold** signals. The application automatically sources data from `yfinance` for Indian markets and the `Twelve Data API` for U.S. markets, presenting the results in a clean, interactive, and real-time Streamlit interface. It's built for educational purposes, to demonstrate a practical end-to-end machine learning workflow, and to serve as an analytical tool for market analysis.

---

##  Key Features

- **Intelligent Price Prediction**: Utilizes `XGBoost` (primary) and `Random Forest` (secondary) models to predict the next-day stock closing price with a high degree of accuracy.
- **Advanced Feature Engineering**: Enriches the input data with over 15 technical indicators, including RSI, MACD, Bollinger Bands, moving averages, volatility, and lag features to improve model performance.
- **Actionable Trading Signals**: Generates clear **Buy / Sell / Hold** recommendations based on the predicted price movement relative to a user-defined threshold.
- **Dual Data Source**: Seamlessly fetches data from `yfinance` for Indian stocks (e.g., `TCS.NS`) and the `Twelve Data API` for U.S. stocks (e.g., `AAPL`).
- **Interactive UI**: A fully interactive and user-friendly web interface built with Streamlit, allowing for easy input, real-time predictions, and dynamic charting.
- **Developer Mode for A/B Testing**: Includes a hidden "Dev Mode" toggle to silently run a Random Forest model alongside the primary XGBoost model and display a side-by-side performance comparison—ideal for validation and analysis.
- **Robust Error Handling**: Features ticker symbol validation and user-friendly error messages to ensure a smooth user experience.

---

## 🧰 Tech Stack

- **Core Language**: Python 3.10+
- **Web Framework**: [Streamlit](https://streamlit.io/) – For the interactive web UI.
- **Data Handling**: [pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) – For data manipulation and numerical operations.
- **Machine Learning**: 
  - [Scikit-learn](https://scikit-learn.org/) – For data preprocessing and model evaluation.
  - [XGBoost](https://xgboost.readthedocs.io/en/stable/) – For the primary prediction model.
- **Technical Analysis**: [pandas-ta](https://github.com/twopirllc/pandas-ta) – For generating technical indicators.
- **Data Sources**:
  - [yfinance](https://pypi.org/project/yfinance/) – For Indian stock data.
  - [Twelve Data API](https://twelvedata.com/) – For U.S. stock data via `requests`.
- **Charting**: [Plotly](https://plotly.com/python/) – For creating interactive price and volume charts.

---

## 💾 Installation Instructions

Follow these steps to set up and run the project locally.

**1. Clone the Repository**
```bash
git clone https://github.com/your-username/smart-stock-predictor.git
cd smart-stock-predictor
```

**2. Create and Activate a Virtual Environment** (Recommended)
```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Set Up API Key**
This project requires a Twelve Data API key for fetching U.S. stock data.
- Create a file named `.env` in the root directory of the project.
- Add your API key to the file as follows:
  ```
  TWELVE_DATA_API_KEY="YOUR_API_KEY_HERE"
  ```
- You can get a free API key from [Twelve Data](https://twelvedata.com/apikey).

**5. Run the Application**
```bash
streamlit run app.py
```
The application will open in your default web browser.

---

## 🧪 Usage Instructions

1.  **Enter a Ticker Symbol**: Input the stock ticker you want to analyze (e.g., `AAPL` for Apple or `RELIANCE.NS` for Reliance Industries).
2.  **Set Signal Threshold**: Adjust the slider to define the minimum price change required to trigger a "Buy" or "Sell" signal.
3.  **Click Predict**: Press the "Predict Stock Price" button to start the analysis.

The application will display:
- The current and predicted next-day price.
- The recommended investment signal (Buy/Sell/Hold).
- An interactive price chart with the prediction plotted.
- Key performance metrics (R² Score, RMSE) for the model.
- An analysis of the most important features used in the prediction.

---

## 🕵️‍♂️ Developer Mode

For development and analysis, a **"Compare Models (Dev Mode)"** checkbox is available in the sidebar. When enabled, the app will:
- Silently train a `Random Forest` model in the background alongside the `XGBoost` model.
- Display a side-by-side comparison of the performance metrics for both models.
- Plot both models' predictions on the price chart for direct visual comparison.

---

## 📸 Demo Screenshot Placeholder

📷 *A screenshot of the app's user interface would go here. A polished image showcasing the input sidebar, the main prediction results, and the interactive chart provides an excellent first impression.*

---

## 🗂️ Folder Structure

```
smart-stock-predictor/
├── app.py                # Main Streamlit application file (UI and orchestration)
├── datafetcher.py        # Handles data fetching from yfinance and Twelve Data
├── model.py              # Core machine learning logic (feature engineering, training, prediction)
├── tune_model.py         # Standalone script for hyperparameter tuning
├── requirements.txt      # Project dependencies
├── .env.example          # Example environment file for API key
└── README.md             # This file
```

---

## 🔐 API Key Requirement

⚠️ **Note**: To use this application with U.S. stocks (e.g., `AAPL`, `MSFT`, `GOOGL`), you must obtain a **free API key** from [Twelve Data](https://twelvedata.com/apikey). Once you have the key, create a `.env` file in the project's root directory and add the line `TWELVE_DATA_API_KEY="YOUR_API_KEY_HERE"`. The application will automatically load the key from this file. Indian stock data via `yfinance` does not require an API key.

---

## 📄 License

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.
