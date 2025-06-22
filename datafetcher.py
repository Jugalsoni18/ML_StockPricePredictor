import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
import time
from typing import Optional, Tuple
from dotenv import load_dotenv
import os
import re


class DataValidationException(Exception):
    """Custom exception for data validation errors during the fetch process."""
    pass


class DataFetcher:
    """Handles fetching stock data from different sources."""
    
    def __init__(self):
        """
        Initialize the data fetcher.
        It loads the Twelve Data API key from a .env file.
        """
        load_dotenv()
        self.twelve_data_api_key = os.getenv("TWELVE_DATA_API_KEY")
        self.twelve_data_base_url = "https://api.twelvedata.com"
    
    def is_indian_stock(self, ticker: str) -> bool:
        """
        Check if the ticker is for an Indian stock.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            True if Indian stock, False otherwise
        """
        return ticker.endswith('.NS') or ticker.endswith('.BO')
    
    def test_yfinance_connection(self) -> bool:
        """
        Test if yfinance is working properly.
        
        Returns:
            True if yfinance is working, False otherwise
        """
        try:
            # Test with a well-known US index to check general connectivity
            print("Testing yfinance with S&P 500 index (^GSPC)...")
            test_stock = yf.Ticker("^GSPC")
            test_data = test_stock.history(period="5d")
            
            if test_data.empty:
                # Fallback to another major index
                print("yfinance connection test failed with ^GSPC. Trying with FTSE 100 (^FTSE)...")
                test_stock = yf.Ticker("^FTSE")
                test_data = test_stock.history(period="5d")
            
            if test_data.empty:
                print("yfinance connection test failed: No data for major indices.")
                return False
            
            print("yfinance connection test successful.")
            return True
        except Exception as e:
            print(f"yfinance connection test failed with exception: {str(e)}")
            return False
    
    def fetch_indian_stock_data(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """
        Fetch Indian stock data using yfinance. It requires a .NS or .BO suffix.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'TCS.NS')
            period: Time period for data ('1y', '2y', '5y', etc.)
            
        Returns:
            DataFrame with stock data
            
        Raises:
            Exception: If data fetching fails
        """
        try:
            # For Indian stocks, the ticker MUST be valid with its suffix.
            # We no longer try to strip the suffix as it leads to ambiguity (e.g., AAPL.NS vs AAPL).
            print(f"Attempting to fetch data for Indian ticker: {ticker}")
            stock = yf.Ticker(ticker)
            
            # Use history first as it's a more reliable check for data existence than .info
            data = stock.history(period=period)
            
            if data.empty:
                # If no data, try getting info as a fallback validation to give a better error.
                info = stock.info
                if not info or info.get('regularMarketPrice') is None:
                    raise Exception(f"Ticker '{ticker}' seems invalid or is not available on Yahoo Finance.")
                # If info exists but history is empty, it might be a delisted stock or have no recent data.
                raise Exception(f"No historical data found for ticker '{ticker}' for the period '{period}'. It might be delisted.")
            
            # Reset index to make Date a column
            data.reset_index(inplace=True)
            
            # Ensure we have the required columns
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
            if not all(col in data.columns for col in required_columns):
                 raise Exception(f"Data from yfinance for {ticker} is missing required columns.")

            # Ensure 'Volume' column exists, if not, fill with 0
            if 'Volume' not in data.columns:
                data['Volume'] = 0
            
            # Drop rows with NaN in key columns to ensure data quality
            data.dropna(subset=required_columns, inplace=True)
            if data.empty:
                raise DataValidationException(f"No valid data remaining for {ticker} after cleaning.")

            print(f"Successfully fetched data for {ticker} with {len(data)} rows")
            return data
            
        except Exception as e:
            # Re-raise as a generic exception to be caught by the main dispatcher
            raise Exception(f"Failed to fetch data for Indian stock '{ticker}': {str(e)}")
    
    def _validate_data_quality(self, data: pd.DataFrame, ticker: str):
        """
        Performs a series of data quality checks.
        Raises DataValidationException if any check fails.
        """
        if data is None or data.empty:
            raise DataValidationException(f"No data could be retrieved for ticker '{ticker}'.")

        # Check for sufficient data
        if len(data) < 50:
            raise DataValidationException(f"Insufficient historical data for '{ticker}' (found {len(data)} records, need at least 50).")

        # Check for invalid prices (zero or negative)
        if (data[['Open', 'High', 'Low', 'Close']] <= 0).any().any():
            raise DataValidationException(f"Historical data for '{ticker}' contains invalid (zero or negative) prices.")

        # Check for unrealistically low prices
        latest_price = data['Close'].iloc[-1]
        is_indian = self.is_indian_stock(ticker)
        
        price_threshold = 10 if is_indian else 1
        currency = "₹" if is_indian else "$"

        if latest_price < price_threshold:
            raise DataValidationException(
                f"Price for '{ticker}' is unrealistically low ({currency}{latest_price:.2f}). Stocks below {currency}{price_threshold} are not supported."
            )

    def fetch_us_stock_data(self, ticker: str, months: int = 24) -> pd.DataFrame:
        """
        Fetch US stock data using Twelve Data API.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            months: Number of months of historical data
            
        Returns:
            DataFrame with stock data
            
        Raises:
            Exception: If data fetching fails or API key is missing
        """
        if not self.twelve_data_api_key:
            raise Exception("Twelve Data API key not found. Please add it to your .env file.")
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=months * 30)
            
            # Prepare API request
            url = f"{self.twelve_data_base_url}/time_series"
            params = {
                'symbol': ticker,
                'interval': '1day',
                'apikey': self.twelve_data_api_key,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'format': 'JSON'
            }
            
            # Make API request with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, params=params, timeout=30)
                    response.raise_for_status()
                    data = response.json()
                    break
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        raise Exception(f"API request failed after {max_retries} attempts: {str(e)}")
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            # Validate API response
            if 'values' not in data:
                if 'message' in data:
                    raise Exception(f"API Error: {data['message']}")
                elif 'status' in data and data['status'] == 'error':
                    raise Exception(f"API Error: {data.get('message', 'Unknown error')}")
                else:
                    raise Exception(f"Invalid API response format for {ticker}")
            
            if not data['values']:
                raise Exception(f"No data returned for ticker {ticker}")
            
            # Convert to DataFrame
            df = pd.DataFrame(data['values'])
            
            # Convert datetime and numeric columns
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.rename(columns={'datetime': 'Date'})
            
            # Convert price columns to numeric
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert volume to numeric
            if 'volume' in df.columns:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            
            # Rename columns to match yfinance format
            column_mapping = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            df = df.rename(columns=column_mapping)
            
            # Sort by date (oldest first)
            df = df.sort_values('Date').reset_index(drop=True)
            
            # Ensure we have the required columns
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
            if not all(col in df.columns for col in required_columns):
                raise Exception(f"Missing required columns in data for {ticker}")
            
            return df
            
        except Exception as e:
            print(f"yfinance fallback also failed for {ticker}: {e}")
            raise DataValidationException(f"Failed to fetch data for '{ticker}' from all available sources. Please check the ticker symbol and your internet connection.") from e

    def _check_yfinance_ticker_exists(self, ticker: str) -> bool:
        """Lightweight check to see if a yfinance ticker likely exists."""
        try:
            stock = yf.Ticker(ticker)
            # A valid ticker usually has a 'regularMarketPrice' in its info.
            if stock.info and stock.info.get('regularMarketPrice') is not None:
                return True
            # Fallback: check if we can get any history. Empty history means it's likely invalid.
            if not stock.history(period="7d").empty:
                return True
            return False
        except Exception:
            return False

    def _check_twelvedata_ticker_exists(self, ticker: str) -> bool:
        """Lightweight check to see if a Twelve Data ticker exists."""
        if not self.twelve_data_api_key:
            return False # Cannot check without an API key
        try:
            url = f"{self.twelve_data_base_url}/stocks?symbol={ticker}"
            params = {'apikey': self.twelve_data_api_key}
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            return 'data' in data and len(data['data']) > 0
        except requests.exceptions.RequestException:
            return False

    def fetch_stock_data(self, ticker: str, years: int = 2) -> Tuple[pd.DataFrame, str]:
        """
        Fetches stock data from the appropriate source based on the ticker.
        This function acts as a dispatcher to ensure consistent data handling.
        """
        period = f"{years}y"

        # --- For Indian Stocks, use the specialized fetcher ---
        if self.is_indian_stock(ticker):
            try:
                print(f"Fetching Indian stock data for {ticker} using specialized yfinance handler.")
                data = self.fetch_indian_stock_data(ticker, period=period)
                return data, "yfinance (India)"
            except Exception as e:
                print(f"Failed to fetch Indian stock data for {ticker}: {e}")
                raise DataValidationException(f"Could not retrieve data for Indian ticker '{ticker}'. Please check the ticker symbol (e.g., 'TCS.NS').") from e
        
        # --- For US/Global Stocks ---
        # 1. Try Twelve Data first if an API key is provided
        if self.twelve_data_api_key:
            try:
                print(f"Fetching global stock data for {ticker} using Twelve Data API.")
                data = self.fetch_us_stock_data(ticker, months=years * 12)
                return data, "Twelve Data"
            except Exception as e:
                print(f"Twelve Data API failed for {ticker}: {e}. Falling back to yfinance.")

        # 2. Fallback to yfinance for non-Indian stocks
        try:
            print(f"Fetching global stock data for {ticker} using yfinance as fallback.")
            data = yf.download(ticker, period=period, progress=False)
            
            if data.empty:
                raise DataValidationException(f"No data returned from yfinance for ticker '{ticker}'. It may be invalid or delisted.")
            
            # IMPORTANT: Reset index to make 'Date' a column for yfinance data
            data.reset_index(inplace=True)
            
            # Ensure 'Date' column is not timezone-aware, which can cause issues with Plotly
            if pd.api.types.is_datetime64_any_dtype(data['Date']):
                data['Date'] = data['Date'].dt.tz_localize(None)

            # Final check for required columns
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
            if not all(col in data.columns for col in required_columns):
                 raise DataValidationException(f"Data from yfinance for {ticker} is missing required columns.")
            
            # Drop rows with NaN in key columns to ensure data quality
            data.dropna(subset=required_columns, inplace=True)
            if data.empty:
                raise DataValidationException(f"No valid data remaining for {ticker} after cleaning.")

            return data, "yfinance"
        except Exception as e:
            print(f"yfinance fallback also failed for {ticker}: {e}")
            raise DataValidationException(f"Failed to fetch data for '{ticker}' from all available sources. Please check the ticker symbol and your internet connection.") from e

    def fetch_and_validate_data(self, ticker: str, years: int = 2) -> Tuple[pd.DataFrame, str]:
        """
        A wrapper function that fetches data and then runs validation checks.
        """
        data, source = self.fetch_stock_data(ticker, years)
        self._validate_data_quality(data, ticker)
        return data, source

    def validate_ticker(self, ticker: str) -> bool:
        """
        Validate if a ticker symbol is properly formatted.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            True if valid format, False otherwise
        """
        if not ticker or not isinstance(ticker, str):
            return False
        
        ticker = ticker.strip()
        if len(ticker) < 1 or len(ticker) > 20:
            return False
        
        # Basic validation - alphanumeric with dots and hyphens allowed
        pattern = r'^[A-Za-z0-9.-]+$'
        return bool(re.match(pattern, ticker))