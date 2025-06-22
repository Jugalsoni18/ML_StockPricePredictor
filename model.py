import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from typing import Dict, Any
import warnings

warnings.filterwarnings('ignore')


class StockPredictor:
    """
    A class to encapsulate the stock prediction model, including
    feature engineering, training, and prediction.
    """
    
    def __init__(self, threshold: float = 1.0, model_type: str = 'RandomForest'):
        """
        Initialize the predictor.
        
        Args:
            threshold: Price change threshold to trigger a buy/sell signal.
            model_type: The type of model to use ('RandomForest' or 'XGBoost').
        """
        self.threshold = threshold
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.features = None
        self.is_trained = False
        self.feature_columns = []
        self.metrics = {}
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create a rich set of features from the stock data.
        
        Args:
            data: DataFrame with OHLCV stock data.
            
        Returns:
            DataFrame with added features.
        """
        df = data.copy()
        
        # Add technical indicators using pandas_ta
        df.ta.sma(length=20, append=True)
        df.ta.ema(length=20, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.macd(append=True)
        df.ta.bbands(append=True)
        df.ta.atr(append=True)
        df.ta.adx(append=True)
        
        # Add custom features
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
        
        # Add lag features for 'Close' price
        for i in range(1, 6):
            df[f'Close_Lag_{i}'] = df['Close'].shift(i)
        
        # Create target variable (next day's Close price)
        df['Target'] = df['Close'].shift(-1)
        
        # Drop rows with NaN values resulting from feature creation
        df = df.dropna()
        
        return df
    
    def train(self, data: pd.DataFrame) -> dict:
        """
        Train the stock prediction model.
        
        Args:
            data: DataFrame with historical stock data.
            
        Returns:
            Dictionary with training results.
        """
        try:
            # 1. Feature Engineering
            df_features = self.create_features(data)
            
            if df_features.empty:
                return {'success': False, 'error': "Not enough data to create features."}

            # 2. Define Features (X) and Target (y)
            self.features = [col for col in df_features.columns if col != 'Target' and col != 'Date']
            X = df_features[self.features]
            y = df_features['Target']

            # 3. Time-aware train/test split
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # 4. Feature Scaling
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # 5. Hyperparameter Tuning with TimeSeriesSplit
            if self.model_type == 'RandomForest':
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [10, 20],
                    'min_samples_split': [2, 5]
                }
                model = RandomForestRegressor(random_state=42)
            elif self.model_type == 'XGBoost':
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5],
                    'learning_rate': [0.05, 0.1]
                }
                model = XGBRegressor(random_state=42, objective='reg:squarederror')
            else:
                raise ValueError("Unsupported model type")

            tscv = TimeSeriesSplit(n_splits=3)
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, n_jobs=-1, scoring='r2')
            grid_search.fit(X_train_scaled, y_train)

            self.model = grid_search.best_estimator_

            # 6. Evaluation
            y_pred = self.model.predict(X_test_scaled)
            
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            # Feature Importance
            feature_importance = pd.Series(self.model.feature_importances_, index=self.features).sort_values(ascending=False)

            self.metrics = {
                'test_r2': r2,
                'test_rmse': rmse,
                'test_mae': mae,
                'train_samples': len(X_train)
            }
            
            self.is_trained = True
            
            return {
                'success': True,
                'metrics': self.metrics,
                'feature_importance': feature_importance.to_dict()
            }

        except Exception as e:
            return {'success': False, 'error': f"Model training failed: {str(e)}"}
    
    def predict_next_price(self, data: pd.DataFrame, currency_symbol: str = "$") -> dict:
        """
        Predict the next day's closing price.
        
        Args:
            data: DataFrame with OHLCV stock data, should have enough history for feature creation.
            currency_symbol: Currency symbol to use in signal reason string.
            
        Returns:
            Dictionary with prediction results.
        """
        try:
            if self.model is None or self.scaler is None or self.features is None:
                return {'success': False, 'error': "Model is not trained yet."}

            # Create features for the most recent data point
            df_with_features = self.create_features(data)
            
            if df_with_features.empty:
                return {'success': False, 'error': "Could not generate features for prediction."}

            last_features = df_with_features[self.features].iloc[-1:]
            
            # Scale the features
            last_features_scaled = self.scaler.transform(last_features)

            # Make prediction
            predicted_price = self.model.predict(last_features_scaled)[0]
            current_price = data['Close'].iloc[-1]
            
            # Calculate price difference and generate signal
            price_difference = predicted_price - current_price
            
            if price_difference > self.threshold:
                action = 'BUY'
                reason = f"Predicted price is {currency_symbol}{price_difference:.2f} higher than current price"
            elif price_difference < -self.threshold:
                action = 'SELL'
                reason = f"Predicted price is {currency_symbol}{abs(price_difference):.2f} lower than current price"
            else:
                action = 'HOLD'
                reason = f"Price change is within the {currency_symbol}{self.threshold} threshold"

            # Calculate confidence based on recent volatility
            confidence = 0.5
            if 'Volatility' in df_with_features.columns:
                recent_volatility = df_with_features['Volatility'].iloc[-1]
                historical_volatility = df_with_features['Volatility'].mean()
                if historical_volatility > 0:
                    # Confidence is inversely proportional to recent volatility compared to the average
                    confidence = max(0.1, min(0.9, 1 - (recent_volatility / historical_volatility)))
            
            return {
                'success': True,
                'predicted_price': predicted_price,
                'current_price': current_price,
                'price_difference': price_difference,
                'signal': {
                    'action': action,
                    'reason': reason
                },
                'confidence': confidence
            }

        except Exception as e:
            return {'success': False, 'error': f"Prediction failed: {str(e)}"}
    
    def generate_signal(self, price_difference: float) -> Dict[str, str]:
        """
        Generate investment signal based on price difference.
        
        Args:
            price_difference: Difference between predicted and current price
            
        Returns:
            Dictionary with signal and reasoning
        """
        if price_difference > self.threshold:
            return {
                'action': 'BUY',
                'reason': f'Predicted price is ${price_difference:.2f} higher than current price'
            }
        elif price_difference < -self.threshold:
            return {
                'action': 'SELL',
                'reason': f'Predicted price is ${abs(price_difference):.2f} lower than current price'
            }
        else:
            return {
                'action': 'HOLD',
                'reason': f'Predicted price change (${price_difference:.2f}) is within threshold'
            }
    
    def _calculate_confidence(self, data: pd.DataFrame) -> float:
        """
        Calculate prediction confidence based on recent volatility.
        
        Args:
            data: DataFrame with stock data
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            if 'Volatility' not in data.columns:
                return 0.5
            
            recent_volatility = data['Volatility'].tail(10).mean()
            historical_volatility = data['Volatility'].mean()
            
            # Lower volatility = higher confidence
            if historical_volatility > 0:
                confidence = max(0.1, min(0.9, 1 - (recent_volatility / historical_volatility)))
            else:
                confidence = 0.5
            
            return confidence
            
        except Exception:
            return 0.5
    
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        Get top N most important features.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary with feature names and importance scores
        """
        if not self.is_trained:
            return {}
        
        feature_importance = dict(zip(
            self.feature_columns, 
            self.model.feature_importances_
        ))
        
        # Sort by importance and return top N
        sorted_features = sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return dict(sorted_features[:top_n])