import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, List, Tuple
import logging

class BreakoutPredictor:
    def __init__(
        self,
        lookback_period: int = 20,
        min_samples: int = 100,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6
    ):
        """
        Initialize Breakout Predictor using XGBoost
        
        Args:
            lookback_period (int): Number of periods to look back for features
            min_samples (int): Minimum samples required for training
            n_estimators (int): Number of trees in XGBoost
            learning_rate (float): Learning rate for XGBoost
            max_depth (int): Maximum depth of trees
        """
        self.lookback_period = lookback_period
        self.min_samples = min_samples
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False
        )
        self.is_trained = False
        
    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical features for breakout prediction
        
        Args:
            data (pd.DataFrame): OHLCV data
            
        Returns:
            pd.DataFrame: Feature matrix
        """
        df = data.copy()
        
        # Price features
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['range'] = (df['High'] - df['Low']) / df['Close']
        df['range_ma'] = df['range'].rolling(window=20).mean()
        df['range_ratio'] = df['range'] / df['range_ma']
        
        # Volume features
        df['volume_ma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma']
        
        # Momentum features
        df['momentum'] = df['Close'].pct_change(5)
        df['momentum_ma'] = df['momentum'].rolling(window=20).mean()
        df['momentum_std'] = df['momentum'].rolling(window=20).std()
        
        # Trend features
        df['ema_short'] = df['Close'].ewm(span=8, adjust=False).mean()
        df['ema_med'] = df['Close'].ewm(span=13, adjust=False).mean()
        df['ema_long'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['trend_strength'] = (df['ema_short'] - df['ema_long']) / df['ema_long']
        
        # Clean up
        df = df.dropna()
        
        return df
        
    def _create_labels(self, data: pd.DataFrame, forward_periods: int = 5) -> np.ndarray:
        """
        Create labels for breakout prediction
        
        Args:
            data (pd.DataFrame): Feature matrix
            forward_periods (int): Number of periods to look forward
            
        Returns:
            np.ndarray: Binary labels (1 for breakout, 0 for no breakout)
        """
        # Calculate future returns
        future_returns = data['Close'].shift(-forward_periods) / data['Close'] - 1
        
        # Define breakout as significant price movement
        breakout_threshold = data['volatility'] * 2
        labels = (abs(future_returns) > breakout_threshold).astype(int)
        
        return labels
        
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the XGBoost model for breakout prediction
        
        Args:
            data (pd.DataFrame): OHLCV data
        """
        if len(data) < self.min_samples:
            logging.warning(f"Insufficient data for training. Need at least {self.min_samples} samples.")
            return
            
        # Calculate features and labels
        features = self._calculate_features(data)
        labels = self._create_labels(features)
        
        # Prepare training data
        X = features.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
        y = labels
        
        # Train model
        self.model.fit(X, y)
        self.is_trained = True
        logging.info("Breakout predictor trained successfully")
        
    def predict(self, data: pd.DataFrame) -> Tuple[bool, float]:
        """
        Predict breakout probability
        
        Args:
            data (pd.DataFrame): OHLCV data
            
        Returns:
            Tuple[bool, float]: (is_breakout, breakout_probability)
        """
        if not self.is_trained:
            logging.warning("Model not trained. Returning default prediction.")
            return False, 0.0
            
        # Calculate features
        features = self._calculate_features(data)
        
        # Prepare prediction data
        X = features.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
        
        # Make prediction
        proba = self.model.predict_proba(X)[:, 1]
        is_breakout = proba[-1] > 0.5
        
        return is_breakout, proba[-1]

    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance scores
        
        Returns:
            pd.Series: Feature importance scores
        """
        feature_names = [
            'returns', 'volatility', 'range', 'range_ma', 'range_ratio',
            'volume_ma', 'volume_ratio', 'momentum', 'momentum_ma', 'momentum_std',
            'ema_short', 'ema_med', 'ema_long', 'trend_strength'
        ]
        return pd.Series(
            self.model.feature_importances_,
            index=feature_names
        ).sort_values(ascending=False) 