import numpy as np
from hmmlearn import hmm
from typing import Tuple, List
import pandas as pd
from sklearn.preprocessing import StandardScaler

class MarketRegimeHMM:
    def __init__(self, n_states: int = 3):
        """
        Initialize Market Regime Detection using HMM
        
        Args:
            n_states (int): Number of market regimes to detect (default: 3 for bullish, bearish, and sideways)
        """
        self.n_states = n_states
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for HMM model
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            np.ndarray: Processed features
        """
        # Calculate returns and volatility
        returns = np.log(data['Close'] / data['Close'].shift(1))
        volatility = returns.rolling(window=20).std()
        
        # Calculate trading volume changes
        volume_change = np.log(data['Volume'] / data['Volume'].shift(1))
        
        # Calculate price momentum
        momentum = (data['Close'] - data['Close'].shift(5)) / data['Close'].shift(5)
        
        # Forward fill NaN values first
        returns = returns.ffill()
        volatility = volatility.ffill()
        volume_change = volume_change.ffill()
        momentum = momentum.ffill()
        
        # Backward fill any remaining NaN values at the start
        returns = returns.bfill()
        volatility = volatility.bfill()
        volume_change = volume_change.bfill()
        momentum = momentum.bfill()
        
        # If there are still any NaN values, replace with zeros
        returns = returns.fillna(0)
        volatility = volatility.fillna(0)
        volume_change = volume_change.fillna(0)
        momentum = momentum.fillna(0)
        
        features = np.column_stack([
            returns,
            volatility,
            volume_change,
            momentum
        ])
        
        return self.scaler.fit_transform(features)
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the HMM model to the market data
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
        """
        features = self._prepare_features(data)
        self.model.fit(features)
        
    def predict_regime(self, data: pd.DataFrame) -> Tuple[List[int], np.ndarray]:
        """
        Predict market regimes and their probabilities
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            Tuple[List[int], np.ndarray]: Predicted regimes and their probabilities
        """
        features = self._prepare_features(data)
        regimes = self.model.predict(features)
        probs = self.model.predict_proba(features)
        
        return regimes, probs
    
    def get_regime_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the learned parameters of each regime
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Means and covariances of each regime
        """
        return self.model.means_, self.model.covars_
    
    def get_transition_matrix(self) -> np.ndarray:
        """
        Get the regime transition probability matrix
        
        Returns:
            np.ndarray: Transition probability matrix
        """
        return self.model.transmat_ 