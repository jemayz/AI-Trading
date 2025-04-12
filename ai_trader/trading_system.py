import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from datetime import datetime

# Local imports
from models.market_regime import MarketRegimeHMM
from models.breakout_predictor import BreakoutPredictor
from utils.datafetch import fetch_bybit_candle, fetch_coinbase_candle, fetch_cryptoquant

class AITrader:
    def __init__(
        self,
        symbols: List[str],
        lookback_period: int = 10,
        regime_states: int = 4,
        confidence_threshold: float = 0.5,     # Much lower threshold
        position_size: float = 0.2,           # Conservative position size
        max_position_size: float = 0.2,       # No scaling
        stop_loss_pct: float = 0.02,          # Tight stop loss
        trailing_stop_pct: float = 0.01,      # Tight trailing stop
        take_profit_pct: float = 0.03,        # Realistic take profit
        max_drawdown_pct: float = 0.03,       # Tight drawdown limit
        cooldown_periods: int = 2             # Moderate cooldown
    ):
        """
        Initialize AI Trading System
        
        Args:
            symbols (List[str]): List of trading symbols
            lookback_period (int): Period for technical indicators
            regime_states (int): Number of market regime states
            confidence_threshold (float): Minimum confidence for trade execution
            position_size (float): Initial position size as fraction of portfolio
            max_position_size (float): Maximum position size per symbol
            stop_loss_pct (float): Stop loss percentage
            trailing_stop_pct (float): Trailing stop percentage
            take_profit_pct (float): Take profit percentage
            max_drawdown_pct (float): Maximum drawdown allowed
            cooldown_periods (int): Number of periods to wait after a stop loss
        """
        self.symbols = symbols
        self.lookback_period = lookback_period
        self.confidence_threshold = confidence_threshold
        self.position_size = position_size
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.cooldown_periods = cooldown_periods
        
        # Initialize models
        self.regime_models = {
            symbol: MarketRegimeHMM(n_states=regime_states)
            for symbol in symbols
        }
        
        # Initialize breakout predictors
        self.breakout_predictors = {
            symbol: BreakoutPredictor(
                lookback_period=lookback_period,
                min_samples=100,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6
            )
            for symbol in symbols
        }
        
        # Initialize portfolio and positions
        self.portfolio = {
            'cash': 100000.0,  # Initial capital
            'positions': {symbol: 0 for symbol in symbols},
            'position_values': {symbol: 0.0 for symbol in symbols},
            'entry_prices': {symbol: 0.0 for symbol in symbols},
            'highest_prices': {symbol: 0.0 for symbol in symbols},
            'stop_losses': {symbol: 0.0 for symbol in symbols},
            'take_profits': {symbol: 0.0 for symbol in symbols},
            'cooldown_counter': {symbol: 0 for symbol in symbols}  # Track cooldown periods
        }
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging system"""
        logging.basicConfig(
            filename=f'trading_log_{datetime.now().strftime("%Y%m%d")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def fetch_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """
        Fetch historical market data from crypto exchanges
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT' or 'BTC-USDT')
            period (str): Time period to fetch ('1y', '2y', or '30d')
            
        Returns:
            pd.DataFrame: Historical OHLCV data
        """
        try:
            # Standardize symbol format
            unified_symbol = symbol.replace('-', '')  # Convert 'BTC-USDT' to 'BTCUSDT'
            cb_symbol = f"{unified_symbol[:-4]}-{unified_symbol[-4:]}"  # Convert to Coinbase format
            
            # Get current timestamp in milliseconds
            end_time = int(datetime.now().timestamp() * 1000)
            
            # Calculate start time and interval based on period
            intervals = {
                "1y": {"delta": 365 * 24 * 60 * 60 * 1000, "interval": "4h"},
                "2y": {"delta": 2 * 365 * 24 * 60 * 60 * 1000, "interval": "6h"},
                "30d": {"delta": 30 * 24 * 60 * 60 * 1000, "interval": "1h"}
            }
            
            period_config = intervals.get(period, intervals["30d"])
            start_time = end_time - period_config["delta"]
            interval = period_config["interval"]
            
            logging.info(f"Fetching data for {symbol} from {datetime.fromtimestamp(start_time/1000)} to {datetime.fromtimestamp(end_time/1000)}")
            
            # Try Bybit first
            data = fetch_bybit_candle(symbol=unified_symbol, start_time=start_time, end_time=end_time, interval=interval)
            
            if data is not None and len(data) > 0:
                df = self._process_ohlcv_data(data)
                if period == "30d":
                    df = self._add_flow_data(df, start_time, end_time)
                return df
            
            # If Bybit fails, try Coinbase
            data = fetch_coinbase_candle(symbol=cb_symbol, start_time=start_time, end_time=end_time, interval=interval)
            
            if data is not None and len(data) > 0:
                df = self._process_ohlcv_data(data)
                if period == "30d":
                    df = self._add_flow_data(df, start_time, end_time)
                return df
            
            logging.error(f"Failed to fetch data for {symbol} from both exchanges")
            return None
            
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
            
    def _process_ohlcv_data(self, data: List[Dict]) -> pd.DataFrame:
        """Helper method to process OHLCV data consistently"""
        df = pd.DataFrame(data)
        
        # Ensure timestamp column exists and convert to datetime
        if 'start_time' not in df.columns and 'timestamp' in df.columns:
            df['start_time'] = df['timestamp']
        
        # Convert timestamp to datetime
        df['start_time'] = pd.to_datetime(df['start_time'], unit='ms')
        df.set_index('start_time', inplace=True)
        
        # Standardize column names
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        return df
        
    def _add_flow_data(self, df: pd.DataFrame, start_time: int, end_time: int) -> pd.DataFrame:
        """Helper method to add flow data consistently"""
        try:
            flow_data = fetch_cryptoquant(start_time, end_time, 'inflow', 'all_exchange')
            if flow_data:
                flow_df = pd.DataFrame(flow_data)
                logging.info(f"Flow data columns: {flow_df.columns.tolist()}")
                
                # Handle both possible timestamp column names
                time_col = 'start_time' if 'start_time' in flow_df.columns else 'time'
                if time_col in flow_df.columns:
                    flow_df[time_col] = pd.to_datetime(flow_df[time_col], unit='ms')
                    flow_df.set_index(time_col, inplace=True)
                    
                    # Resample flow data to match OHLCV data frequency
                    flow_df = flow_df.resample(df.index.freq).mean()
                    
                    # Merge flow data with price data
                    df = df.join(flow_df, how='left')
                    
                    # Forward fill missing values for up to 24 hours
                    df = df.fillna(method='ffill', limit=24)
                
        except Exception as e:
            logging.warning(f"Error adding flow data: {str(e)}")
            
        return df
        
    def train_models(self, training_data: Dict[str, pd.DataFrame]) -> None:
        """
        Train HMM and breakout models
        
        Args:
            training_data (Dict[str, pd.DataFrame]): Historical data for each symbol
        """
        for symbol in self.symbols:
            if symbol in training_data:
                logging.info(f"Training models for {symbol}")
                
                # Train HMM model
                self.regime_models[symbol].fit(training_data[symbol])
                
                # Train breakout predictor
                self.breakout_predictors[symbol].fit(training_data[symbol])

    def _calculate_trend_strength(self, data: pd.DataFrame) -> Tuple[float, bool]:
        """Calculate trend strength using multiple indicators"""
        # Calculate EMAs
        ema_short = data['Close'].ewm(span=8, adjust=False).mean()
        ema_med = data['Close'].ewm(span=13, adjust=False).mean()
        ema_long = data['Close'].ewm(span=21, adjust=False).mean()
        
        # Calculate trend strength as percentage difference between EMAs
        trend_strength = ((ema_short - ema_long) / ema_long * 100).iloc[-1]
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - signal
        
        # Calculate volume trend
        volume_sma = data['Volume'].rolling(window=20).mean()
        volume_trend = ((data['Volume'] - volume_sma) / volume_sma * 100).iloc[-1]
        
        # Define minimum requirements
        min_trend_strength = 0.15  # Lower threshold
        min_rsi = 45  # More lenient RSI
        min_volume_trend = 0
        
        # Check if trend is strong enough
        is_strong_trend = (
            abs(trend_strength) > min_trend_strength and 
            rsi.iloc[-1] > min_rsi and 
            volume_trend > min_volume_trend and
            macd_hist.iloc[-1] > 0  # Positive MACD histogram
        )
        
        return trend_strength, is_strong_trend

    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        """Add RSI for momentum confirmation"""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Consider both price momentum and RSI
        momentum = data['Close'].pct_change(5).mean() * 100
        rsi_signal = 1 if rsi.iloc[-1] > 50 else -1
        
        return momentum * rsi_signal

    def _detect_breakout(self, data: pd.DataFrame, symbol: str) -> Tuple[bool, float]:
        """
        Detect price breakouts using both statistical and ML methods
        
        Args:
            data (pd.DataFrame): Market data
            symbol (str): Trading symbol
            
        Returns:
            Tuple[bool, float]: (is_breakout, breakout_strength)
        """
        # Statistical breakout detection
        high_low_range = (data['High'] - data['Low']).rolling(window=20).mean()
        returns = data['Close'].pct_change()
        volatility = returns.rolling(window=20).std()
        momentum = data['Close'].pct_change(5)
        
        current_range = data['High'].iloc[-1] - data['Low'].iloc[-1]
        range_ratio = current_range / high_low_range.iloc[-1]
        
        # Statistical breakout strength
        stat_breakout_strength = (
            0.4 * min(1.0, range_ratio / 2) +  # Price range expansion
            0.3 * min(1.0, abs(momentum.iloc[-1]) / 0.02) +  # Momentum
            0.3 * min(1.0, volatility.iloc[-1] / 0.01)  # Volatility
        )
        
        # Statistical breakout conditions
        stat_is_breakout = (
            range_ratio > 1.5 and  # Significant range expansion
            abs(momentum.iloc[-1]) > volatility.iloc[-1] * 2 and  # Strong momentum
            volatility.iloc[-1] > volatility.rolling(window=20).mean().iloc[-1]  # Increasing volatility
        )
        
        # ML breakout detection
        ml_is_breakout, ml_breakout_strength = self.breakout_predictors[symbol].predict(data)
        
        # Combine both approaches
        is_breakout = stat_is_breakout or ml_is_breakout
        breakout_strength = max(stat_breakout_strength, ml_breakout_strength)
        
        return is_breakout, breakout_strength

    def generate_signals(self, symbol_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Generate trading signals using HMM model and breakout detection
        
        Args:
            symbol_data (Dict[str, pd.DataFrame]): Current market data for each symbol
            
        Returns:
            Dict[str, Dict]: Trading signals for each symbol
        """
        signals = {}
        
        for symbol, data in symbol_data.items():
            if symbol not in self.regime_models:
                continue
                
            # Get current price and volume data
            current_price = data['Close'].iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            
            # Calculate volume moving averages
            volume_ma5 = data['Volume'].rolling(window=5).mean().iloc[-1]
            volume_ma20 = data['Volume'].rolling(window=20).mean().iloc[-1]
            
            # Get market regime prediction
            regime, regime_probs = self.regime_models[symbol].predict_regime(data)
            current_regime = regime[-1]
            regime_confidence = np.max(regime_probs[-1])
            
            # Calculate trend strength
            trend_strength, is_strong_trend = self._calculate_trend_strength(data)
            
            # Calculate momentum
            momentum = self._calculate_momentum(data)
            
            # Detect breakout using combined approach
            is_breakout, breakout_strength = self._detect_breakout(data, symbol)
            
            # Volume confirmation
            volume_ratio = current_volume / volume_ma20
            volume_trend = current_volume > volume_ma5
            
            # Calculate RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # Calculate MACD
            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_hist = macd - signal
            
            # Calculate combined confidence with breakout consideration
            confidence = (
                0.2 * regime_confidence +  # Regime confidence
                0.25 * min(1.0, abs(momentum)) +  # Momentum
                0.2 * min(1.0, abs(trend_strength) / 3) +  # Trend strength
                0.15 * breakout_strength +  # Breakout strength
                0.1 * min(1.0, volume_ratio / 1.5) +  # Volume confirmation
                0.1 * min(1.0, abs(macd_hist.iloc[-1]) / 0.02)  # MACD confirmation
            )
            
            # Initialize signal
            signal = "HOLD"
            
            # Current position for this symbol
            current_position = self.portfolio['positions'][symbol]
            
            # Generate signals based on regime, technical indicators, and breakout
            if current_position == 0:  # No position, look for buy
                if ((current_regime == 0 or  # Bullish regime
                     (current_regime == 1 and trend_strength > 0)) and  # Or transitioning to bullish
                    (momentum > -0.001 or is_breakout) and  # Positive momentum or breakout
                    current_rsi < 75 and  # Not overbought
                    (macd_hist.iloc[-1] > -0.001 or trend_strength > 0.1) and  # Positive MACD or strong trend
                    (volume_trend or volume_ratio > 1.1) and  # Good volume
                    confidence > self.confidence_threshold):
                    signal = "BUY"
                    
            elif current_position > 0:  # Have position, look for sell
                if (current_regime == 2 or  # Bearish regime
                    momentum < -0.002 or  # Negative momentum
                    current_rsi > 75 or  # Overbought
                    macd_hist.iloc[-1] < -0.002 or  # Negative MACD
                    confidence < self.confidence_threshold * 0.7):  # Lower threshold for exit
                    signal = "SELL"
                    
            signals[symbol] = {
                "signal": signal,
                "confidence": confidence,
                "regime": current_regime,
                "trend_strength": trend_strength,
                "momentum": momentum,
                "volume_ratio": volume_ratio,
                "rsi": current_rsi,
                "macd": macd_hist.iloc[-1],
                "is_breakout": is_breakout,
                "breakout_strength": breakout_strength
            }
            
        return signals
    
    def _check_risk_levels(self, symbol: str, current_price: float) -> str:
        """
        Check risk levels and return appropriate action
        
        Args:
            symbol (str): Trading symbol
            current_price (float): Current price
            
        Returns:
            str: Action to take ('EXIT', 'HOLD', or None)
        """
        if self.portfolio['positions'][symbol] <= 0:
            return None
            
        entry_price = self.portfolio['entry_prices'][symbol]
        highest_price = max(self.portfolio['highest_prices'][symbol], current_price)
        self.portfolio['highest_prices'][symbol] = highest_price
        
        # Update trailing stop if price has moved in our favor
        if current_price > entry_price:
            trailing_stop = highest_price * (1 - self.trailing_stop_pct)
            self.portfolio['stop_losses'][symbol] = max(
                trailing_stop,
                self.portfolio['stop_losses'][symbol]
            )
        
        # Check stop loss
        if current_price <= self.portfolio['stop_losses'][symbol]:
            logging.info(
                f"Stop loss triggered for {symbol} at {current_price}"
                f" (Entry: {entry_price}, Stop: {self.portfolio['stop_losses'][symbol]})"
            )
            return 'EXIT'
            
        # Check take profit
        if current_price >= self.portfolio['take_profits'][symbol]:
            logging.info(
                f"Take profit triggered for {symbol} at {current_price}"
                f" (Entry: {entry_price}, Target: {self.portfolio['take_profits'][symbol]})"
            )
            return 'EXIT'
            
        # Check maximum drawdown
        current_drawdown = (highest_price - current_price) / highest_price
        if current_drawdown >= self.max_drawdown_pct:
            logging.info(
                f"Maximum drawdown reached for {symbol} at {current_price}"
                f" (Highest: {highest_price}, Drawdown: {current_drawdown:.2%})"
            )
            return 'EXIT'
            
        return 'HOLD'

    def execute_trades(
        self,
        signals: Dict[str, Dict],
        current_prices: Dict[str, float]
    ) -> None:
        """Execute trades with improved risk management"""
        # Update position values first
        total_portfolio_value = self.get_portfolio_value(current_prices)
        
        # Update cooldown counters
        for symbol in self.symbols:
            if self.portfolio['cooldown_counter'][symbol] > 0:
                self.portfolio['cooldown_counter'][symbol] -= 1
        
        # Sort symbols by confidence
        sorted_symbols = sorted(
            signals.keys(),
            key=lambda x: signals[x]['confidence'] if x in signals else 0,
            reverse=True
        )
        
        for symbol in sorted_symbols:
            if symbol not in current_prices:
                continue
                
            current_price = current_prices[symbol]
            
            # Skip if in cooldown
            if self.portfolio['cooldown_counter'][symbol] > 0:
                continue
                
            # Check risk levels first
            risk_action = self._check_risk_levels(symbol, current_price)
            if risk_action == 'EXIT':
                self._exit_position(symbol, current_price)
                # Start cooldown period
                self.portfolio['cooldown_counter'][symbol] = self.cooldown_periods
                continue
                
            # Update position values
            self.portfolio['position_values'][symbol] = (
                self.portfolio['positions'][symbol] * current_price
            )
            
            # Get trading signal
            if symbol not in signals:
                continue
                
            signal = signals[symbol]
            action = signal['signal']
            confidence = signal['confidence']
            
            # Skip low confidence signals
            if confidence < self.confidence_threshold:
                continue
                
            # Calculate position metrics
            position_value = self.portfolio['position_values'][symbol]
            position_ratio = position_value / total_portfolio_value if total_portfolio_value > 0 else 0
            
            if action == 'BUY' and position_ratio < self.max_position_size:
                # Calculate position size based on confidence
                base_size = self.position_size * min(1.0, confidence / 2)  # Scale down high confidence values
                remaining_capacity = self.max_position_size - position_ratio
                adjusted_size = min(remaining_capacity, base_size)
                
                # Calculate shares to buy
                position_value = total_portfolio_value * adjusted_size
                shares_to_buy = position_value / current_price
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    if cost <= self.portfolio['cash']:
                        # Set initial risk levels
                        if self.portfolio['positions'][symbol] == 0:
                            self.portfolio['entry_prices'][symbol] = current_price
                            self.portfolio['highest_prices'][symbol] = current_price
                            self.portfolio['stop_losses'][symbol] = current_price * (1 - self.stop_loss_pct)
                            self.portfolio['take_profits'][symbol] = current_price * (1 + self.take_profit_pct)
                        
                        # Execute trade
                        self.portfolio['cash'] -= cost
                        self.portfolio['positions'][symbol] += shares_to_buy
                        self.portfolio['position_values'][symbol] += cost
                        logging.info(
                            f"BUY {shares_to_buy:.6f} shares of {symbol} at {current_price}"
                        )
                        
            elif action == 'SELL' and self.portfolio['positions'][symbol] > 0:
                self._exit_position(symbol, current_price)

    def _exit_position(self, symbol: str, current_price: float) -> None:
        """
        Exit a position completely
        
        Args:
            symbol (str): Trading symbol
            current_price (float): Current price
        """
        shares = self.portfolio['positions'][symbol]
        if shares > 0:
            revenue = shares * current_price
            self.portfolio['cash'] += revenue
            self.portfolio['positions'][symbol] = 0
            self.portfolio['position_values'][symbol] = 0
            self.portfolio['entry_prices'][symbol] = 0
            self.portfolio['highest_prices'][symbol] = 0
            self.portfolio['stop_losses'][symbol] = 0
            self.portfolio['take_profits'][symbol] = 0
            logging.info(
                f"EXIT {shares} shares of {symbol} at {current_price}"
            )

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate current portfolio value
        
        Args:
            current_prices (Dict[str, float]): Current market prices
            
        Returns:
            float: Total portfolio value
        """
        portfolio_value = self.portfolio['cash']
        
        for symbol, shares in self.portfolio['positions'].items():
            if symbol in current_prices:
                portfolio_value += shares * current_prices[symbol]
                
        return float(portfolio_value)
        
    def get_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics
        
        Returns:
            Dict: Performance metrics
        """
        # Implement performance metrics calculation
        # (e.g., Sharpe ratio, max drawdown, etc.)
        pass 