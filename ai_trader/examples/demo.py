import sys
import os
from pathlib import Path

# Add the parent directory to Python path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

import pandas as pd
from trading_system import AITrader
import time
from datetime import datetime, timedelta

def main():
    # Initialize trading system with just BTC and ETH for testing
    symbols = ['BTCUSDT', 'ETHUSDT']  # Reduced number of symbols
    trader = AITrader(
        symbols=symbols,
        lookback_period=3,            # Back to 3 for better trend detection
        regime_states=3,              # Keep 3 states for clear regime detection
        confidence_threshold=0.15,    # Back to 0.15 threshold
        position_size=0.6,           # Back to 0.6 position size
        max_position_size=0.7,       # Back to 0.7 max position
        stop_loss_pct=0.05,          # Back to 0.05 stop loss
        trailing_stop_pct=0.04,      # Back to 0.04 trailing stop
        take_profit_pct=0.1,         # Back to 0.1 take profit
        max_drawdown_pct=0.1,        # Back to 0.1 drawdown limit
        cooldown_periods=0           # No cooldown
    )
    
    # Fetch historical data for training (reduced to 30 days)
    print("Fetching historical data for training...")
    training_data = {}
    for symbol in symbols:
        data = trader.fetch_data(symbol, period="30d")  # Reduced period
        if data is not None:
            training_data[symbol] = data
            
    # Train models
    print("Training models...")
    trader.train_models(training_data)
    
    # Simulation parameters
    initial_portfolio_value = trader.get_portfolio_value({
        symbol: training_data[symbol].iloc[-1]['Close'].item()
        for symbol in symbols
        if symbol in training_data
    })
    print(f"Initial portfolio value: ${initial_portfolio_value:.2f}")
    
    # Run trading simulation
    print("\nStarting trading simulation...")
    simulation_days = 30 # Extended simulation days
    
    # Create rolling windows for simulation
    window_size = 48  # 2 days of hourly data
    for day in range(simulation_days):
        # Get current data using rolling window
        start_idx = -(window_size + simulation_days - day)
        end_idx = -(simulation_days - day)
        
        current_data = {
            symbol: training_data[symbol].iloc[start_idx:end_idx]
            for symbol in symbols
            if symbol in training_data
        }
        
        current_prices = {
            symbol: data.iloc[-1]['Close'].item()
            for symbol, data in current_data.items()
        }
        
        # Generate trading signals
        signals = trader.generate_signals(current_data)
        
        # Print signals for debugging
        print(f"\nDay {day + 1} Signals:")
        for symbol, signal in signals.items():
            print(f"  {symbol}: {signal['signal']} (Confidence: {signal['confidence']:.2f})")
        
        # Execute trades
        trader.execute_trades(signals, current_prices)
        
        # Print daily summary
        portfolio_value = trader.get_portfolio_value(current_prices)
        print(f"\nDay {day + 1}")
        print(f"Portfolio Value: ${portfolio_value:.2f}")
        print("Current Positions:")
        for symbol, shares in trader.portfolio['positions'].items():
            if shares > 0:
                value = shares * current_prices[symbol]
                print(f"  {symbol}: {shares} units (${value:.2f})")
        print(f"Cash: ${trader.portfolio['cash']:.2f}")
        
    # Print final results
    final_portfolio_value = trader.get_portfolio_value(current_prices)
    total_return = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value * 100
    
    print("\nSimulation Complete")
    print("===================")
    print(f"Initial Portfolio Value: ${initial_portfolio_value:.2f}")
    print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    
if __name__ == "__main__":
    main() 