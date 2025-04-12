from flask import Flask, render_template
import sys
import os
from pathlib import Path

# Add the parent directory to Python path
parent_dir = str(Path(__file__).resolve().parent)
sys.path.append(parent_dir)

# Import the AITrader class
from trading_system import AITrader

app = Flask(__name__)

def run_simulation():
    # Initialize trading system with parameters from demo.py
    symbols = ['BTCUSDT', 'ETHUSDT']
    trader = AITrader(
        symbols=symbols,
        lookback_period=3,
        regime_states=3,
        confidence_threshold=0.15,
        position_size=0.6,
        max_position_size=0.7,
        stop_loss_pct=0.05,
        trailing_stop_pct=0.04,
        take_profit_pct=0.1,
        max_drawdown_pct=0.1,
        cooldown_periods=0
    )
    
    # Fetch historical data for training
    training_data = {}
    for symbol in symbols:
        data = trader.fetch_data(symbol, period="30d")
        if data is not None:
            training_data[symbol] = data
            
    # Train models
    trader.train_models(training_data)
    
    # Simulation parameters
    initial_portfolio_value = trader.get_portfolio_value({
        symbol: training_data[symbol].iloc[-1]['Close'].item()
        for symbol in symbols
        if symbol in training_data
    })
    
    # Run trading simulation
    simulation_days = 30
    window_size = 48
    logs = []
    
    for day in range(simulation_days):
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
        
        signals = trader.generate_signals(current_data)
        trader.execute_trades(signals, current_prices)
        
        portfolio_value = trader.get_portfolio_value(current_prices)
        logs.append({
            'day': day + 1,
            'portfolio_value': portfolio_value,
            'cash': trader.portfolio['cash'],
            'positions': trader.portfolio['positions'].copy(),
            'signals': signals
        })
    
    final_portfolio_value = trader.get_portfolio_value(current_prices)
    total_return = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value * 100
    
    return {
        'logs': logs,
        'initial_portfolio_value': initial_portfolio_value,
        'final_portfolio_value': final_portfolio_value,
        'total_return': total_return
    }

@app.route('/')
def index():
    results = run_simulation()
    
    # Ensure logs are structured
    structured_logs = []
    for log in results['logs']:
        structured_logs.append({
            'day': log.get('day'),
            'portfolio_value': log.get('portfolio_value'),
            'cash': log.get('cash'),
            'positions': log.get('positions'),
            'signals': log.get('signals')
        })

    results['logs'] = structured_logs
    results['initial_value'] = round(results['initial_portfolio_value'], 2)
    results['final_value'] = round(results['final_portfolio_value'], 2)
    results['total_return'] = round(results['total_return'], 2)

    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)