# AI Trading System - UMHACKATHON 2025 🚀

## Project Overview
An advanced AI-powered trading system that utilizes Machine Learning models to generate trading signals for cryptocurrency pairs (BTC/USDT and ETH/USDT). The system implements sophisticated risk management strategies and provides real-time portfolio tracking through a web interface.

## Features 🌟
- **Advanced Trading Algorithms**
  - Market Regime Detection using HMM (Hidden Markov Models)
  - Breakout Prediction using Machine Learning
  - Dynamic Position Sizing based on Confidence Levels

- **Risk Management**
  - Stop Loss Management
  - Trailing Stop Implementation
  - Take Profit Targets
  - Maximum Drawdown Protection
  - Position Size Controls

- **Real-time Monitoring**
  - Web-based Dashboard
  - Portfolio Value Tracking
  - Trading Signal Visualization
  - Position Management Interface

## Technology Stack 💻
- **Backend**
  - Python
  - Flask (Web Framework)
  - NumPy & Pandas (Data Processing)
  - Scikit-learn (Machine Learning)

- **Frontend**
  - HTML/CSS
  - Chart.js (Data Visualization)
  - Tailwind CSS (Styling)

## System Architecture 🏗
The system consists of three main components:
1. **Trading Engine (`trading_system.py`)**
   - Core trading logic
   - Risk management implementation
   - Portfolio tracking

2. **Model Components**
   - Market regime detection
   - Breakout prediction
   - Signal generation

3. **Web Interface**
   - Real-time dashboard
   - Performance metrics
   - Trading signals display

## Installation & Setup 🔧
1. Clone the repository
```bash
git clone [your-repository-url]
cd ai-trader
```

2. Install required packages
```bash
pip install -r requirements.txt
```

3. Run the Flask application
```bash
python app.py
```

## Usage 📈
1. Access the web interface at `http://localhost:5000`
2. Monitor real-time trading signals and portfolio performance
3. View position updates and market analysis

## Trading Parameters
- Lookback Period: 3 days
- Regime States: 3
- Confidence Threshold: 0.15
- Position Size: 0.6
- Max Position Size: 0.7
- Stop Loss: 5%
- Trailing Stop: 4%
- Take Profit: 10%
- Max Drawdown: 10%

## Performance Metrics 📊
- Portfolio Value Tracking
- Position Management
- Signal Confidence Levels
- Return on Investment

## Future Enhancements 🔮
- Additional cryptocurrency pairs
- Enhanced ML models
- Advanced risk management features
- Mobile-responsive interface
- API integration with multiple exchanges

## Team Name 👥
- Codestar

## Acknowledgments 🙏
- UMHACKATHON 2025 Organizers
