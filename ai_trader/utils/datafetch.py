import requests
import time
import pandas as pd
import logging

API_KEY = "Paoymy9jLXrLpzt9cHfhmi0bWLDfktptrAcpROoZSatde3J2"
BASE_URL = "https://api.datasource.cybotrade.rs"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_cryptoquant(start_time: int, end_time: int, search: str, exchange: str):
    logger.info(f"Fetching CryptoQuant data: {search} from {exchange}")
    endpoint = f"btc/exchange-flows/{search}"

    url = f"{BASE_URL}/cryptoquant/{endpoint}"
    logger.info(f"Making request to: {url}")
    
    response = requests.get(url,
                            headers={
                                "X-API-Key":API_KEY
                            }, 
                            params={
                                "start_time": start_time,
                                "end_time": end_time,
                                "exchange": exchange,
                                "window": "hour"
                            }
                            )

    if response.status_code == 200:
        data = response.json()["data"]
        logger.info(f"Successfully fetched {len(data)} records")
        return data
    elif response.status_code == 429:
        reset_ts = int(response.headers.get("X-Api-Limit-Reset-Timestamp", time.time() * 1000))
        sleep_time = (reset_ts - int(time.time() * 1000)) / 1000
        logger.warning(f"⏳ Rate limit hit. Sleeping for {sleep_time:.2f} seconds.")
        time.sleep(sleep_time)
        return fetch_cryptoquant(start_time, end_time, search, exchange)
    else:
        logger.error(f"❌ Error {response.status_code}: {response.text}")
        return None

def fetch_bybit_candle(symbol: str = "BTCUSDT", start_time: int = None, end_time: int = None, interval: str = "6h"):
    """
    Fetch candlestick data from Bybit
    
    Args:
        symbol (str): Trading pair symbol
        start_time (int): Start time in milliseconds
        end_time (int): End time in milliseconds
        interval (str): Candle interval
    """
    logger.info(f"Fetching Bybit data: {symbol} from {interval} candles")
    
    if start_time is None:
        start_time = int(time.time() * 1000) - (30 * 24 * 60 * 60 * 1000)  # 30 days ago
    if end_time is None:
        end_time = int(time.time() * 1000)

    url = f"{BASE_URL}/bybit-spot/candle"
    logger.info(f"Making request to: {url}")
    
    response = requests.get(url,
                            headers={
                                "X-API-Key":API_KEY
                            }, 
                            params={
                                "symbol": symbol,
                                "start_time": start_time,
                                "end_time": end_time,
                                "interval": interval
                            }
                            )

    if response.status_code == 200:
        data = response.json()["data"]
        logger.info(f"Successfully fetched {len(data)} candles")
        # Log sample data structure
        if data:
            sample = data[0]
            logger.info(f"Sample data structure: {list(sample.keys())}")
        return data
    elif response.status_code == 429:
        reset_ts = int(response.headers.get("X-Api-Limit-Reset-Timestamp", time.time() * 1000))
        sleep_time = (reset_ts - int(time.time() * 1000)) / 1000
        logger.warning(f"⏳ Rate limit hit. Sleeping for {sleep_time:.2f} seconds.")
        time.sleep(sleep_time)
        return fetch_bybit_candle(symbol, start_time, end_time, interval)
    else:
        logger.error(f"❌ Error {response.status_code}: {response.text}")
        return None

def fetch_coinbase_candle(symbol: str = "BTC-USDT", start_time: int = None, end_time: int = None, interval: str = "6h"):
    """
    Fetch candlestick data from Coinbase
    
    Args:
        symbol (str): Trading pair symbol
        start_time (int): Start time in milliseconds
        end_time (int): End time in milliseconds
        interval (str): Candle interval
    """
    logger.info(f"Fetching Coinbase data: {symbol} from {interval} candles")
    
    if start_time is None:
        start_time = int(time.time() * 1000) - (30 * 24 * 60 * 60 * 1000)  # 30 days ago
    if end_time is None:
        end_time = int(time.time() * 1000)

    url = f"{BASE_URL}/coinbase/candle"
    logger.info(f"Making request to: {url}")
    
    response = requests.get(url,
                            headers={
                                "X-API-Key":API_KEY
                            }, 
                            params={
                                "symbol": symbol,
                                "start_time": start_time,
                                "end_time": end_time,
                                "interval": interval
                            }
                            )

    if response.status_code == 200:
        data = response.json()["data"]
        logger.info(f"Successfully fetched {len(data)} candles")
        return data
    elif response.status_code == 429:
        reset_ts = int(response.headers.get("X-Api-Limit-Reset-Timestamp", time.time() * 1000))
        sleep_time = (reset_ts - int(time.time() * 1000)) / 1000
        logger.warning(f"⏳ Rate limit hit. Sleeping for {sleep_time:.2f} seconds.")
        time.sleep(sleep_time)
        return fetch_coinbase_candle(symbol, start_time, end_time, interval)
    else:
        logger.error(f"❌ Error {response.status_code}: {response.text}")
        return None

if __name__ == "__main__":
    # Example usage
    start_time = 1704067200000  # 2024-01-01 00:00:00
    end_time = 1735603200000    # 2024-12-31 23:59:59
    
    # Test CryptoQuant data
    search = 'inflow'
    exchange = 'all_exchange'
    data_cryptoquant = fetch_cryptoquant(start_time, end_time, search, exchange)
    if data_cryptoquant:
        data_cryptoquant = pd.DataFrame(data_cryptoquant)
        print("\nCryptoQuant Data:")
        print("Unique rows:", len(data_cryptoquant))
        print("Columns:", data_cryptoquant.columns.tolist())
        print(data_cryptoquant.head())
    
    # Test Bybit data
    data_bybit = fetch_bybit_candle("BTCUSDT", start_time, end_time)
    if data_bybit:
        data_bybit = pd.DataFrame(data_bybit)
        print("\nBybit Data:")
        print("Unique rows:", len(data_bybit))
        print("Columns:", data_bybit.columns.tolist())
        print(data_bybit.head())
    
    # Test Coinbase data
    data_coinbase = fetch_coinbase_candle("BTC-USDT", start_time, end_time)
    if data_coinbase:
        data_coinbase = pd.DataFrame(data_coinbase)
        print("\nCoinbase Data:")
        print("Unique rows:", len(data_coinbase))
        print("Columns:", data_coinbase.columns.tolist())
        print(data_coinbase.head())