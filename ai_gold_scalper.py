import ccxt
import pandas as pd
import numpy as np
import ta
import time
import logging
import os
from telegram import Bot
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# 1. Logging & Telegram
# -----------------------------
logging.basicConfig(filename='ai_gold_scalper.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
bot = Bot(token=TELEGRAM_TOKEN)

def send_alert(message):
    try:
        bot.send_message(chat_id=CHAT_ID, text=message)
    except Exception as e:
        logging.error(f"Telegram alert failed: {e}")

# -----------------------------
# 2. Exchange Setup
# -----------------------------
exchange = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_API_SECRET'),
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'},
})
symbol = 'XAU/USDT'
lot_size = float(os.getenv('LOT_SIZE', 0.1))
tp_amount = float(os.getenv('TP_AMOUNT', 10))
sl_amount = float(os.getenv('SL_AMOUNT', 5))

# -----------------------------
# 3. Fetch OHLCV & Indicators
# -----------------------------
def fetch_data(symbol, timeframe, limit=200):
    ohlc = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlc, columns=['timestamp','open','high','low','close','volume'])
    df['ema20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    df['ema200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['stoch'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14).stoch()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    return df

# -----------------------------
# 4. Trend Check (M15)
# -----------------------------
def trend_check(df_m15):
    last = df_m15.iloc[-1]
    if last['ema50'] > last['ema200']:
        return 'BULL'
    elif last['ema50'] < last['ema200']:
        return 'BEAR'
    return 'NEUTRAL'

# -----------------------------
# 5. Order Book / Level 2 Filter
# -----------------------------
def order_book_filter(symbol, signal, threshold=0.6):
    ob = exchange.fetch_order_book(symbol)
    bids = sum([price[1] for price in ob['bids'][:5]])
    asks = sum([price[1] for price in ob['asks'][:5]])
    if signal == 'BUY' and bids / (bids + asks) > threshold:
        return True
    elif signal == 'SELL' and asks / (bids + asks) > threshold:
        return True
    return False

# -----------------------------
# 6. AI Feature Preparation
# -----------------------------
def prepare_features(df, order_book=None):
    df['ema20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    df['ema200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['stoch'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14).stoch()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    if order_book:
        bids = sum([price[1] for price in order_book['bids'][:5]])
        asks = sum([price[1] for price in order_book['asks'][:5]])
        df['bid_ask_ratio'] = bids / (bids + asks)
    else:
        df['bid_ask_ratio'] = 0.5
    df['close_next'] = df['close'].shift(-1)
    df.dropna(inplace=True)
    df['label'] = np.where(df['close_next'] > df['close'], 'BUY', 'SELL')
    return df

# -----------------------------
# 7. Train AI Model
# -----------------------------
def train_ai_model(df_features):
    X = df_features[['ema20','ema50','ema200','bb_upper','bb_lower','rsi','stoch','macd','macd_signal','bid_ask_ratio']]
    y = LabelEncoder().fit_transform(df_features['label'])
    model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05)
    model.fit(X, y)
    return model

# -----------------------------
# 8. Signal Check
# -----------------------------
def check_signal(df_m1, trend_m15):
    last = df_m1.iloc[-1]
    if trend_m15 == 'BULL':
        if last['close'] <= last['bb_lower'] and last['rsi'] < 30 and last['stoch'] < 20 and last['macd'] > last['macd_signal']:
            return 'BUY'
    if trend_m15 == 'BEAR':
        if last['close'] >= last['bb_upper'] and last['rsi'] > 70 and last['stoch'] > 80 and last['macd'] < last['macd_signal']:
            return 'SELL'
    return 'HOLD'

# -----------------------------
# 9. Place OCO Order
# -----------------------------
def place_oco_order(signal):
    ticker = exchange.fetch_ticker(symbol)
    price = ticker['last']
    try:
        if signal == 'BUY':
            exchange.create_market_buy_order(symbol, lot_size)
            tp_price = price + tp_amount
            sl_price = price - sl_amount
            msg = f"BUY executed at {price} | TP {tp_price}, SL {sl_price}"
        elif signal == 'SELL':
            exchange.create_market_sell_order(symbol, lot_size)
            tp_price = price - tp_amount
            sl_price = price + sl_amount
            msg = f"SELL executed at {price} | TP {tp_price}, SL {sl_price}"
        send_alert(msg)
        logging.info(msg)
    except Exception as e:
        send_alert(f"OCO Error: {e}")
        logging.error(f"OCO Error: {e}")

# -----------------------------
# 10. Main Loop
# -----------------------------
while True:
    try:
        df_m1 = fetch_data(symbol, '1m')
        df_m15 = fetch_data(symbol, '15m')
        trend = trend_check(df_m15)
        order_book = exchange.fetch_order_book(symbol)
        features = prepare_features(df_m1.tail(1), order_book)
        model = train_ai_model(prepare_features(df_m1, order_book))
        X_live = features[['ema20','ema50','ema200','bb_upper','bb_lower','rsi','stoch','macd','macd_signal','bid_ask_ratio']]
        ai_signal_num = model.predict(X_live)[0]
        ai_signal = 'BUY' if ai_signal_num == 1 else 'SELL'
        classic_signal = check_signal(df_m1, trend)
        if classic_signal in ['BUY','SELL'] and classic_signal == ai_signal:
            if order_book_filter(symbol, classic_signal):
                place_oco_order(classic_signal)
        print(f"Trend: {trend}, Signal: {classic_signal}, AI: {ai_signal}")
        time.sleep(60)
    except Exception as e:
        logging.error(f"Bot Error: {e}")
        send_alert(f"Bot Error: {e}")
        time.sleep(60)
