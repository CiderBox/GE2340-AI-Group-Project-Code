import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pywt  # 添加小波变换库

def fetch_stock_data(start_date='2022-01-01', end_date='2024-01-01', ticker='^IXIC'):
    """
    从雅虎财经获取NASDAQ指数数据，获取2022-2024的数据
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def add_noise(data, noise_factor=0.05):
    """添加随机噪声进行数据增强"""
    noise = np.random.normal(0, noise_factor, data.shape)
    return data + noise

def denoise_data(data, wavelet='db4', level=1):
    """
    使用小波变换进行数据去噪
    """
    # 小波分解
    coeffs = pywt.wavedec(data, wavelet, level=level)
    
    # 阈值去噪
    threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(data)))
    coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    
    # 重构信号
    return pywt.waverec(coeffs_thresholded, wavelet)

def prepare_data(data, sequence_length=60):
    """
    准备训练、验证和测试数据，添加去噪处理
    """
    # 对收盘价进行去噪
    data['Close'] = denoise_data(data['Close'].values)
    
    # 计算技术指标
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['RSI'] = calculate_rsi(data['Close'], periods=14)
    data['MACD'] = calculate_macd(data['Close'])
    data['EMA12'] = data['Close'].ewm(span=12).mean()
    data['EMA26'] = data['Close'].ewm(span=26).mean()
    
    # 计算布林带
    data['Upper'], data['Middle'], data['Lower'] = calculate_bollinger_bands(data['Close'])
    
    # 添加价格变化率
    data['Price_Change'] = data['Close'].pct_change()
    data['Volume_Change'] = data['Volume'].pct_change()
    
    # 添加波动率
    data['Volatility'] = data['Close'].rolling(window=20).std()
    
    # 添加动量指标
    data['Momentum'] = data['Close'] - data['Close'].shift(4)
    
    # 添加滞后特征
    for i in [1, 2, 3, 5]:
        data[f'Close_lag_{i}'] = data['Close'].shift(i)
        data[f'Volume_lag_{i}'] = data['Volume'].shift(i)
    
    # 选择特征
    features = [
        'Close', 'Volume', 'MA5', 'MA20', 'RSI', 'MACD',
        'EMA12', 'EMA26', 'Upper', 'Middle', 'Lower',
        'Price_Change', 'Volume_Change', 'Volatility', 'Momentum'
    ] + [f'Close_lag_{i}' for i in [1, 2, 3, 5]] + [f'Volume_lag_{i}' for i in [1, 2, 3, 5]]
    
    # 删除含有NaN的行
    df = data[features].dropna().values
    
    # 使用StandardScaler
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    # 创建序列数据
    X, y = [], []
    for i in range(len(df_scaled) - sequence_length):
        X.append(df_scaled[i:(i + sequence_length)])
        y.append(df_scaled[i + sequence_length, 0])  # 只预测收盘价
    
    X, y = np.array(X), np.array(y)
    
    # 分割数据
    total_days = len(X)
    train_size = int(total_days * 0.5)  # 大约一年的数据
    val_size = int(train_size * 0.2)    # 验证集使用训练集的20%
    
    X_train = X[:train_size-val_size]
    y_train = y[:train_size-val_size]
    
    X_val = X[train_size-val_size:train_size]
    y_val = y[train_size-val_size:train_size]
    
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    # 数据增强
    X_train_augmented = add_noise(X_train)
    X_train = np.concatenate([X_train, X_train_augmented], axis=0)
    y_train = np.concatenate([y_train, y_train], axis=0)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

def calculate_rsi(prices, periods=14):
    """计算RSI指标"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, slow=26, fast=12, signal=9):
    """计算MACD指标"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    return macd

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """计算布林带"""
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    middle_band = rolling_mean
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, middle_band, lower_band