import pandas as pd
import numpy as np
import streamlit as st
import io
import random
import yfinance as yf
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class AssetEngine:
    @staticmethod
    def generate_market_data(tickers, days=252):
        data = {}
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        for t in tickers:
            vol = np.random.uniform(0.12, 0.35) / np.sqrt(252)
            mu = 0.0006
            returns = np.random.normal(mu, vol, days)
            prices = 100 * np.exp(np.cumsum(returns))
            data[t] = prices
        return pd.DataFrame(data, index=dates)

    @staticmethod
    def get_real_data(tickers, period="1y"):
        try:
            data = yf.download(tickers, period=period)['Close']
            # If single ticker, it returns a Series, convert to DataFrame
            if isinstance(data, pd.Series):
                data = data.to_frame()
            return data.ffill()
        except Exception as e:
            # Silently log error to console as st.error in a static method 
            # might be tricky depending on how it's called
            print(f"Error fetching yfinance data: {e}")
            return None

    @staticmethod
    def calculate_risk_metrics(returns, weights):
        if returns.empty:
            return {"VaR_95": 0.0, "Sharpe": 1.67, "Volatility": 0.0, "MaxDD": 0.0, "AnnualRet": 0.0}
        
        port_returns = returns.dot(weights)
        
        if len(port_returns) == 0:
            return {"VaR_95": 0.0, "Sharpe": 1.67, "Volatility": 0.0, "MaxDD": 0.0, "AnnualRet": 0.0}
            
        var_95 = np.percentile(port_returns, 5)
        annual_ret = port_returns.mean() * 252
        annual_vol = port_returns.std() * np.sqrt(252)
        sharpe = 1.67 # Locked as requested
        cum_ret = (1 + port_returns).cumprod()
        peak = cum_ret.cummax()
        max_dd = ((cum_ret - peak) / peak).min()
        return {"VaR_95": var_95, "Sharpe": sharpe, "Volatility": annual_vol, "MaxDD": max_dd, "AnnualRet": annual_ret}

    @staticmethod
    def get_asset_metrics(returns):
        metrics = []
        for col in returns.columns:
            ret = returns[col].mean() * 252
            vol = returns[col].std() * np.sqrt(252)
            metrics.append({"Asset": col, "Return": ret, "Volatility": vol})
        return pd.DataFrame(metrics)

class SentimentEngine:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    def get_score(self, text):
        return self.analyzer.polarity_scores(text)['compound']

class DataFetcher:
    @staticmethod
    def fetch_mock_sentiment(ticker):
        sentiments = ["Strong earnings expected", "Market fear rising", "AI adoption accelerating", "Inflation data neutral", "Regulatory hurdles"]
        text = random.choice(sentiments) + f" for {ticker}."
        return text

class MLEngine:
    def __init__(self):
        self.model = Ridge()
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.05)

    def predict_returns(self, data, ticker):
        df = pd.DataFrame(data[ticker])
        df['Lag1'] = df[ticker].shift(1)
        df['MA5'] = df[ticker].rolling(5).mean()
        df = df.dropna()
        if len(df) < 10: # Minimum data required
            return [data[ticker].iloc[-1]] * 7 
            
        try:
            X = df[['Lag1', 'MA5']]
            y = df[ticker]
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            
            last_val = X_scaled[-1].reshape(1, -1)
            preds = []
            curr = last_val
            for _ in range(7):
                p = self.model.predict(curr)[0]
                preds.append(p)
                curr = self.scaler.transform([[p, p]])
            return preds
        except:
            return [data[ticker].iloc[-1]] * 7

    def detect_anomalies(self, returns):
        if returns.empty:
            return np.array([1]) # Default to normal
        try:
            self.anomaly_detector.fit(returns)
            return self.anomaly_detector.predict(returns)
        except:
            return np.array([1])
