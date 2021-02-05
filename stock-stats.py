import yfinance as yf 
import pandas as pd 
import numpy as np 
from datetime import datetime, date, time 
import matplotlib.pyplot as plt 
plt.style.use('dark_background')
import FundamentalAnalysis as fa 
from scipy.stats import norm

#	We create the (parent) class Data
class Data:
	"""A culmination of trading data for equities trading."""
	def __init__(self, symbol, start, end=datetime.strftime(datetime.now(), '%Y-%m-%d')):
		"""Initialize the stock's attributes via the input of its ticker symbol."""
		self.symbol = symbol
		self.start = start
		self.end = end

	def sp500(self):
		"""Return S&P 500 stock close data."""
		data = yf.download('^GSPC', start = self.start, end = self.end)

		return pd.DataFrame(data['Close'])

	def dowjones(self):
		"""Return Dow Jones Industrial Average stock close data."""
		data = yf.download('^DJI', start = self.start, end = self.end)

		return pd.DataFrame(data['Close'])

	def data(self):
		"""Return stock data downloaded from Yahoo! Finance. Input either frequency and period, or start and end dates."""
		return yf.download(self.symbol, start = self.start, end = self.end)

	def close_data(self):
		"""Return close data only."""
		data = yf.download(self.symbol, start = self.start, end = self.end)	

		return data['Close']

	def absrets(self):
		"""Return the absolute returns of a stock over the given time period."""
		data = yf.download(self.symbol, start = self.start, end = self.end)	

		return data['Close'].iloc[-1] / data['Close'].iloc[0] - 1

	def dailyrets(self):
		"""Returns (a vector of) the daily returns/price changes in a stock over time."""
		data = yf.download(self.symbol, start = self.start, end = self.end)	

		rets = data['Close'] / data['Close'].shift(1) - 1
		rets.iloc[0] = 0
		return rets 

	def logrets(self):
		"""Returns (a vector of) the daily returns/price changes in a stock over time."""
		data = yf.download(self.symbol, start = self.start, end = self.end)	

		rets = np.log(data['Close'] / data['Close'].shift(1))
		rets.iloc[0] = 0
		return rets 

	def drawdowns(self):
		"""Return the maximum drawdown (as a %) of a stock over a given period of time (using daily data)."""
		data = yf.download(self.symbol, start=self.start, end=self.end)
		rets = data['Close'] / data['Close'].shift(1)
		rets = rets[1:]

		wealth_index = 1000*rets.cumprod() # Cumulative growth of $1000 investment.
		previous_peaks = wealth_index.cummax() # Returns max value to date
		drawdown = (wealth_index - previous_peaks)/previous_peaks
		return drawdown.min()

	def skewness(self):
		"""Return the measure of skewness of a stock's daily returns. A Gaussian's is zero for reference."""
		data = yf.download(self.symbol, start=self.start, end=self.end)
		rets = data['Close'] / data['Close'].shift(1)
		rets = rets[1:]
		std = rets.std()
		skew = np.mean((rets-np.mean(rets))**3) / (std**3)
		return skew 

	def kurtosis(self):
		"""Return the measure of how 'fat' a distribution's tails are. A Gaussian's is three for reference."""
		data = yf.download(self.symbol, start=self.start, end=self.end)
		rets = data['Close'] / data['Close'].shift(1)
		rets = rets[1:]
		std = rets.std()
		kurt = np.mean((rets - np.mean(rets))**4) / (std**4)
		return kurt

	def semidev(self):
		"""Return the semi-deviation of a stock, i.e. the volatility/st. dev. only of negative returns."""
		data = yf.download(self.symbol, start=self.start, end=self.end)
		
		rets = data['Close'] / data['Close'].shift(1)
		rets = rets[1:]
		negrets = rets[rets<1]
		print(rets)
		print(negrets)

		return np.std(negrets)

	def valueatrisk(self, confidence_level=0.99):
		"""Returns Value-at-Risk (VaR), a maximum potential risk threshold at a given confidence level over a given time period."""
		data = yf.download(self.symbol, start=self.start, end=self.end)
		
		rets = data['Close'] / data['Close'].shift(1)
		rets = rets[1:]

		mean = np.mean(rets)
		std = np.std(rets)
		valueatrisk = norm.ppf(q=(1-confidence_level), loc=mean, scale=std)
		return valueatrisk - 1


	def cvar(self, confidence_level=0.99):
		"""Returns Conditional Value-at-Risk (CVaR), or the expected/average loss beyond the Value-at-Risk."""
		data = yf.download(self.symbol, start=self.start, end=self.end)
		
		rets = data['Close'] / data['Close'].shift(1)
		rets = rets[1:]

		mean = np.mean(rets)
		std = np.std(rets)
		valueatrisk = norm.ppf(q=(1-confidence_level), loc=mean, scale=std)

		rets = rets[rets<valueatrisk]
		return np.mean(rets) - 1

	def cornishfisher(self, confidence_level=0.99):
		"""Return a VaR that is adjusted based on the skewness and/or kurtosis of the actual distribution."""
		data = yf.download(self.symbol, start=self.start, end=self.end)
		
		rets = data['Close'] / data['Close'].shift(1)
		rets = rets[1:]

		mean = np.mean(rets)
		std = np.std(rets)

		z = norm.ppf(confidence_level)
		mean, var, skew, kurt = norm.stats(moments='mvsk')
		z = (z + (z**2 - 1)*skew/6 + (z**3 - 3*z)*(kurt-3)/24 - (2*z**3 - 5*z)*(skew**2)/36)
		return -(mean + z*std)

		

class Plots(Data):
	"""Graphs, plots and visualizations of downloaded raw data."""
	def __init__(self, symbol, start, end=datetime.strftime(datetime.now(), '%Y-%m-%d')):
		self.symbol = symbol
		self.start = start 
		self.end = end

	def ticker_velo_accel_plot(self, window):
		"""Plot the stock price, its 'velocity (1st) derivative', and its 'acceleration (2nd) derivative', where the window parameter is taken as the rolling average function's input."""
		velo = Data(self.symbol, self.start, self.end).dailyrets() + 1
		velo_ma = velo.rolling(window).mean()
		accel = velo / velo.shift(1)
		accel_ma = accel.rolling(window).mean()
		close = Data(self.symbol, self.start, self.end).close_data()
		close_ma = close.rolling(window).mean()
		fig, (ax1, ax2, ax3) = plt.subplots(3)
		plt.title(f'{self.symbol.upper()} Motion')
		ax1.plot(close, c='red', label='Close Price', lw=0.5)
		ax1.plot(close_ma, c='pink', label='Close MA')
		ax1.legend(loc=0, prop={'size':6})
		ax2.plot(velo, color='dodgerblue', label='1st Difference/Velo.', lw=0.5)
		ax2.plot(velo_ma, c='cyan', label='Close MA')
		ax2.hlines(1, xmin=datetime.strptime(self.start, '%Y-%m-%d'), xmax=datetime.strptime(self.end, '%Y-%m-%d'), lw=0.7, color='white', linestyle='--')
		ax2.legend(loc=0, prop={'size':6})
		ax3.plot(accel, color='green', label='2nd Difference/Accel.', lw=0.5)
		ax3.plot(accel_ma, c='yellow', label='Close MA')
		ax3.hlines(1, xmin=datetime.strptime(self.start, '%Y-%m-%d'), xmax=datetime.strptime(self.end, '%Y-%m-%d'), lw=0.7, color='white', linestyle='--')
		ax3.legend(loc=0, prop={'size':6})
		plt.show()

	def firstdiff_velo_plot(self):
		"""Plot the first difference (i.e. daily % returns) of a stock over time. It's like the "velocity" or first derivative of a stock."""
		rets = Data(self.symbol, self.start, self.end).dailyrets()
		plt.plot(rets, c='red')
		plt.title(f'{self.symbol.upper()} First Difference (Daily % Returns) Plot')
		plt.show()

	def seconddiff_accel_plot(self):
		"""Plot the second difference (second derivative or "acceleration") of a stock over time (change in daily % return)."""
		rets = Data(self.symbol, self.start, self.end).dailyrets()
		accel = rets.pct_change()
		plt.plot(accel, c='r')
		plt.title(f'{self.symbol.upper()} Second Difference (Daily % Returns Change) Plot')
		plt.show()

	def tickerplot(self):
		"""Plot the ticker of a stock given a certain start/end or period/frequency combination."""
		stockclose = super().close_data()
		sp500close = super().sp500()
		if spquestion == 'Y':
			fig, ax1 = plt.subplots()

			color = 'tab:red'
			ax1.set_xlabel('Time')
			ax1.set_ylabel(f'{self.symbol.upper()} Close', color=color)
			ax1.plot(stockclose, color=color, lw=0.8)
			ax1.tick_params(axis='y', labelcolor=color)
			ax1.tick_params(axis='x', labelsize=6)	

			ax2 = ax1.twinx()

			color = 'tab:green'
			ax2.set_ylabel('S&P 500 Close', color=color)
			ax2.plot(sp500close, color=color, lw=0.8)
			ax2.tick_params(axis='y', labelcolor=color)

			plt.title(f'{self.symbol.upper()} vs S&P 500')
			fig.tight_layout()
			plt.autoscale(enable=True, axis='x')
			plt.show()
		else:
			plt.figure(figsize=(10,6))
			plt.plot(stockclose, color='red')
			plt.xlabel('Time')
			plt.xticks(fontsize=8)
			plt.title(f'{self.symbol.upper()} Close')
			plt.show()

	def drawdownsplot(self):
		"""Plot the drawdowns of a stock over time against the stock's cumulative returns."""
		data = yf.download(self.symbol, start=self.start, end=self.end)
		rets = data['Close'] / data['Close'].shift(1)
		rets = rets[1:]
		print(rets)

		wealth_index = 1000*rets.cumprod() # Cumulative growth of $1000 investment.
		print(wealth_index)
		previous_peaks = wealth_index.cummax() # Returns max value to date
		drawdown = (wealth_index - previous_peaks)/previous_peaks
		
		fig, (ax1, ax2) = plt.subplots(2)

		ax1.set_xlabel('Time')
		ax1.set_ylabel(f'{symbol.upper()} Cumulative Returns', color='black')
		ax1.plot(wealth_index, color='red', lw=0.8, label='Return on $1000')
		ax1.plot(previous_peaks, color='blue', lw=0.8, label='Previous Max')
		ax1.tick_params(axis='x', labelsize=6)
		ax1.legend(loc=0)

		ax2.set_ylabel('Drawdowns', color='black')
		ax2.plot(drawdown, color='black', lw=0.8)
		ax2.hlines(drawdown.min(), xmin=datetime.strptime(self.start, '%Y-%m-%d'), xmax=datetime.strptime(self.end, '%Y-%m-%d'), color='red', linestyle='--', label=f'Max Drawdown: {drawdown.min().round(2)}')
		ax2.legend(loc=0)

		plt.title(f'{symbol.upper()} Cumulative Returns and Drawdowns')
		fig.tight_layout()
		plt.autoscale(enable=True, axis='x')
		plt.show()

	def dailyretsplot(self):
		"""Plot the daily returns of a stock over a certain time."""
		dailyrets = super().dailyrets()
		sp500rets = super().sp500() / super().sp500().shift(1) - 1
		sp500rets.iloc[0] = 0
		if spquestion == 'Y':
			fig, ax1 = plt.subplots()

			color = 'tab:red'
			ax1.set_xlabel('Time')
			ax1.set_ylabel(f'{symbol.upper()} Daily Returns', color=color)
			ax1.plot(dailyrets, color=color, lw=0.8)
			ax1.tick_params(axis='y', labelcolor=color)
			ax1.tick_params(axis='x', labelsize=6)	

			ax2 = ax1.twinx()

			color = 'tab:green'
			ax2.set_ylabel('S&P 500 Daily Returns', color=color)
			ax2.plot(sp500rets, color=color, lw=0.8)
			ax2.tick_params(axis='y', labelcolor=color)

			plt.title(f'{symbol.upper()} vs S&P 500 Daily Returns')
			fig.tight_layout()
			plt.autoscale(enable=True, axis='x')
			plt.show()
		else:
			plt.figure(figsize=(10,6))
			plt.plot(dailyrets, color='red')
			plt.xlabel('Time')
			plt.xticks(fontsize=8)
			plt.title(f'{symbol.upper()} Daily Returns')
			plt.show()		

	def cumulretsplot(self):
		"""Plot the cumulative returns of a stock over a certain time."""
		cumulrets = super().dailyrets().cumsum()
		sp500cumulrets = ((super().sp500() / super().sp500().shift(1)) - 1).cumsum()
		sp500cumulrets.iloc[0] = 0
		if spquestion == 'Y':
			fig, ax1 = plt.subplots()

			color = 'tab:red'
			ax1.set_xlabel('Time')
			ax1.set_ylabel(f'{symbol.upper()} Cumulative Returns', color=color)
			ax1.plot(cumulrets, color=color, lw=0.8)
			ax1.tick_params(axis='y', labelcolor=color)
			ax1.tick_params(axis='x', labelsize=6)	

			ax2 = ax1.twinx()

			color = 'tab:green'
			ax2.set_ylabel('S&P 500 Cumulative Returns', color=color)
			ax2.plot(sp500cumulrets, color=color, lw=0.8)
			ax2.tick_params(axis='y', labelcolor=color)

			plt.title(f'{symbol.upper()} vs S&P 500 Cumulative Returns')
			fig.tight_layout()
			plt.autoscale(enable=True, axis='x')
			plt.show()
		else:
			plt.figure(figsize=(10,6))
			plt.plot(cumulrets, color='red')
			plt.xlabel('Time')
			plt.xticks(fontsize=8)
			plt.title(f'{symbol.upper()} Cumulative Returns')
			plt.show()	

	def logretsplot(self):
		"""Plot the log returns of a stock over time."""
		logrets = super().logrets()
		sp500logrets = np.log(super().sp500() / super().sp500().shift(1))
		sp500logrets.iloc[0] = 0
		if spquestion == 'Y':
			fig, ax1 = plt.subplots()

			color = 'tab:red'
			ax1.set_xlabel('Time')
			ax1.set_ylabel(f'{symbol.upper()} Log Returns', color=color)
			ax1.plot(logrets, color=color, lw=0.8)
			ax1.tick_params(axis='y', labelcolor=color)
			ax1.tick_params(axis='x', labelsize=6)	

			ax2 = ax1.twinx()

			color = 'tab:green'
			ax2.set_ylabel('S&P 500 Log Returns', color=color)
			ax2.plot(sp500logrets, color=color, lw=0.8)
			ax2.tick_params(axis='y', labelcolor=color)

			plt.title(f'{symbol.upper()} vs S&P 500 Log Returns')
			fig.tight_layout()
			plt.autoscale(enable=True, axis='x')
			plt.show()
		else:
			plt.figure(figsize=(10,6))
			plt.plot(logrets, color='red')
			plt.xlabel('Time')
			plt.xticks(fontsize=8)
			plt.title(f'{symbol.upper()} Log Returns')
			plt.show() 

class Indicators(Data):
	"""Trading indicators that can be used to analyze stock data."""
	def __init__(self, symbol, start, end=datetime.strftime(datetime.now(), '%Y-%m-%d')):
		self.symbol = symbol
		self.start = start 
		self.end = end

	def velo_rets(self, window, upper=1.02, lower=0.98):
		"""Plot and print cumulative returns of a strategy where we trade on the stock's 'velocity' or first different percentage change."""
		close = Data(self.symbol, self.start, self.end).close_data()

		close = close.to_frame()

		close['velo'] = Data(self.symbol, self.start, self.end).dailyrets() + 1
		close['velo_ma'] = close['velo'].rolling(window).mean()

		close['velo_ma'].iloc[:window] = 1

		close['signal'] = np.where(close['velo_ma'] > (upper), 1, np.where(close['velo_ma'] < (lower), 0, np.nan))
		if np.isnan(close['signal'].iloc[0]) == True:
			close['signal'].iloc[0] = 0
		close['signal'] = close['signal'].fillna(method='ffill')
		close['signal'] = close['signal'].shift(1)
		close['signal'].iloc[0] = 0

		fig, (ax1, ax2) = plt.subplots(2)
		plt.title(f'{self.symbol.upper()} Close Price and Velocity Returns')
		ax1.plot(close['Close'], c='red', label='Close Price')
		ax2.plot(close['velo'], c='blue', label='Velocity', lw=0.5)
		ax2.plot(close['velo_ma'], c='cyan', label='Velocity MA')
		ax2.legend(loc=0, prop={'size':6})
		ax2.hlines(1, xmin=datetime.strptime(self.start, '%Y-%m-%d'), xmax=datetime.strptime(self.end, '%Y-%m-%d'), lw=0.7, color='white', linestyle='--')
		plt.show()

		logrets = Data(self.symbol, self.start, self.end).logrets()

		close['passive'] = np.exp(logrets.cumsum())

		close['strategy'] = np.exp((logrets*close['signal']).cumsum())

		plt.figure(figsize=(12,6))
		plt.plot(close['passive'], label='Passive Returns', color='green', lw=0.8)
		plt.plot(close['strategy'], label='Strategy Returns', color='red', lw=0.8)
		plt.title(f'{self.symbol.upper()}')
		plt.legend(loc=0)
		plt.show()

		print(f"Passive returns over this period for {symbol.upper()}: {close['passive'].iloc[-1]}")
		print(f"Strategy returns over this period for {symbol.upper()}: {close['strategy'].iloc[-1]}")


	def accel_rets(self, window, upper=1.02, lower=0.98):
		"""Plot and print cumulative returns of a strategy where we trade on the stock's 'acceleration' or second difference change."""
		close = Data(self.symbol, self.start, self.end).close_data()

		close = close.to_frame()

		close['velo'] = Data(self.symbol, self.start, self.end).dailyrets() + 1
		close['velo_ma'] = close['velo'].rolling(window).mean()
		close['accel'] = close['velo'] / close['velo'].shift(1)
		close['accel_ma'] = close['accel'].rolling(window).mean()

		close['signal'] = np.where(close['accel_ma'] > (upper), 1, np.where(close['accel_ma'] < (lower), 0, np.nan))
		if np.isnan(close['signal'].iloc[0]) == True:
			close['signal'].iloc[0] = 0
		close['signal'] = close['signal'].fillna(method='ffill')
		close['signal'] = close['signal'].shift(1)
		close['signal'].iloc[0] = 0

		fig, (ax1, ax2) = plt.subplots(2)
		plt.title(f'{self.symbol.upper()} Close Price and Acceleration Returns')
		ax1.plot(close['Close'], c='red', label='Acceleration')
		ax2.plot(close['accel'], c='green', label='Acceleration', lw=0.5)
		ax2.plot(close['accel_ma'], c='yellow', label='Acceleration MA')
		ax2.legend(loc=0, prop={'size':6})
		ax2.hlines(1, xmin=datetime.strptime(self.start, '%Y-%m-%d'), xmax=datetime.strptime(self.end, '%Y-%m-%d'), lw=0.7, color='white', linestyle='--')
		plt.show()

		logrets = Data(self.symbol, self.start, self.end).logrets()

		close['passive'] = np.exp(logrets.cumsum())

		close['strategy'] = np.exp((logrets*close['signal']).cumsum())

		print(close) 

		plt.figure(figsize=(12,6))
		plt.plot(close['passive'], label='Passive Returns', color='green', lw=0.8)
		plt.plot(close['strategy'], label='Strategy Returns', color='red', lw=0.8)
		plt.legend(loc=0)
		plt.show()

		print(f"Passive returns over this period for {symbol.upper()}: {close['passive'].iloc[-1]}")
		print(f"Strategy returns over this period for {symbol.upper()}: {close['strategy'].iloc[-1]}")

	def accel_and_velo_rets(self, velo_window, accel_window, upper=1.02, lower=0.98):
		"""An indicator that incorporates both acceleration and velocity. Buy if both moving averages are positive (or above some threshold). Sell when velocity drops below zero (or some threshold)"""
		close = Data(self.symbol, self.start, self.end).close_data()

		close = close.to_frame()

		close['velo'] = Data(self.symbol, self.start, self.end).dailyrets() + 1
		close['velo_ma'] = close['velo'].rolling(velo_window).mean()
		close['accel'] = close['velo'] / close['velo'].shift(1)
		close['accel_ma'] = close['accel'].rolling(accel_window).mean()

		close['signal'] = None
		for i in range(len(close)):
			if close['accel_ma'].iloc[i] > upper:
				if close['velo_ma'].iloc[i] > upper:
					close['signal'].iloc[i] = 1
			elif close['velo_ma'].iloc[i] < lower and close['accel_ma'].iloc[i] < lower:
				close['signal'].iloc[i] = 0
			else:
				close['signal'].iloc[i] = np.nan

		if np.isnan(close['signal'].iloc[0]) == True:
			close['signal'].iloc[0] = 0
		close['signal'] = close['signal'].fillna(method='ffill')
		close['signal'] = close['signal'].shift(1)
		close['signal'].iloc[0] = 0

		fig, (ax1, ax2) = plt.subplots(2)
		plt.title(f'{self.symbol.upper()} Close Price and Velo-Accel Hybrid Returns')
		ax1.plot(close['Close'], c='red', label='Close Price')
		ax2.plot(close['velo_ma'], c='cyan', label='Velocity MA', lw=0.5)
		ax2.plot(close['accel_ma'], c='yellow', label='Acceleration MA')
		ax2.legend(loc=0, prop={'size':6})
		ax2.hlines(1, xmin=datetime.strptime(self.start, '%Y-%m-%d'), xmax=datetime.strptime(self.end, '%Y-%m-%d'), lw=0.7, color='white', linestyle='--')
		plt.title(f'{self.symbol.upper()} ')
		plt.show()

		logrets = Data(self.symbol, self.start, self.end).logrets()

		close['passive'] = np.exp(logrets.cumsum())

		close['strategy'] = np.exp((logrets*close['signal']).cumsum())

		print(close) 

		plt.figure(figsize=(12,6))
		plt.plot(close['passive'], label='Passive Returns', color='green', lw=0.8)
		plt.plot(close['strategy'], label='Strategy Returns', color='red', lw=0.8)
		plt.title(f'{symbol.upper()} Passive and Strategy Returns')
		plt.legend(loc=0)
		plt.show()

		print(f"Passive returns over this period for {symbol.upper()}: {close['passive'].iloc[-1]}")
		print(f"Strategy returns over this period for {symbol.upper()}: {close['strategy'].iloc[-1]}")

	def macdplot(self, shortspan, longspan, signalspan):
		"""Given a short-term EMA, long-term EMA, and signal, plot the the MACD trading strategy beside the ticker."""
		df = yf.download(self.symbol, period='max', frequency='1d')
		df['shortEMA'] = df['Close'].ewm(span=shortspan).mean()
		df['longEMA'] = df['Close'].ewm(span=longspan).mean()
		df['macd'] = df['shortEMA'] - df['longEMA']
		df['signal'] = df['macd'].ewm(span=signalspan).mean()

		macd = df['macd']
		signal = df['signal']
		close = df['Close']

		data = super().close_data()
		df = df.iloc[-len(data):]

		fig, (ax1, ax2) = plt.subplots(2)

		ax1.set_xlabel('Time')
		ax1.set_ylabel(f'{self.symbol.upper()} Close')
		ax1.plot(df['Close'], color='red', lw=0.8)
		ax1.tick_params(axis='x', labelsize=6)	

		plt.title(f'{self.symbol.upper()} Close and MACD ({shortspan}-{longspan}-{signalspan})')

		def a1(y):
			return y / close

		def a2(y):
			return y * close

		ax2.set_xlabel('Time')
		ax2.set_ylabel('MACD')
		ax2.plot(df['macd'], color='magenta', lw=0.8, label='MACD')
		ax2.plot(df['signal'], color='cyan', lw=0.8, label='Signal')
		ax2.legend(loc=0)
		ax2.tick_params(axis='y')
		ax2.tick_params(axis='x', labelsize=6)
		ax3 = ax2.twinx()
		mn, mx = ax2.get_ylim()
		ax3.set_ylim(mn/df['Close'].iloc[-1], mx/df['Close'].iloc[-1])
		ax3.set_ylabel('MACD as Fraction of Recent Close Price')

		fig.tight_layout()
		plt.autoscale(enable=True, axis='x')
		plt.show()

	def macdretsandplot(self, shortspan, longspan, signalspan):
		"""Given short-term EMA, long-term EMA and signal, plot and calculate MACD returns (and passive returns as comparison)."""
		df = yf.download(self.symbol, period='max', frequency='1d')
		df['shortEMA'] = df['Close'].ewm(span=shortspan).mean()
		df['longEMA'] = df['Close'].ewm(span=longspan).mean()
		df['macd'] = df['shortEMA'] - df['longEMA']
		df['signal'] = df['macd'].ewm(span=signalspan).mean()
		df['trading signal'] = np.where(df['macd'] > df['signal'], 1, 0) # buy signals only
		df['trading signal'] = df['trading signal'].shift(1) # eliminate foresight bias
		df['passive'] = super().logrets()
		df['strategy'] = df['passive'] * df['trading signal']

		data = super().close_data()
		df = df.iloc[-len(data):-1]
		
		passivetot = np.exp(df['passive'].sum())
		strategytot = np.exp(df['strategy'].sum())

		print(f'{self.symbol.upper()} Cumulative Strategy Returns (long positions only): {strategytot}')
		print(f'{self.symbol.upper()} Passive Returns: {passivetot}')

		plt.figure(figsize=(12,6))
		plt.plot(np.exp(df['strategy'].cumsum()), color='red', lw=0.8, label='Strategy Returns')
		plt.plot(np.exp(df['passive'].cumsum()), color='green', lw=0.8, label='Passive Returns')
		plt.tick_params(axis='x', labelsize=6)
		plt.legend(loc=0)
		plt.title(f'MACD ({shortspan}-{longspan}-{signalspan}) and Passive Cumulative Returns for {self.symbol.upper()}')
		plt.show()

	def macdrets(self, shortspan, longspan, signalspan):
		"""Given short EMA, long EMA and signal, calculate MACD returns (and passive returns for comparison)."""
		df = yf.download(self.symbol, period='max', frequency='1d')
		df['shortEMA'] = df['Close'].ewm(span=shortspan).mean()
		df['longEMA'] = df['Close'].ewm(span=longspan).mean()
		df['macd'] = df['shortEMA'] - df['longEMA']
		df['signal'] = df['macd'].ewm(span=signalspan).mean()
		df['trading signal'] = np.where(df['macd'] > df['signal'], 1, 0) # buy signals only. track macd/signal trading signal
		
		# The actual trading signal will require a trading signal as well as a greater than 2% deviation from shortEMA/longEMA equivalence.
		# i.e. MACD / Stock price < -2% + MACD line crosses above signal line = BUY
		# MACD / Stock price > 2% + MACD line crosses below signal line = SELL
		      
		df['w_macd'] = df['macd'] / df['Close'] # weigh MACD relative to stock price
		df['w_signal'] = df['signal'] / df['Close'] # weigh signal relative to stock price
		df['crossover'] = np.sign(df['w_macd'] - df['w_signal'])
		df['crossover'] = df['crossover'].diff() / 2 # This will yield +1 if MACD crossed above the signal line yesterday, -1 if it crossed below, and zero otherwise!
		df['tradeable_yn'] = None
		for i in range(len(df)):
			if (df['w_signal'].iloc[i] > 0.02) & (df['crossover'].iloc[i] < 0):
				df['tradeable_yn'].iloc[i] = 1
			elif (df['w_signal'].iloc[i] < -0.02) & (df['crossover'].iloc[i] > 0):
				df['tradeable_yn'].iloc[i] = 1
			else:
				df['tradeable_yn'].iloc[i] = 0
		
		df['trading signal'] = np.nan
		for i in range(len(df)):
			if df['tradeable_yn'].iloc[i] == 1:
				if df['crossover'].iloc[i] == 1:
					df['trading signal'] = 1
		      		elif df['crossover'].iloc[i] == -1:
		      			df['trading signal'] = 0 # long securities only, no short selling (although 0 could simply be changed to -1).		      

		df['trading signal'] = df['trading signal'].shift(1) # eliminate foresight bias

		df['trading signal'] = df['trading signal'].ffill()
		df['passive'] = super().logrets()
		df['strategy'] = df['passive'] * df['trading signal']

		data = super().close_data()
		df = df.iloc[-len(data):-1]
		
		passivetot = np.exp(df['passive'].sum())
		strategytot = np.exp(df['strategy'].sum())

	def adaptmacd(self, shortspan, longspan, signalspan, vf=2, bbf=2, rollingstd=5):
		"""Given short-term EMA, long-term EMA and signal, plot and calculate an adaptive moving average indicator -- one where the spans depend on volatility 
		(specifically volatility squared or variance). More volatile times produce longer-term averages to drown out the added 'noise.'"""
		df = yf.download(self.symbol, period='10y', frequency='1d')
		df['longBullBear'] = None
		for i in range(len(df)-longspan):
			df['longBullBear'].iloc[i+longspan] = df['Close'].iloc[i+longspan] / df['Close'].iloc[i]
		df['shortBullBear'] = None
		for i in range(len(df)-longspan):
			df['shortBullBear'].iloc[i+longspan] = df['Close'].iloc[i+longspan] / df['Close'].iloc[i]
		df['signalBullBear'] = None
		for i in range(len(df)-longspan):
			df['signalBullBear'].iloc[i+longspan] = df['Close'].iloc[i+longspan] / df['Close'].iloc[i]

		print(df)

		df['Daily Rets'] = df['Close'].pct_change() + 1
		df['Ratio'] = (df['Daily Rets'].rolling(rollingstd).std()) / df['Daily Rets'].std()
		df['adaptshort'] = shortspan / df['Ratio']**vf * df['shortBullBear']**bbf
		df['adaptlong'] = longspan / df['Ratio']**vf * df['longBullBear']**bbf
		df['adaptsignal'] = signalspan / df['Ratio']**vf * df['signalBullBear']**bbf

		df = df[np.maximum(shortspan, np.maximum(longspan, np.maximum(signalspan, rollingstd)))+1:]

		shortsp = df['adaptshort'].to_list()
		longsp = df['adaptlong'].to_list()
		signalsp = df['adaptsignal'].to_list()
		df['shortspan'] = [int(i) for i in shortsp]
		df['longspan'] = [int(i) for i in longsp]
		df['signalspan'] = [int(i) for i in signalsp]
		print(df['Ratio'].iloc[-30:])
		print(df['Daily Rets'].std())

		df['shortEMA'] = None
		df['longEMA'] = None
		df['signal'] = None
		for i in range(len(df)):
			if np.isnan(df['shortspan'].iloc[i]) == True:
				df['shortEMA'].iloc[i] = np.nan
			else: 			
				df['shortEMA'].iloc[i] = np.mean(df['Close'].iloc[(i-df['shortspan'].iloc[i]):i])
		for i in range(len(df)):
			if np.isnan(df['longspan'].iloc[i]) == True:
				df['longEMA'].iloc[i] = np.nan
			else: 	
				df['longEMA'].iloc[i] = np.mean(df['Close'].iloc[(i-df['longspan'].iloc[i]):i])
		
		df['adaptmacd'] = df['shortEMA'] - df['longEMA']
		
		for i in range(len(df)):
			if np.isnan(df['signalspan'].iloc[i]) == True:
				df['signal'].iloc[i] = np.nan
			else: 	
				df['signal'].iloc[i] = np.mean(df['adaptmacd'].iloc[(i-df['signalspan'].iloc[i]):i])

		df['trading signal'] = np.where(df['adaptmacd'] > df['signal'], 1, 0) # buy signals only
		df['trading signal'] = df['trading signal'].shift(1) # eliminate foresight bias
		
		df = df[self.start:self.end]

		print(df)
		print(df['trading signal'])

		df['passive'] = super().logrets()
		df['strategy'] = df['passive'] * df['trading signal']

		passive = df['passive']
		strategy = df['strategy']

		print(f'{self.symbol.upper()} Cumulative Strategy Returns (long positions only): {np.exp(strategy.sum())}')
		print(f'{self.symbol.upper()} Passive Returns: {np.exp(passive.sum())}')

		plt.figure(figsize=(12,6))
		plt.plot(np.exp(df['strategy'].cumsum()), color='red', lw=0.8, label='Strategy Returns')
		plt.plot(np.exp(df['passive'].cumsum()), color='green', lw=0.8, label='Passive Returns')
		plt.tick_params(axis='x', labelsize=6)
		plt.title(f'Variance-Dependent MACD ({shortspan}-{longspan}-{signalspan}) and Passive Cumulative Returns for {self.symbol.upper()}')
		plt.legend(loc=0)
		plt.show()
