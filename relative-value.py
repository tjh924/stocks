import seaborn
from datetime import datetime
import matplotlib.pyplot as plt 
import yfinance as yf
import pandas as pd
import numpy as np
from numpy import log, polyfit, sqrt, std, subtract
import itertools
from itertools import product
import scipy as sci
from scipy import stats
import statsmodels
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts 
import hurst
from hurst import compute_Hc

from stock-stats import Data

class RelativeValue(Data):
	"""A starting point for programming relative value trading."""
	def __init__(self, symbols, start='2016-01-01', end=datetime.now().strftime('%Y-%m-%d'), **kwargs):
		"""Initiate a stock's attributes via the input of its ticker symbol."""
		self.symbols = symbols
		self.start = start
		self.end = end
		self.__dict__.update(**kwargs)

	def corr_matrix(self):
		closedata = pd.DataFrame()
		for symbol in symbols:
			close = Data(symbol, self.start, self.end).logrets()
			closedata.insert(0, f'{symbol.upper()} LogRets', close)

		seaborn.heatmap(closedata.corr(), xticklabels=symbols, yticklabels=symbols, cmap='RdYlGn', annot=True)
		plt.title('Correlation Matrix of Given Stocks')
		plt.show()

	def coint_matrix(self):
		"""Create a cointegration matrix 'heatmap' that shows the cointegration of stock pairs."""
		n = len(symbols)
		score_matrix = np.zeros((n,n))
		pvalue_matrix = np.ones((n,n))
		for i in range(n):
			for j in range(n):
				S1 = Data(symbols[i], self.start, self.end).logrets()
				S2 = Data(symbols[j], self.start, self.end).logrets()
				spread = S1 - ((S1/S2).mean())*S2
				result = ts.adfuller(spread)
				score = result[0]
				pvalue = result[1]
				score_matrix[i, j] = score
				pvalue_matrix[i, j] = pvalue
		seaborn.heatmap(pvalue_matrix, xticklabels=symbols, yticklabels=symbols, cmap='RdYlGn_r', annot=True)
		plt.title('Cointegration Matrix of Given Stocks')
		plt.show()


	def coint_pairs(self):
		"""Return stock pairs from symbols list which are well-cointegrated, i.e. pairs whose p-value is less than 0.05."""
		n = len(symbols)
		pairs = []
		for i in range(n):
			for j in range(n):
				S1 = Data(symbols[i], self.start, self.end).logrets()
				S2 = Data(symbols[j], self.start, self.end).logrets()
				spread = S1 - ((S1/S2).mean())*S2
				result = ts.adfuller(spread)
				pvalue = result[1]
				if pvalue < 0.05:
					if symbols[i] == symbols[j]:
						pass
					else:
						pairs.append((symbols[i], symbols[j]))
		if not pairs:
			print(f'\nThere are no sufficiently cointegrated pairs in this group of stocks!')
		else:
			print('')
			print(*pairs, sep='\n')
		
	def spread(self):
		"""Plot the spread of the prices of two stocks."""
		n = len(symbols)
		for i in range(n):
			for j in range(i+1, n):
				y = Data(symbols[i], self.start, self.end).close_data()
				x = Data(symbols[j], self.start, self.end).close_data()

				(y/x).plot(figsize=(10,5), color='orchid')
				plt.axhline((y/x).mean(), color='red', linestyle='--')
				plt.legend([f'Spread', 'Mean (Hedge Ratio)'])
				plt.title(f'{symbols[i]} / {symbols[j]} Spread')
				plt.ylabel(f'{symbols[i]}/{symbols[j]}')
				plt.show()

	def logretsspread(self):
		"""Plot the spread of the log returns of two stocks."""
		n = len(symbols)
		for i in range(n):
			for j in range(i+1, n):
				y = Data(symbols[i], self.start, self.end).logrets()
				x = Data(symbols[j], self.start, self.end).logrets()

				(y-x).plot(figsize=(10,5), color='orchid')
				plt.axhline((y-x).mean(), color='red', linestyle='--')
				plt.legend(['Log Returns Spread', 'Mean (Hedge Ratio)'])
				plt.title(f'{symbols[i]} - {symbols[j]} Log Returns Spread')
				plt.ylabel(f'{symbols[i]} LogRets - {symbols[j]} LogRets')
				plt.show()

	def hurst_matrix(self):
		"""Create a matrix that prints Hurst exponents."""
		n = len(symbols)
		matrix = np.ones((n, n))
		np.seterr(divide='ignore')
		for i in range(n):
			for j in range(n):
				if symbols[i] == symbols[j]:
					matrix[i, j] = None
				else:
					S1 = Data(symbols[i], self.start, self.end).logrets().dropna()
					S2 = Data(symbols[j], self.start, self.end).logrets().dropna()
					S1 = S1[1:]
					S2 = S2[1:]
					ratio = S1/S2
					spread = S1 - (np.mean(ratio))*S2
					result = compute_Hc(spread)
					hurst = result[0]
					matrix[i, j] = hurst 
		seaborn.heatmap(matrix, xticklabels=symbols, yticklabels=symbols, cmap='RdYlGn_r', vmin=0.4, vmax=0.6, center=0.5, annot=True)
		plt.title('Matrix of Hurst Exponents for Given Stocks')
		plt.show()
