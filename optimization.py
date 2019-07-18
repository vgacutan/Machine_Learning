"""Optimize a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
import scipy.optimize as spo


def sr_min(allocs, prices):
    #print "allocs", allocs
    normPrice = prices/prices.ix[0,:]
    allocPrice = normPrice * allocs
    port_val = allocPrice.sum(axis=1)
    dailyPriceReturn = (port_val / port_val.shift(1)) -1
    dailyPriceReturn = dailyPriceReturn[1:] # Exclude the first row
    
    sddr = dailyPriceReturn.std()
    rfr = 0.0  # Assume 252 trading days in a year and a risk free return of 0.0 per day
    sf = 252.0 # Assume 252 trading days in a year and a risk free return of 0.0 per day

    sr =  np.sqrt(sf) * np.mean(dailyPriceReturn - rfr)/sddr
    return sr*(-1)  #
    
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be correct for a test case
    allocs = np.asarray([0.2, 0.2, 0.3, 0.3, 0.0])
    
    tickerSymbCnt = len(syms)
    init_guess = [1.0 / tickerSymbCnt] * tickerSymbCnt #assumed equal distribution
    bnd = [(0.0, 1.0)] * tickerSymbCnt  #slow, high bounds.  stocks not exceeding 100% per stocks
    cstr = {'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs)}
    #finding solutions for optimum stocks allocations
    #used these reference to understand concept of scipy.minimize: https://www.youtube.com/watch?v=cXHvC_FGx24,https://www.youtube.com/watch?v=o4_Mkz-EeXE
    allocs = spo.minimize(sr_min, init_guess, args=prices, method = 'SLSQP', options={'disp': False}, bounds=bnd, constraints=cstr).x
    cr, adr, sddr, sr = [0.25, 0.001, 0.0005, 2.1]
    
    normPrice = prices/prices.ix[0,:]
    allocPrice = normPrice * allocs

    # Get daily portfolio value
    port_val = prices_SPY 
    port_val = allocPrice.sum(axis = 1)
    dailyPriceReturn = (port_val / port_val.shift(1)) -1
    dailyPriceReturn = dailyPriceReturn[1:] #ignoring first row
    
    cr = port_val[-1]/port_val[0] - 1
    adr = dailyPriceReturn.mean()
    sddr = dailyPriceReturn.std()
    rfr = 0.0  # Assume 252 trading days in a year and a risk free return of 0.0 per day
    sf = 252.0  # Assume 252 trading days in a year and a risk free return of 0.0 per day
    sr =  np.sqrt(sf) * np.mean(dailyPriceReturn - rfr)/sddr

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
 
        pvalNorm = port_val / port_val.ix[0,:]
        spyNormVal = prices_SPY / prices_SPY.ix[0,:]
        port_val = pvalNorm
        prices_SPY = spyNormVal
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        ax = df_temp.plot()
        ax.set_title('Daiy Portfolio value and SPY')
        ax.set_ylabel('Price')
        ax.set_xlabel('Date')
        plt.show()
        pass

    return allocs, cr, adr, sddr, sr

def test_code():
    

    start_date = dt.datetime(2008,6,1)
    end_date = dt.datetime(2009,6,1)
    symbols = ['IBM', 'X', 'GLD']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = True)

    # Print statistics
    print("Start Date:", start_date)
    print("End Date:", end_date)
    print("Symbols:", symbols)
    print("Allocations:", allocations)
    print("Sharpe Ratio:", sr)
    print("Volatility (stdev of daily returns):", sddr)
    print("Average Daily Return:", adr)
    print("Cumulative Return:", cr)

if __name__ == "__main__":
    test_code()
