"""Analyze a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data

def assess_portfolio(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms = ['GOOG','AAPL','GLD','XOM'], \
    allocs=[0.1,0.2,0.3,0.4], \
    sv=1000000, rfr=0.0, sf=252.0, \
    gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get daily portfolio value 
    port_val = prices_SPY 
    normPrice = prices/prices.ix[0,:]
    allocPrice = normPrice * allocs
    posnValsPrice = allocPrice * sv
    port_val = allocPrice.sum(axis = 1) #posnValsPrice sum for each row
    
    
    # Get portfolio statistics (note: std_daily_ret = volatility)
    cr, adr, sddr, sr = [0.25, 0.001, 0.0005, 2.1] 
    dailyPriceReturn = (port_val / port_val.shift(1)) -1
    dailyPriceReturn = dailyPriceReturn[1:]
 
    cr = (port_val[-1]/port_val[0]) - 1  #current price / price at he beginning
    adr = dailyPriceReturn.mean()
    sddr = dailyPriceReturn.std()
    print("sddr:", sddr) 
    sr =  np.sqrt(sf) * np.mean(dailyPriceReturn - rfr    )/sddr
    print("sr:", sr)
    
    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        pvalNorm=port_val / port_val.ix[0,:]
        spyNormVal = prices_SPY / prices_SPY.ix[0,:]
        port_val = pvalNorm
        prices_SPY = spyNormVal
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
             
        ax = df_temp.plot()
        ax.set_title('Daiy Portfolio value and SPY')
        ax.set_ylabel('Normalized price')
        ax.set_xlabel('Date')
        plt.show()
        pass
    

    sv =  port_val[-1] #value of the portfolio at end of the investent period
    ev = sv

    return cr, adr, sddr, sr, ev

def test_code():

    start_date = dt.datetime(2010,1,1)
    end_date = dt.datetime(2010,12,31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']#['GOOG', 'AAPL', 'GLD', 'XOM']
    
    print("symbols",type(symbols))
    
    allocations = [0.2, 0.2, 0.3, 0.3] #[0.2, 0.3, 0.4, 0.1]
    start_val = 1000000  
    risk_free_rate = 0.0
    sample_freq = 252

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        allocs = allocations,\
        sv = start_val, \
        gen_plot = True)

    # Print statistics
    print("Start Date:", start_date)
    print( "End Date:", end_date)
    print("Symbols:", symbols)
    print("Allocations:", allocations)
    print("Sharpe Ratio:", sr)
    print("Volatility (stdev of daily returns):", sddr)
    print("Average Daily Return:", adr)
    print("Cumulative Return:", cr)

if __name__ == "__main__":
    test_code()
