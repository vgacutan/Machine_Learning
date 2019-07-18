
"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def author(self):
         return 'vgacutan3'
         
def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000):
    # this is the function the autograder will call to test your code
    # TODO: Your code here
    
    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2008,6,1)
    portvals = get_data(['IBM'], pd.date_range(start_date, end_date))
    portvals = portvals[['IBM']]  # remove SPY

    orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    print(orders)
    orders = orders.sort_index(ascending = True)
    symb = pd.unique(orders[['Symbol']].values.ravel())
    symblist = symb.tolist()
 
    sd = orders.index.min()
    ed = orders.index.max()
    dates = pd.date_range(sd, ed)
    dates = get_data(['SPY'], dates).index.tolist() #get dates where SPY is trading
 
    pricesdf = get_data(symbols=symblist, dates=dates)
    prices = pricesdf[symblist]

    prices = pd.concat([prices, pd.DataFrame(index=dates)], axis=1)
    prices = prices.fillna(method='ffill') 
  
    dfHoldVal = pd.DataFrame(index=[dates],columns=['holdCashVal'])
    dfHoldVal.iloc[0]['holdCashVal'] = start_val
    dfHoldVal.fillna(0, inplace = True)

    dfSharesVal = pd.DataFrame(index=[dates], columns=symblist)
    dfSharesVal.fillna(0, inplace = True)
    
    dfCashVal = pd.DataFrame(index=[dates], columns=['cash']) 
    dfCashVal.iloc[0] = 0

    for i in range(orders.shape[0]):
        
        if orders.iloc[i]['Order'] == 'BUY':
            
            ordValIndicator = orders.iloc[i]['Shares'] *(1.0)
            sym=dfSharesVal.loc[orders.index[i], orders.iloc[i]['Symbol']]
            dfSharesVal.loc[orders.index[i], orders.iloc[i]['Symbol']] = sym +  ordValIndicator
            amntTrade =  prices.loc[orders.index[i]][orders.iloc[i]['Symbol']] * ordValIndicator
                      
        else:           
            ordValIndicator = orders.iloc[i]['Shares'] *(-1.0)
            sym=dfSharesVal.loc[orders.index[i], orders.iloc[i]['Symbol']]
            dfSharesVal.loc[orders.index[i], orders.iloc[i]['Symbol']] = sym + ordValIndicator 
            amntTrade =  prices.loc[orders.index[i]][orders.iloc[i]['Symbol']] * ordValIndicator 
        hval = dfHoldVal.loc[orders.index[i], ['holdCashVal']]      
        dfHoldVal.loc[orders.index[i],['holdCashVal']] = hval - amntTrade

    dfCashVal.cash = dfHoldVal.holdCashVal.cumsum()
    tradeShares = dfSharesVal.cumsum()
    dfCashVal = pd.concat([tradeShares * prices,dfCashVal],axis=1)
    portvals = dfCashVal.sum(axis=1)
   
    #print(portvals)
    return portvals

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2008,6,1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    # Compare portfolio against $SPX
    print("Date Range: {} to {}".format(start_date, end_date))
    print
    print("Sharpe Ratio of Fund: {}".format(sharpe_ratio))
    print("Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY))
    print
    print("Cumulative Return of Fund: {}".format(cum_ret))
    print("Cumulative Return of SPY : {}".format(cum_ret_SPY))
    print
    print("Standard Deviation of Fund: {}".format(std_daily_ret))
    print("Standard Deviation of SPY : {}".format(std_daily_ret_SPY))
    print
    print("Average Daily Return of Fund: {}".format(avg_daily_ret))
    print("Average Daily Return of SPY : {}".format(avg_daily_ret_SPY))
    print
    print("Final Portfolio Value: {}".format(portvals[-1]))
    
if __name__ == "__main__":
    test_code()

