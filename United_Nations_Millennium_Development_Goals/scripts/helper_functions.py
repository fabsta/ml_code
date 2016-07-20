#DateTime will be converted to categorical Year, Month and Day of the Week
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm

# graphix
import matplotlib.pyplot as plt
import prettyplotlib as pplt
import seaborn as sns
import statsmodels.graphics.tsaplots as tsaplots
from sklearn.linear_model import LinearRegression

from pandas.tseries.offsets import *
#%matplotlib inline

# utility
import os





def predict_for_category(df,column, plot_it=None):
    """ 
    make a list of column names for specific years
    in the format they appear in the data frame start/stop inclusive
    """   
    try:
        training_drop = df[column]
        ts_data = pd.Series(training_drop.values, index=pd.to_datetime(training_drop.index))
        ts_log_data = np.log(ts_data)

        model = sm.tsa.ARMA(ts_log_data, order=(1,0)).fit()

        y_pred = model.predict(ts_data.index[0].isoformat(), ts_data.index[-1].isoformat())
        start_date = ts_data.index[-1] + Day(365)
        end_date = ts_data.index[-1] + Day(1826)
        y_forecast = model.predict(start_date.isoformat(), end_date.isoformat())
        
        if(plot_it):
            plt.figure()
            aus, = plt.plot(ts_data)
            log, = plt.plot(ts_log_data)
            wrld, = plt.plot(y_pred)
            new_pred, = plt.plot(y_forecast)
            plt.legend([aus, log,wrld,new_pred], ['Real','Log', 'Predicted','New'])
            plt.title('Real vs predicted rates')
            
        reverted_forecasts = np.exp(y_forecast)
        #print     reverted_forecasts[0],reverted_forecasts[4]
        return (reverted_forecasts[0],reverted_forecasts[4])
    except Exception:
        return simple_model(df[column])




def simple_model(series):
    
    point_2007 = series[-1]
    point_2006 = series[-2]
    
    # if just one point, status quo
    if np.isnan(point_2006):
        return point_2007,point_2007
    else:
        #if np.isnan(point_2005):
        slope = point_2007 - point_2006
        #else:
        #    slope = (point_2006 - point_2005) + (point_2007 - point_2006)
        # one year
        pred_2008 = point_2007 + slope
        
        # five years
        pred_2012 = point_2007 + 2.5*slope
        return (pred_2008,pred_2012)



def generate_year_list(start, stop=None):
    """ 
    make a list of column names for specific years
    in the format they appear in the data frame start/stop inclusive
    """
    
    if isinstance(start, list):
        data_range = start
    elif stop:
        data_range = range(start, stop+1)
    else:
        data_range = [start]
    
    yrs = []
    
    for yr in data_range:
        yrs.append("{0}".format(yr))
        
    return yrs



def get_correlations(training_data,country, to_predict_ix):
    kenya_data = training_data[training_data["Country Name"] == country]
    kenya_values = kenya_data[generate_year_list(1972, 2007)].values
    
    only_years = kenya_data[generate_year_list(1972, 2007)]
    only_years.plot()
    # get the total number of time series we have for Kenya
    nseries = kenya_values.shape[0]
    
    # -1 as default
    # returns a new array filled with ones
    lag_corr_mat = np.ones([nseries, nseries], dtype=np.float64)*-1
    
    # create a matrix to hold our lagged correlations
    for i in range(nseries):
        for j in range(nseries):
            # skip comparing a series with itself
            if i!=j:
                # get original (1972-2006) and shifted (1973-2007)
                original = kenya_values[i,1:]
                shifted = kenya_values[j,:-1]
                
                # for just the indices where neither is nan
                non_nan_mask = (~np.isnan(original) & ~np.isnan(shifted))
                
                # if we have at least 2 data points
                if non_nan_mask.sum() >= 2:
                    lag_corr_mat[i,j] = np.correlate(original[non_nan_mask], shifted[non_nan_mask])
                    
    
    # first, we get the index of that row in the correlation matrix
    i = np.where(kenya_data.index.values == to_predict_ix)[0][0]
    
    # then, we see which value in the matrix is the largest for that row
    j_max = np.argmax(lag_corr_mat[i,:])
    
    # finally, let's see what these correspond to
    max_corr_ix = kenya_data.index.values[j_max]
    
    j_max = np.argmax(lag_corr_mat[i,:])                
    return j_max                        