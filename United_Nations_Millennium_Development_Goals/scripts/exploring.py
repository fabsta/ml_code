# -*- coding: utf-8 -*-
# data manipulation and modeling
import numpy as np
import pandas as pd
import statsmodels.api as sm

# graphix
import matplotlib.pyplot as plt
import prettyplotlib as pplt
import seaborn as sns
import statsmodels.graphics.tsaplots as tsaplots

# utility
import os

# notebook parameters
pd.set_option('display.max_columns', 40) # number of columns in training set
plt.rcParams['figure.figsize'] = (14.0, 8.0)


in_file = '/Users/fabianschreiber/Documents/projects/kaggle/united_nations/data/TrainingSet.csv'
submission_file = '/Users/fabianschreiber/Documents/projects/kaggle/united_nations/SubmissionRows.csv'
print("Loading data...\n")

training_data = pd.read_csv(in_file, index_col=0)
submission_labels = pd.read_csv(submission_file, index_col=0)

training_data.loc[559]
all_series = np.array(np.unique(training_data['Series Name']))

#print all_series

prediction_rows = training_data.loc[submission_labels.index]
np.unique(prediction_rows['Series Name'])
prediction_rows = prediction_rows[generate_year_list(1972, 2007)]



# grab a random sample of 10 of the timeseries
np.random.seed(896)
rand_rows = np.random.choice(prediction_rows.index.values, size=10)

plot_rows(prediction_rows, ids=rand_rows)
plt.show()



# let's try just these predictions on the first five rows
test_data = prediction_rows.head()
test_predictions = test_data.apply(simple_model, axis=1)

# combine the data and the predictions
test_predictions = test_data.join(test_predictions)

# let's take a look at 2006, 2007, and our predictions
test_predictions[generate_year_list([2006, 2007, 2008, 2012])]

# make the predictions
predictions = prediction_rows.loc[rand_rows].apply(simple_model, axis=1)

# plot the data
plot_rows(prediction_rows, ids=rand_rows)

# plot the predictions
plot_rows(predictions, linestyle="--", legend=False)

plt.show()


simple_predictions = prediction_rows.apply(simple_model, axis=1)
write_submission_file(simple_predictions, "/Users/fabianschreiber/Documents/projects/kaggle/united_nations/first_submission.csv")





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


def plot_rows(data, ids=None, linestyle="-", legend=True):
    # get some colors for the lines
    bmap = pplt.brewer2mpl.get_map('Set3','Qualitative', 10)
    colors = bmap.mpl_colors
    
    if not None == ids:
        get_rows = lambda: enumerate(ids)
    else:
        get_rows = lambda: enumerate(data.index.values)
    
    for i, r in get_rows():
        # get the time series values
        time_data = data.loc[r]

        # create an x axis to plot along
        just_years = [y[:4] for y in data.columns]
        X = pd.DatetimeIndex(just_years)

        # get time series info for labeling
        country, descrip = training_data[["Country Name", "Series Name"]].loc[r]

        # plot the series
        plt.plot(X, time_data, c=colors[i],
                 label="{} - {}".format(country, descrip), ls=linestyle)
        plt.scatter(X, time_data, alpha=0.8, c=colors[i])

    if legend:
        plt.legend(loc=0)
    plt.title("Progress Towards Subset of MDGs")


def simple_model(series):
    
    point_2007 = series.iloc[-1]
    point_2006 = series.iloc[-2]
    point_2005 = series.iloc[-3]
    
    # if just one point, status quo
    if np.isnan(point_2006):
        predictions = np.array([point_2007, point_2007])
    else:
        #if np.isnan(point_2005):
        slope = point_2007 - point_2006
        #else:
        #    slope = (point_2006 - point_2005) + (point_2007 - point_2006)
        # one year
        pred_2008 = point_2007 + slope
        
        # five years
        pred_2012 = point_2007 + 2.5*slope
        
        predictions = np.array([pred_2008, pred_2012])

    ix = pd.Index(generate_year_list([2008, 2012]))
    return pd.Series(data=predictions, index=ix)


def write_submission_file(preds, filename):
    # load the submission labels
    file_format = pd.read_csv(os.path.join("/Users/fabianschreiber/Documents/projects/kaggle/united_nations/", "SubmissionRows.csv"), index_col=0)
    expected_row_count = file_format.shape[0]

    if isinstance(preds, pd.DataFrame):
        # check indices
        assert(preds.index == file_format.index).all(), \
            "DataFrame: Prediction indices must match submission format."
        
        # check columns
        assert (preds.columns == file_format.columns).all(), \
            "DataFrame: Column names must match submission format."
        
        final_predictions = preds
        
    elif isinstance(preds, np.ndarray):
        rows, cols = preds.shape
        
        if cols == 3:
            assert (preds[:,0] == file_format.index.values).all(), \
                "Numpy Array: First column must be indices."
            
            # now we know the indices are cool, ditch them
            preds = preds[:,1:]
        
        assert rows == expected_row_count, \
            "Numpy Array: The predictions must have the right number of rows."
        
        # put the predictions into the dataframe
        final_predictions = file_format.copy()
        final_predictions[generate_year_list([2008, 2012])] = preds
            
    elif isinstance(preds, list):
        assert len(preds) == 2, \
            "list: Predictions must be a list containing two lists"
        assert len(preds[0]) == expected_row_count, \
            "list: There must be the right number of predictions in the first list."
        assert len(preds[1]) == expected_row_count, \
            "list: There must be the right number of predictions in the second list."
    
        # write the predictions
        final_predictions = file_format.copy()
        final_predictions[generate_year_list(2008)] = np.array(preds[0], dtype=np.float64).reshape(-1, 1)
        final_predictions[generate_year_list(2012)] = np.array(preds[1], dtype=np.float64).reshape(-1, 1)
        
    elif isinstance(preds, dict):
        assert preds.keys() == generate_year_list([2008, 2012]), \
            "dict: keys must be properly formatted"
        assert len(preds[generate_year_list(2008)[0]]) == expected_row_count, \
            "dict: length of value for 2008 must match the number of predictions"
        assert len(preds[generate_year_list(2012)[0]]) == expected_row_count, \
            "dict: length of value for 2012 must match the number of predictions"
        
        # create dataframe from dictionary
        final_predictions = pd.DataFrame(preds, index=file_format.index)

    final_predictions.to_csv(filename)
 
