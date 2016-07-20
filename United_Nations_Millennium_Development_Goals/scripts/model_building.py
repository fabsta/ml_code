# -*- coding: utf-8 -*-
import helper_functions

in_file = '/Users/fabianschreiber/Documents/projects/ml_code/United_Nations_Millennium_Development_Goals/data/TrainingSet.csv'
submission_file = '/Users/fabianschreiber/Documents/projects/ml_code/United_Nations_Millennium_Development_Goals/data/SubmissionRows.csv'
out_filename = '/Users/fabianschreiber/Documents/projects/ml_code/United_Nations_Millennium_Development_Goals/submissions/next_submission.csv'
print("Loading data...\n")
training_data = pd.read_csv(in_file, index_col=0)
submission_data = pd.read_csv(submission_file, index_col=0)

# joinging files
prediction_rows = training_data.loc[submission_data.index]
#prediction_rows.loc[1030]
training_transposed = prediction_rows[generate_year_list(1972,2007)].transpose()

#columns = ['ID','2008 [YR2008]','2012 [YR2012]']
#indices= ['2008 [YR2008]','2012 [YR2012]']
#results = pd.DataFrame(index = indices)
results = pd.DataFrame()

for column in training_transposed:
    #print(training_transposed[column])
    print "looking at column ",column
    pred_2008, pred_2012 = helper_functions.predict_for_category(training_transposed,column)
    #pred_2008, pred_2012 = helper_functions.predict_for_category(training_transposed,training_transposed.columns[0])
    
    results[column] = pd.Series([pred_2008,pred_2012], np.arange(1, 3))
    #results[training_transposed.columns[0]] = pd.Series([pred_2008,pred_2012])


results.transpose().to_csv(out_filename)   



#category = "Combat HIV/AIDS"
#country = "Zimbabwe"
## test correlations data
#get_correlations(training_data,country, category)
#
#
#
#kenya_data = training_data[training_data["Country_Name"] == country]
#kenya_values = kenya_data[generate_year_list(1972, 2007)].values
#
#only_years = kenya_data[generate_year_list(1972, 2007)]
#only_years.plot()
## get the total number of time series we have for Kenya
#nseries = kenya_values.shape[0]
#
## -1 as default
## returns a new array filled with ones
#lag_corr_mat = np.ones([nseries, nseries], dtype=np.float64)*-1
#
## create a matrix to hold our lagged correlations
#for i in range(nseries):
#    for j in range(nseries):
#        # skip comparing a series with itself
#        if i!=j:
#            # get original (1972-2006) and shifted (1973-2007)
#            original = kenya_values[i,1:]
#            shifted = kenya_values[j,:-1]
#            
#            # for just the indices where neither is nan
#            non_nan_mask = (~np.isnan(original) & ~np.isnan(shifted))
#            
#            # if we have at least 2 data points
#            if non_nan_mask.sum() >= 2:
#                lag_corr_mat[i,j] = np.correlate(original[non_nan_mask], shifted[non_nan_mask])
#                
#
#
##to_predict_ix = 285811 
#
## first, we get the index of that row in the correlation matrix
#i = np.where(kenya_data.index.values == to_predict_ix)[0][0]
#
## then, we see which value in the matrix is the largest for that row
#j_max = np.argmax(lag_corr_mat[i,:])
#
## finally, let's see what these correspond to
#max_corr_ix = kenya_data.index.values[j_max]
##kenya_data[generate_year_list(1972, 2007)].values
#max_corr_series = kenya_data[generate_year_list(1972, 2007)][max_corr_ix]
#slope = max_corr_series[-1] - max_corr_series[-2]
#
#training_data.loc[357]


