import pandas
import numpy as np
import sklearn.linear_model as lm
from sklearn.neighbors import NearestNeighbors

# The file contains a collection of functions for filling missing values in the dataset
# Each of the functions below accept housing data stored in DataFrame format,
# which is the format outputted by the read_csv function from pandas library 

# Remove all entries with any missing values
def remove_incomplete_entries(housing):
    housing.dropna(inplace=True)

# Fill all missing total_bedrooms values with the average of the existing total_bedrooms values
def fill_average(housing):
    notna = housing.total_bedrooms.notna()
    average = np.mean(housing.total_bedrooms.values[notna], axis=0)
    isna = housing.total_bedrooms.isna()
    housing.total_bedrooms.loc[isna] = average

# Fill all missing total_bedrooms values with linear regression prediction from other column's values
def fill_lr_prediction_from_other_column(housing, column):
    notna = housing.total_bedrooms.notna()
    model = lm.LinearRegression()
    model.fit(housing[column].values[notna].reshape(-1,1), housing['total_bedrooms'].values[notna].reshape(-1,1))
    isna = housing.total_bedrooms.isna()
    missing_bedrooms = model.predict(housing[column].values[isna].reshape(-1,1))
    housing.total_bedrooms.loc[isna] = np.squeeze(missing_bedrooms)

# Fill each row's missing total_bedrooms value with that row's k nearest neighbours' average total_bedrooms value
def fill_nn_prediction(housing, k):
    isna = housing.total_bedrooms.isna()
    all_rows_without_total_bedrooms = housing.drop(columns=['total_bedrooms'], inplace=False).values
    all_rows_without_total_bedrooms[isna] = np.nan_to_num(np.inf)
    isna_rows_without_total_bedrooms = housing.drop(columns=['total_bedrooms'], inplace=False).values[isna]
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(all_rows_without_total_bedrooms)
    knn_indices = nbrs.kneighbors(isna_rows_without_total_bedrooms, return_distance=False)
    knn_missing_bedrooms = np.array([housing.total_bedrooms.values[e] for e in knn_indices])
    knn_average_missing_bedrooms = np.mean(knn_missing_bedrooms, axis=1)
    housing.total_bedrooms.loc[isna] = knn_average_missing_bedrooms
