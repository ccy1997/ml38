# -*- coding: utf-8 -*-
"""
@author: Andrew Milton
"""

import pandas
import numpy as np
from sklearn.model_selection import KFold
import random
import sklearn.linear_model as lm
from sklearn import preprocessing
import DataPrepUtil
import Impute


housing = pandas.read_csv('housing.csv')

DataPrepUtil.transform_ocean_proximity(housing)
Impute.fill_lr_prediction_from_other_column(housing, 'total_rooms')

StandardScaler = preprocessing.StandardScaler()
MaxAbsScaler = preprocessing.MaxAbsScaler()
MinMaxScaler = preprocessing.MinMaxScaler()
RobustScaler = preprocessing.RobustScaler()

scaled_1 = pandas.DataFrame(preprocessing.StandardScaler().fit_transform(housing))
scaled_2 = pandas.DataFrame(preprocessing.MaxAbsScaler().fit_transform(housing))
scaled_3 = pandas.DataFrame(preprocessing.MinMaxScaler().fit_transform(housing))
scaled_4 = pandas.DataFrame(preprocessing.RobustScaler().fit_transform(housing))

scaled_1.columns = housing.columns
scaled_2.columns = housing.columns
scaled_3.columns = housing.columns
scaled_4.columns = housing.columns

scaler_names = ['StandardScaler', 'MaxAbsScaler', 'MinMaxScaler', 'RobustScaler'] 

scaler_count = 0

for i in (scaled_1, scaled_2, scaled_3, scaled_4): 
    
 
    housing = i
    
    y = housing.median_house_value.values.reshape(-1,1)
    X = housing.drop(columns=['median_house_value'], inplace=False).values
    
    holdout = random.sample(range(0,10640),1000)
    X_holdout = X[holdout]
    y_holdout = y[holdout]
    Xt = np.delete(X, holdout, 0)
    yt = np.delete(y, holdout, 0)
    
    fold_number = 10
    
    kf = KFold(n_splits=fold_number, shuffle=True)
    
    Model = lm.LinearRegression()

    train_R2_average = 0
    test_R2_average = 0
    mae_average = 0
    accuracy_average = 0
        
    for train_index, test_index in kf.split(Xt): 
        X_train, X_test = Xt[train_index], Xt[test_index]
        y_train, y_test = yt[train_index], yt[test_index]
        
        Model.fit(X_train, y_train.ravel()) 
        
        
        train_R2_average += Model.score(X_train, y_train)/fold_number
        test_R2_average += Model.score(X_test, y_test)/fold_number

        
    print('Average training R^2 score for ', scaler_names[scaler_count], ': ', train_R2_average)
    
    print('Average test R^2 scorefor ', scaler_names[scaler_count], ': ', test_R2_average, '\n')
    scaler_count += 1
    
housing = scaled_1
    
