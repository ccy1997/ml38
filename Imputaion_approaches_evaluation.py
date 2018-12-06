import pandas
import numpy as np
import sklearn.linear_model as lm
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
import random
import Impute

# Transform ocean_proximity column to 5 binary-valued columns (from Iain's Jupyter)
def transform_ocean_proximity(housing):
    housing['1h_ocean'] = [1 if i=='<1H OCEAN' else 0 for i in housing.ocean_proximity.values]
    housing['island'] = [1 if i=='ISLAND' else 0 for i in housing.ocean_proximity.values]
    housing['inland'] = [1 if i=='INLAND' else 0 for i in housing.ocean_proximity.values]
    housing['near_ocean'] = [1 if i=='NEAR OCEAN' else 0 for i in housing.ocean_proximity.values]
    housing['near_bay'] = [1 if i=='NEAR BAY' else 0 for i in housing.ocean_proximity.values]
    housing.drop(columns=['ocean_proximity'], inplace=True)

def read_housing_data():
    housing = pandas.read_csv("./housing.csv")
    transform_ocean_proximity(housing)
    return housing

def lr_evaluate(housing):
    y = housing.median_house_value.values.reshape(-1,1)
    X = housing.drop(columns=['median_house_value'], inplace=False).values
    
    # Pull out 1000 values into a holdout set
    holdout = random.sample(range(0,10640),1000)
    X_holdout = X[holdout]
    y_holdout = y[holdout]
    Xt = np.delete(X, holdout, 0)
    yt = np.delete(y, holdout, 0)

    Model = lm.LinearRegression()
    kf = KFold(n_splits=5, shuffle=True)
    total_train_err = 0
    total_test_err = 0
    for train_index, test_index in kf.split(Xt):
        X_train, X_test = Xt[train_index], Xt[test_index]
        y_train, y_test = yt[train_index], yt[test_index]
        Model.fit(X_train, y_train)
        total_train_err += Model.score(X_train, y_train)
        total_test_err += Model.score(X_test, y_test)
    avg_train_err = total_train_err / 5
    avg_test_err = total_test_err / 5
    print("Average training r2 score: " + str(avg_train_err))
    print("Average validation r2 score: " + str(avg_test_err))
    print("Average of training and validation r2 score: " + str((avg_train_err + avg_test_err) / 2))

#If __name__ so I can import it in SVM
if __name__ == "__main__":
    housing1 = read_housing_data()
    housing2 = read_housing_data()
    housing3 = read_housing_data()
    housing4 = read_housing_data()

    Impute.remove_incomplete_entries(housing1)
    print("Evaluation of remove incomplete entries approach")
    lr_evaluate(housing1)

    Impute.fill_average(housing2)
    print("Evaluation of fill with average approach")
    lr_evaluate(housing2)

    Impute.fill_lr_prediction_from_other_column(housing3, 'total_rooms')
    print("Evaluation of linear regression approach")
    lr_evaluate(housing3)

    Impute.fill_nn_prediction(housing4, 4)
    print("Evaluation of nearest neighbours approach")
    lr_evaluate(housing4)
