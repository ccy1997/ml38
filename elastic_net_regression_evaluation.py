import pandas
import numpy as np
import sklearn.linear_model as lm
import random
import Impute
import ElasticNetRegression

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

def train_test_split(housing):
    y = housing.median_house_value.values.reshape(-1,1)
    X = housing.drop(columns=['median_house_value'], inplace=False).values
    # Pull out 1000 values into a holdout set
    holdout = random.sample(range(0,10640),1000)
    X_holdout = X[holdout]
    y_holdout = y[holdout]
    Xt = np.delete(X, holdout, 0)
    yt = np.delete(y, holdout, 0)
    return Xt, yt, X_holdout, y_holdout

def elastic_net_evaluate(housing):
    Xt, yt, X_holdout, y_holdout = train_test_split(housing)
    
    # Test diffferent regularization parameters
    for a in np.arange(0,0.1,0.01):
        print("a = " + str(a))
        Model = ElasticNetRegression.getTrainedModel(Xt, yt, a)
        print("Test r2 score: " + str(Model.score(X_holdout, y_holdout)) + "\n")

housing = read_housing_data()
Impute.fill_lr_prediction_from_other_column(housing, 'total_rooms')
print("Evaluation of ElasticNet Regression")
elastic_net_evaluate(housing)


