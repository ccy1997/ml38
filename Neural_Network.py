import pandas
import numpy as np
from sklearn import preprocessing
from scipy import stats
from sklearn.model_selection import KFold
import random
import DataPrepUtil
import Impute
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV


housing = pandas.read_csv('housing.csv')


DataPrepUtil.transform_ocean_proximity(housing)
Impute.fill_lr_prediction_from_other_column(housing, 'total_rooms')

# Normalisation of data (feature scaling)
scaled = pandas.DataFrame(preprocessing.StandardScaler().fit_transform(housing))
scaled.columns = housing.columns
housing = scaled

median_house_value_bc, maxlog, interval = stats.boxcox(housing.median_house_value, alpha=0.05)
population_bc, maxlog, interval = stats.boxcox(housing.population, alpha=0.05)
housing_median_age_bc, maxlog, interval = stats.boxcox(housing.housing_median_age, alpha=0.05)
total_rooms_bc, maxlog, interval = stats.boxcox(housing.total_rooms, alpha=0.05)
total_bedrooms_bc, maxlog, interval = stats.boxcox(housing.total_bedrooms, alpha=0.05)
households_bc, maxlog, interval = stats.boxcox(housing.households, alpha=0.05)
median_income_bc, maxlog, interval = stats.boxcox(housing.median_income, alpha=0.05)

housing_boxcox = housing.copy()

housing_boxcox.drop(columns=['housing_median_age'], inplace=True)
housing_boxcox.drop(columns=['total_rooms'], inplace=True)
housing_boxcox.drop(columns=['total_bedrooms'], inplace=True)
housing_boxcox.drop(columns=['population'], inplace=True)
housing_boxcox.drop(columns=['households'], inplace=True)
housing_boxcox.drop(columns=['median_income'], inplace=True)
housing_boxcox.drop(columns=['median_house_value'], inplace=True)

housing_boxcox['housing_median_age'] = housing_median_age_bc
housing_boxcox['total_rooms'] = total_rooms_bc
housing_boxcox['total_bedrooms'] = total_bedrooms_bc
housing_boxcox['population'] = population_bc
housing_boxcox['households'] = households_bc
housing_boxcox['median_income'] = median_income_bc
housing_boxcox['median_house_value'] = median_house_value_bc

y = housing_boxcox.median_house_value.values.reshape(-1,1)
X = housing_boxcox.drop(columns=['median_house_value'], inplace=False).values

print(X.shape)
print(y.shape)
# Pull out 1000 values into a holdout set
holdout = random.sample(range(0,10640),1000)
X_holdout = X[holdout]
y_holdout = y[holdout]
Xt = np.delete(X, holdout, 0)
yt = np.delete(y, holdout, 0)
print(Xt.shape)
print(yt.shape)

train_R2_average = 0
test_R2_average = 0
mae_average = 0
accuracy_average = 0

fold_number = 10

kf = KFold(n_splits=fold_number, shuffle=True)


Model = MLPRegressor(hidden_layer_sizes=(20,),
                                       activation='relu',
                                       solver='adam',
                                       learning_rate='adaptive',
                                       max_iter=500,
                                       learning_rate_init=0.005,
                                       early_stopping=True,
                                       validation_fraction=0.1)

# Finding optimal regularisation parameter (alpha)

alphas = np.array([1, 0.1, 0.01, 0.001, 0])

fold_counter = 1

for train_index, test_index in kf.split(Xt): 
    X_train, X_test = Xt[train_index], Xt[test_index]
    y_train, y_test = yt[train_index], yt[test_index]
      
    
    grid = GridSearchCV(estimator=Model, param_grid=dict(alpha=alphas), cv=3)
    
    grid.fit(X_train, y_train.ravel()) 
    
    train_R2_average += grid.score(X_train, y_train)/fold_number
    test_R2_average += grid.score(X_test, y_test)/fold_number
    
    print('fold ', fold_counter, ' of ', fold_number, ' completed')
    fold_counter += 1

print('Mean CV score of best alpha: ', grid.best_score_)
print('Best alpha value: ', grid.best_estimator_.alpha)
print('Average training R^2 score: ', train_R2_average)
print('Average test R^2 score: ', test_R2_average)