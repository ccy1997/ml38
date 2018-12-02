import math
import time
import numpy
import pandas
import sklearn
from sklearn.utils import shuffle	
from sklearn.cross_validation import (train_test_split, ShuffleSplit, KFold)
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import (MinMaxScaler,LabelBinarizer)
from sklearn.decomposition import (FactorAnalysis, PCA, TruncatedSVD, FastICA)
from sklearn.manifold import Isomap
from sklearn.grid_search import (RandomizedSearchCV, GridSearchCV)
from sklearn.ensemble import RandomForestRegressor
from pandas.tools.plotting import scatter_matrix
from scipy.stats import (spearmanr, pearsonr)
from scipy.linalg import svd

import matplotlib.pyplot as plt

import dim_reduction


def handle_nan(dataset):
	return dataset.dropna()

def normalizer(dataset):
	scaler = MinMaxScaler() 
	scaled_values = scaler.fit_transform(dataset) 
	dataset_normalized = pandas.DataFrame(scaled_values)
	dataset_normalized.columns = dataset.columns
	return dataset_normalized

def one_hot_encoder(column):
	ocean_proximity_encoder = LabelBinarizer()
	encoded_class = ocean_proximity_encoder.fit_transform(column.values)
	encoded_class = pandas.DataFrame(data=encoded_class, columns=ocean_proximity_encoder.classes_)
	return encoded_class

def evaluate_model(model, X, Y):
	Y = Y.transpose().values
	predictions = rf.predict(X)
	errors = abs(predictions - Y)
	mae = numpy.mean(errors)
	mape = 100 * (errors / Y)
	accuracy = 100 - numpy.mean(mape)
	return mae, accuracy

def random_rf_tuning(X, Y):
	# Number of trees in random forest
	n_estimators = [int(x) for x in numpy.linspace(start = 200, stop = 2000, num = 10)]
	# Number of features to consider at every split
	max_features = ['auto', 'sqrt']
	# Maximum number of levels in tree
	max_depth = [int(x) for x in numpy.linspace(10, 110, num = 11)]
	max_depth.append(None)
	# Minimum number of samples required to split a node
	min_samples_split = [2, 5, 10]
	# Minimum number of samples required at each leaf node
	min_samples_leaf = [1, 2, 4]
	# Method of selecting samples for training each tree
	bootstrap = [True, False]

	# Create the random grid
	random_grid = {'n_estimators': n_estimators,
	               'max_features': max_features,
	               'max_depth': max_depth,
	               'min_samples_split': min_samples_split,
	               'min_samples_leaf': min_samples_leaf,
	               'bootstrap': bootstrap}

	rf = RandomForestRegressor()
	# Random search of parameters, using 3 fold cross validation, 
	# search across 100 different combinations, and use all available cores
	rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
	rf_random.fit(X, Y)
	print rf_random.best_params_

def grid_search_tuning(X, Y, param_grid):
	rf = RandomForestRegressor()
	grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
	grid_search.fit(X, Y)

if __name__ == "__main__":
	path = "./housing.csv"
	dataset = pandas.read_csv(path)
	
	ocean_proximity = one_hot_encoder(dataset[["ocean_proximity"]])

	dataset = dataset.drop('ocean_proximity', axis=1)
	dataset[ocean_proximity.columns] = ocean_proximity

	dataset = handle_nan(dataset)

	dataset = shuffle(dataset)
	Y = dataset[["median_house_value"]]
	X = dataset.drop(['median_house_value'], axis=1)

	#dataset.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
	#dataset.hist()
	# scatter_matrix(dataset)
	# plt.show()

	rf = RandomForestRegressor()

	time_start = time.clock()
	# X = X.drop(['total_rooms','total_bedrooms', 'households'], axis=1)
	# X = dim_reduction.low_variance_filter(X, normalizer, threshold=0.001)
	# X = dim_reduction.high_correlation_filter(X, threshold=0.94)
	# X = dim_reduction.factor_analysis_filter(X, normalizer, n_components=7, tolerance=1e-1, max_iteration=10000000)
	# X = dim_reduction.PCA_filter(X, normalizer, n_components=11)
	# X = dim_reduction.SVD_filter(X, normalizer, n_components=12)
	# X = dim_reduction.ICA_filter(X, normalizer, n_components=11, tolerance=1e-14, max_iteration=200000)
	# X = dim_reduction.ISOMAP_filter(X, normalizer, n_components=12)
	
	features = dim_reduction.forward_feature_selection(rf, X, Y, evaluate_model, kfold=10)
	# features = dim_reduction.backward_feature_selection(rf, X, Y, evaluate_model)

	print (time.clock() - time_start)

	X = X[features]
	# print features
	
	# random_rf_tuning(X, Y.values.ravel())
	# param_grid = {
 #    'bootstrap': [True],
 #    'max_depth': [90, 100, 110, 120],
 #    'max_features': ['sqrt'],
 #    'min_samples_leaf': [0, 1, 2, 3],
 #    'min_samples_split': [3, 5, 7],
 #    'n_estimators': [100, 200, 300, 1000]
	# }
	#grid_search_tuning(X, Y.values.ravel(), param_grid)

	# rs = ShuffleSplit(n=X.shape[0], test_size=0.25, n_iter=10)
	kf = KFold(X.shape[0], n_folds=10)

	# rf = RandomForestRegressor(n_estimators=400, min_samples_split=5, min_samples_leaf=1, max_features='sqrt', max_depth=100, bootstrap=True)
	# rf = RandomForestRegressor(n_estimators=100, min_samples_split=5, min_samples_leaf=1, max_features='sqrt', max_depth=90, bootstrap=True)
	# rf = RandomForestRegressor(n_estimators=2000, min_samples_split=2, min_samples_leaf=2, max_features='auto', max_depth=90, bootstrap=True)

	# X = X[['total_bedrooms', 'longitude', 'total_rooms', 'households', 'population', 'ISLAND', 'NEAR OCEAN', 'median_income', 'NEAR BAY', 'latitude']]

	for train_idx, test_idx in kf:
		X_train = X.iloc[train_idx,:]
		Y_train = Y.iloc[train_idx]
		X_test = X.iloc[test_idx,:]
		Y_test = Y.iloc[test_idx]

		rf.fit(X_train, Y_train.values.ravel())
		predictions = rf.predict(X_test)

		errors = abs(predictions - Y_test.transpose().values)
		print('Mean Absolute Error:', round(numpy.mean(errors), 2))
		mape = 100 * (errors / Y_test.transpose().values)
		accuracy = 100 - numpy.mean(mape)
		print('Accuracy:', round(accuracy, 2), '%.')

		# importances = list(rf.feature_importances_)

		# feature_list = list(X.columns)
		# feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

		# # Sort the feature importances by most important first
		# feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

		# print feature_importances