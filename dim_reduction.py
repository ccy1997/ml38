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

#whithdraw any feature having a variance below the threshold
#take normalizer function as argument
#return the dataset
def low_variance_filter(dataset, normalizer, threshold=0):
	dataset_normalized = normalizer(dataset)
	print "Low variance filter"
	thresholder = VarianceThreshold(threshold)
	thresholder.fit(dataset_normalized)
	dataset_high_variance = dataset.loc[:,dataset.columns[thresholder.get_support(indices=True)]]
	print "variances : "
	print thresholder.variances_
	print "caught by the filter : ",
	print [x for x in dataset.columns if x not in dataset_high_variance.columns]
	return dataset_high_variance

#whitdraw any feature with correlation value with another feature above the threshold
#return the dataset
def high_correlation_filter(dataset, threshold=1):
	print "High correlation filter"
	dataset_low_correlation = pandas.DataFrame()
	pearson_matrix = []
	spearman_matrix = []

	for i,entry_1 in enumerate(dataset.columns):
		pearson_matrix.append([])
		spearman_matrix.append([])
		for j,entry_2 in enumerate(dataset.columns):
			pearson_matrix[i].append(pearsonr(dataset[entry_1], dataset[entry_2])[0])
			spearman_matrix[i].append(spearmanr(dataset[entry_1], dataset[entry_2])[0])

	for i in range(len(dataset.columns)):
		dropped = False
		for j in range(i):
			if(abs(pearson_matrix[i][j]) > threshold or abs(spearman_matrix[i][j]) > threshold):
				print dataset.columns[i]+" correlated with "+dataset.columns[j]
				dropped = True
		if(not dropped):
			dataset_low_correlation[dataset.columns[i]] = dataset[dataset.columns[i]]

	return dataset_low_correlation

#build a training dataset, adding one feature at a time, stop when performance decrease
#performance is evaluate with kfold X-validation and an evaluation function (k and the function are passed in arguments)
#return a list of feature's name
def forward_feature_selection(model, X, Y, evaluation_function, kfold=3):
	kf = KFold(X.shape[0], n_folds=kfold)
	available_features = list(X.columns.values)
	selected_features = []
	best_accuracy = 0
	prediction_improved = True
	while(prediction_improved is True and len(available_features)!=0):
		prediction_improved = False
		for feature in available_features:
			accuracy = 0
			selected_features.append(feature)
			for train_idx, test_idx in kf:
				X_train = X.iloc[train_idx,:]
				X_train = X_train[selected_features]
				Y_train = Y.iloc[train_idx]
				X_test = X.iloc[test_idx,:]
				X_test = X_test[selected_features]
				Y_test = Y.iloc[test_idx]
				model.fit(X_train, Y_train.values.ravel())
				mae, fold_accuracy = evaluation_function(model, X_test, Y_test)
				accuracy += fold_accuracy
			accuracy = accuracy / kfold
			if(accuracy > best_accuracy):
				prediction_improved = True
				best_accuracy = accuracy
				best_feature = feature
				print "best_accuracy ", best_accuracy
				print "best_feature", best_feature
			selected_features.remove(feature)
		if(prediction_improved is True):
			selected_features.append(best_feature)
			available_features.remove(best_feature)

	return selected_features


#build a training dataset, removing one feature at a time, stop when performance decrease
#performance is evaluate with kfold X-validation and an evaluation function (k and the function are passed in arguments)
#return a list of feature's name
def backward_feature_selection(model, X, Y, evaluation_function, kfold=3):
	kf = KFold(X.shape[0], n_folds=kfold)
	# available_features = list(X.columns.values)
	kept_features = list(X.columns.values)
	best_accuracy = 0
	prediction_improved = True
	while(prediction_improved is True and len(kept_features)!=0):
		prediction_improved = False
		for feature in kept_features:
			accuracy = 0
			kept_features.remove(feature)
			for train_idx, test_idx in kf:
				X_train = X.iloc[train_idx,:]
				X_train = X_train[kept_features]
				Y_train = Y.iloc[train_idx]
				X_test = X.iloc[test_idx,:]
				X_test = X_test[kept_features]
				Y_test = Y.iloc[test_idx]
				model.fit(X_train, Y_train.values.ravel())
				mae, fold_accuracy = evaluation_function(model, X_test, Y_test)
				accuracy += fold_accuracy
			accuracy = accuracy / kfold
			if(accuracy > best_accuracy):
				prediction_improved = True
				best_accuracy = accuracy
				worst_feature = feature
				print "best_accuracy ", best_accuracy
				print "worst_feature", worst_feature
			kept_features.append(feature)
		if(prediction_improved is True):
			kept_features.remove(worst_feature)

	return kept_features


#build a training dataset, removing one feature at a time, stop when performance decrease
#performance is evaluate with kfold X-validation and an evaluation function (k and the function are passed in arguments)
#here the feature importances list of the model is used to remove a feature
#return a list of feature's name
def backward_feature_selection2(model, X, Y, kfold=3):
	kf = KFold(X.shape[0], n_folds=kfold)
	kept_features = list(X.columns.values)
	best_accuracy = 0
	prediction_improved = True
	while(prediction_improved is True and len(kept_features)!=0):
		prediction_improved = False
		accuracy = 0
		print kept_features
		for train_idx, test_idx in kf:
			X_train = X.iloc[train_idx,:]
			X_train = X_train[kept_features]
			Y_train = Y.iloc[train_idx]
			X_test = X.iloc[test_idx,:]
			X_test = X_test[kept_features]
			Y_test = Y.iloc[test_idx]
			model.fit(X_train, Y_train.values.ravel())
			mae, fold_accuracy = evaluation_function(model, X_test, Y_test)
			accuracy += fold_accuracy
		accuracy = accuracy / kfold
		importances = list(model.feature_importances_)
		feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(kept_features, importances)]
		feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
		print feature_importances
		if(accuracy > best_accuracy):
			prediction_improved = True
			best_accuracy = accuracy
			worst_feature = feature_importances[-1][0]
			print "best_accuracy ", best_accuracy
			print "worst_feature", worst_feature
			kept_features.remove(worst_feature)


	return kept_features

#apply factor analysis
#take normalizer function as argument
#return the transformed dataset
def factor_analysis_filter(dataset, normalizer, n_components=None, tolerance=0.01, max_iteration=1000):
	dataset_normalized = normalizer(dataset)
	transformer = FactorAnalysis(n_components=n_components, tol=tolerance, max_iter=max_iteration)
	factor = transformer.fit(dataset_normalized)
	df_factors = pandas.DataFrame(factor.components_,columns=dataset_normalized.columns)
	dataset_transformed = factor.transform(dataset_normalized)
	dataset_transformed = pandas.DataFrame(dataset_transformed)
	return dataset_transformed

#apply principal component analysis
#take normalizer function as argument
#return the transformed dataset
def PCA_filter(dataset, normalizer, n_components=None):
	dataset_normalized = normalizer(dataset)
	transformer = PCA(n_components=n_components)
	factor = transformer.fit(dataset_normalized)
	df_factors = pandas.DataFrame(factor.components_,columns=dataset_normalized.columns)
	dataset_transformed = factor.transform(dataset_normalized)
	dataset_transformed = pandas.DataFrame(dataset_transformed)
	print transformer.explained_variance_ratio_ 
	return dataset_transformed

#apply singular value decomposition
#take normalizer function as argument
#return the transformed dataset
def SVD_filter(dataset, normalizer, n_components=2):
	dataset_normalized = normalizer(dataset)
	transformer = TruncatedSVD(n_components=n_components)
	factor = transformer.fit(dataset_normalized)
	df_factors = pandas.DataFrame(factor.components_,columns=dataset_normalized.columns)
	dataset_transformed = factor.transform(dataset_normalized)
	dataset_transformed = pandas.DataFrame(dataset_transformed)
	print transformer.explained_variance_ratio_ 
	return dataset_transformed

#apply independant component analysis
#take normalizer function as argument
#return the transformed dataset
def ICA_filter(dataset, normalizer, n_components=None, tolerance=0.0001, max_iteration=200):
	dataset_normalized = normalizer(dataset)
	transformer = FastICA(n_components=n_components, tol=tolerance, max_iter=max_iteration)
	factor = transformer.fit(dataset_normalized)
	df_factors = pandas.DataFrame(factor.components_,columns=dataset_normalized.columns)
	dataset_transformed = factor.transform(dataset_normalized)
	dataset_transformed = pandas.DataFrame(dataset_transformed)
	return dataset_transformed

#apply isomap
#take normalizer function as argument
#return the transformed dataset
def ISOMAP_filter(dataset, normalizer, n_components=2, n_neighbors=5):
	dataset_normalized = normalizer(dataset)
	transformer = Isomap(n_components=n_components, n_neighbors=n_neighbors)
	factor = transformer.fit(dataset_normalized)
	df_factors = pandas.DataFrame(factor.components_,columns=dataset_normalized.columns)
	dataset_transformed = factor.transform(dataset_normalized)
	dataset_transformed = pandas.DataFrame(dataset_transformed)
	return dataset_transformed

