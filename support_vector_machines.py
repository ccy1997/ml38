import Impute
import pandas
import math
import DataPrepUtil
import random
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold


FEATURES = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','1h_ocean','island','inland','near_ocean','near_bay']


def load_data(incomplete = 0):
    # I have put read_housing_data function in a separate file so changed a bit on naming
    housing_data = pandas.read_csv('./housing.csv')
    DataPrepUtil.transform_ocean_proximity(housing_data)
    
    if incomplete == 0:
        Impute.remove_incomplete_entries(housing_data)
    elif incomplete == 1:
        Impute.fill_average(housing_data)
    elif incomplete == 2:
        Impute.fill_lr_prediction_from_other_column(housing_data,'total_rooms')
    elif incomplete == 3:
        Impute.fill_nn_prediction(housing_data,3)
    return housing_data

def k_fold_model(X,Y,model,eval_x,eval_y):
    folder = KFold(n_splits=5, shuffle=True)
    r2_scores = []
    mae_scores = []
    mse_scores = []
    X = np.array(X)
    Y = np.array(Y)
    for train_index, test_index in folder.split(X):
        x_train,x_test = X[train_index],X[test_index]
        y_train,y_test = Y[train_index],Y[test_index]
        model.fit(x_train,y_train)
        r2_scores += [model.score(x_train,y_train),model.score(x_test,y_test)]
        preds = model.predict(x_test)
        mae_scores += [np.mean(abs(preds - y_test))]
        mse_scores += [np.mean(np.power((preds - y_test),2))]
    print("[training r2, test r2]")
    for i in range(len(r2_scores)):
        print(str(i) + " : "+str(r2_scores[i]))
        
    print("test MAE, test MSE")
    for i in range(len(mae_scores)):
        print(str(i)+" : "+str(mae_scores[i])+","+str(mse_scores[i]))
        
    print("Eval r2, Eval MSA, Eval MSE")
    preds = model.predict(x_test)
    print(model.score(eval_x,eval_y),", ",np.mean(abs(preds - y_test)), np.mean(np.power((preds - y_test),2)))
    
def create_kfold_sets(data,eval_size=2000):
    #Split into cross validation set and evaluation set
    Y = data['median_house_value'].values
    X = data[FEATURES].values
    
    random_indexes = random.sample(range(0,len(X)),2000)
    
    eval_set = (X[random_indexes],Y[random_indexes])
    cross_set = ([X[i] for i in range(0,len(X))],[Y[i] for i in range(0,len(Y))])
    
    return cross_set, eval_set

def run_svr(cross_set,eval_set,c=10**5):
    regressor = svm.SVR(gamma='scale',C=c)
    k_fold_model(cross_set[0],cross_set[1],regressor,eval_set[0],eval_set[1])

if __name__ == "__main__":
    housing = load_data()
    cross_set, eval_set = create_kfold_sets(housing)
    
    print("C = 6")
    run_svr(cross_set, eval_set,c=10**6)
    print("C = 7")
    run_svr(cross_set, eval_set,c=10**7)
