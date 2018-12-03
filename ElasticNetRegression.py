import pandas
import numpy as np
import sklearn.linear_model as lm
from sklearn.model_selection import KFold

# Produce a trained regression model using ElasticNet regularization, model's r2 scores are printed when training
def getTrainedModel(Xt, yt, a):
    Model = lm.ElasticNet(alpha=a)
    
    # Have to shuffle the data because it is grouped.
    kf = KFold(n_splits=5, shuffle=True)
    total_train_err = 0
    total_valid_err = 0
    fold_ct = 1
    for train_index, valid_index in kf.split(Xt):
        X_train, X_valid = Xt[train_index], Xt[valid_index]
        y_train, y_valid = yt[train_index], yt[valid_index]
        Model.fit(X_train, y_train)
        total_train_err += Model.score(X_train, y_train)
        total_valid_err += Model.score(X_valid, y_valid)
        print(str(fold_ct) + "th fold " + "training r2 score: " + str(Model.score(X_train, y_train)))
        print(str(fold_ct) + "th fold " + "validation r2 score: " + str(Model.score(X_valid, y_valid)))
        fold_ct += 1
    avg_train_err = total_train_err / 5
    avg_valid_err = total_valid_err / 5
    print("Average training r2 score: " + str(avg_train_err))
    print("Average validation r2 score: " + str(avg_valid_err))
    print("Average of training and validation r2 score: " + str((avg_train_err + avg_valid_err) / 2))
    return Model
