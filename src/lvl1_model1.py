from joblib.parallel import _verbosity_filter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.pipeline import Pipeline, _name_estimators, make_pipeline
import warnings
warnings.filterwarnings('ignore')

from xgboost.training import train; 
from time import time
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier

np.random.seed(420)

skfold = StratifiedKFold(
    n_splits=6, 
    shuffle=True
)

train_df = pd.read_csv('../dataset/features_train/lvl1_features.csv')
y_train = np.load('../dataset/features_train/emotions1.npy')
test_df = pd.read_csv('../dataset/features_test/lvl1_featurestest.csv')
test_files = np.load('../dataset/features_test/testfiles1.npy')

model_nums = [1,2,3,6,7,8,9,10,11,12,13,14]
train_feature_cols = []
train_pred_cols = []

for i in model_nums:
    for j in range(1, 8):
        train_feature_cols.append("m"+str(i)+"_pred_"+str(j))
    train_pred_cols.append("m"+str(i)+"_predictions")

X_train = train_df[train_feature_cols]
X_train_pred = train_df[train_pred_cols]

X_test = test_df[train_feature_cols]
X_test_pred = test_df[train_pred_cols]

for pred_no in range(1,8):
    X_train["avg_"+str(pred_no)] = np.mean(train_df[["m"+str(i)+"_pred_"+str(pred_no) for i in model_nums]].values, axis=1)

print(X_train.head())

base_estimator = XGBClassifier(
    n_estimators = 200,
    learning_rate = 0.1,
    max_depth = 20,
    min_child_weight = 1,
    gamma = 0.2,
    colsample_bytree = 0.7,
    subsample = 0.7,
    n_jobs = 1
)
lvl1_model1 = BaggingClassifier(
    base_estimator=base_estimator,
    n_estimators= 100,
    max_features=0.4,
    bootstrap_features=True,
    verbose=3
)

# param_grid = {
#     'base_estimator__max_depth' : [3, 5, 11, 15, 20, 30, 40],
#     'base_estimator__min_child_weight' : [1, 2, 3, 5, 7],
#     'base_estimator__gamma' : [0, 0.1, 0.2, 0.3, 0.4],
#     'base_estimator__subsample' : [0.4, 0.5, 0.6, 0.7, 0.8],
#     'base_estimator__colsample_bytree' : [0.4, 0.5, 0.6, 0.7, 0.8],
# }
              

# grid = RandomizedSearchCV(estimator=lvl1_model1, 
#                             param_distributions=param_grid, 
#                             n_iter = 50, scoring='accuracy', 
#                             cv = skfold, 
#                             verbose=3,
#                             n_jobs=4
# )

# # grid = GridSearchCV(estimator=lvl1_model1,
# #                         param_grid=param_grid,
# #                         cv = skfold,
# #                         scoring= 'accuracy',
# #                         verbose=5,
# #                         n_jobs=1
# # )

# grid.fit(X_train, y_train)
# best_score = grid.best_score_
# best_params = grid.best_params_
# print('Best Accuracy : {:.3f}%'.format(best_score))
# print('Best Parameters : ',best_params)

# num_classes = pd.Series(y_train).nunique()
# ypredtrain = np.empty((len(y_train)), dtype=np.dtype('<U8'))
# ytrainprobs = np.empty((len(y_train), num_classes))
# # ypredtest = np.empty((len(X_test)), dtype= np.dtype('<U8'))
# # ytestprobs = np.empty((len(X_test), num_classes))

# print(type(X_train), type(y_train))

# scores = []
# i = 1
# for train_indices, test_indices in skfold.split(X_train, y_train):
#     print("entered for")
    
#     # fit model to training fold
#     trainX, testX = X_train.loc[train_indices], X_train.loc[test_indices]
#     trainy, testy = y_train[train_indices], y_train[test_indices]
#     print("done splitting")

#     scaler = StandardScaler()
#     scaler.fit_transform(trainX)
#     scaler.transform(testX)

#     start = time()
#     lvl1_model1.fit(trainX, trainy)
#     # print(model14.evals_result_['valid_0']['multi_logloss'])
#     # loss_lists.append(lvl1_model1.evals_result_['valid_0']['multi_logloss'])
#     end = time()
#     # get predictions
#     ypredtrain[test_indices] = lvl1_model1.predict(testX)
#     ytrainprobs[test_indices] = lvl1_model1.predict_proba(testX)
#     # score the model on validation fold
#     scores.append(lvl1_model1.score(testX, testy))
#     print('Score on Fold {} : {:.3f}%, Time taken = {:.3f}s'.format(i, scores[-1]*100, end-start))
#     i += 1
# print(f'\nMean score of KFold CV for {type(lvl1_model1).__name__}: {100*np.mean(scores):.3f}% ± {100*np.std(scores):.3f}%')
