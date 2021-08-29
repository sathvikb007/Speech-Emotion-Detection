from joblib.parallel import _verbosity_filter
import joblib
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
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier

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
    X_test["avg_"+str(pred_no)] = np.mean(test_df[["m"+str(i)+"_pred_"+str(pred_no) for i in model_nums]].values, axis=1)

# print(X_train.head())
# print(X_test.head())

base_estimator = ExtraTreesClassifier(
    n_estimators = 100,
)
adaboost = AdaBoostClassifier(
    base_estimator = base_estimator,
    n_estimators = 10
)
lvl1_model2 = BaggingClassifier(
    base_estimator=adaboost,
    n_estimators= 10,
    max_features=0.5,
    bootstrap_features=True,
    verbose=3,
    n_jobs=-1
)

print(lvl1_model2.get_params)

param_grid = {
    'max_features': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'base_estimator__base_estimator__max_depth' : [5, 10, 15, 20, 25, 30, 35, 40, 45],
    'base_estimator__base_estimator__min_samples_split': [2, 3, 5, 7, 9],
    'base_estimator__base_estimator__min_samples_leaf': [1,2,3,4],
    'base_estimator__base_estimator__max_features': [None, 'sqrt'],
    'base_estimator__learning_rate': [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
}

# Best Accuracy : 0.596%
# Best Parameters :  {'base_estimator__min_samples_split': 2, 'base_estimator__min_samples_leaf': 1, 'base_estimator__max_features': 0.5, 'base_estimator__max_depth': 40}           

# Best Accuracy : 0.597%
# Best Parameters :  {'max_features': 0.5, 'base_estimator__min_samples_split': 2, 'base_estimator__min_samples_leaf': 2, 'base_estimator__max_features': None, 'base_estimator__max_depth': 40}

# base_estimator__base_estimator__max_depth=45, base_estimator__base_estimator__max_features=sqrt, base_estimator__base_estimator__min_samples_leaf=2, base_estimator__base_estimator__min_samples_split=7, base_estimator__learning_rate=0.7, max_features=0.6

grid = RandomizedSearchCV(estimator=lvl1_model2, 
                            param_distributions=param_grid, 
                            n_iter = 80, scoring='accuracy', 
                            cv = skfold, 
                            verbose=3,
                            n_jobs=4
)

# grid = GridSearchCV(estimator=lvl1_model2,
#                         param_grid=param_grid,
#                         cv = skfold,
#                         scoring= 'accuracy',
#                         verbose=5,
#                         n_jobs=-1
# )

grid.fit(X_train, y_train)
best_score = grid.best_score_
best_params = grid.best_params_
print('Best Accuracy : {:.3f}%'.format(best_score))
print('Best Parameters : ',best_params)
print('Best Estimator', grid.best_estimator_)
joblib.dump(grid.best_estimator_, 'lvl1_model2.pkl')

# base_estimator = ExtraTreesClassifier(
#     n_estimators = 300,
#     max_depth = 30,
#     min_samples_split=2,
#     min_samples_leaf=2
# )
# lvl1_model2 = BaggingClassifier(
#     base_estimator=base_estimator,
#     n_estimators= 50,
#     max_features=0.5,
#     bootstrap_features=True,
#     verbose=3,
#     n_jobs=2
# )

# num_classes = pd.Series(y_train).nunique()
# ypredtrain = np.empty((len(y_train)), dtype=np.dtype('<U8'))
# ytrainprobs = np.empty((len(y_train), num_classes))
# # ypredtest = np.empty((len(X_test)), dtype= np.dtype('<U8'))
# # ytestprobs = np.empty((len(X_test), num_classes))

# print(type(X_train), type(y_train))

# scores = []
# i = 1
# for train_indices, test_indices in skfold.split(X_train, y_train):
    
#     # fit model to training fold
#     trainX, testX = X_train.loc[train_indices], X_train.loc[test_indices]
#     trainy, testy = y_train[train_indices], y_train[test_indices]
#     # print("done splitting")

#     scaler = RobustScaler()
#     scaler.fit_transform(trainX)
#     scaler.transform(testX)

#     start = time()
#     lvl1_model2.fit(trainX, trainy)
#     end = time()
#     # get predictions
#     ypredtrain[test_indices] = lvl1_model2.predict(testX)
#     ytrainprobs[test_indices] = lvl1_model2.predict_proba(testX)
#     # score the model on validation fold
#     scores.append(lvl1_model2.score(testX, testy))
#     print('Score on Fold {} : {:.3f}%, Time taken = {:.3f}s'.format(i, scores[-1]*100, end-start))
#     i += 1
# print(f'\nMean score of KFold CV for {type(lvl1_model2).__name__}: {100*np.mean(scores):.3f}% ± {100*np.std(scores):.3f}%')

# print('\nTraining on train data and predicting for test...')
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# lvl1_model2.fit(X_train, y_train)
# print('Finished Training')
# ypredtest = lvl1_model2.predict(X_test)
# ytestprobs = lvl1_model2.predict_proba(X_test)


# train_df = pd.DataFrame(data = ytrainprobs, columns = ["l1_m1_pred_"+str(i) for i in range(1, num_classes+1)])
# train_df['l1_m1_predictions'] = ypredtrain

# test_df = pd.DataFrame(data=ytestprobs, columns = ["l1_m1_pred_"+str(i) for i in range(1, num_classes+1)])
# test_df['l1_m1_predictions'] = ypredtest
# test_df['filename'] = test_files

# train_df.to_csv('../predictions/train_predictions/lvl1_model2.csv', index = None)
# test_df.to_csv('../predictions/test_predictions/lvl1_model2.csv', index = None)
