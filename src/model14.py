import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.pipeline import Pipeline, make_pipeline
import warnings

from xgboost.callback import early_stop; 
warnings.filterwarnings('ignore')
from time import time
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

np.random.seed(420)

X_train = np.load('../dataset/features_train/features2.npy')
y_train = np.load('../dataset/features_train/emotions2.npy')
X_test = np.load('../dataset/features_test/features2test.npy')
test_files = np.load('../dataset/features_test/testfiles2.npy')


skfold = StratifiedKFold(
    n_splits=6, 
    shuffle=True
)

# model = LGBMClassifier(
#     boosting_type='goss',
#     n_estimators=110,
#     learning_rate=0.05,
#     max_depth  = 20,
#     num_leaves=90,
#     subsample=0.4,
#     colsample_bytree=0.7,
#     n_jobs=-1
# )

# param_grid = {
#     'num_leaves': [90, 128, 256, 512]
# }
              

# # grid = RandomizedSearchCV(estimator=model14, 
# #                             param_distributions=param_grid, 
# #                             n_iter = 50, scoring='accuracy', 
# #                             cv = skfold, 
# #                             verbose=3
# # )

# grid = GridSearchCV(estimator=model,
#                         param_grid=param_grid,
#                         cv = skfold,
#                         scoring= 'accuracy',
#                         verbose=5,
#                         n_jobs=-1
# )

# grid.fit(X_train, y_train, verbose = True)
# best_score = grid.best_score_
# best_params = grid.best_params_
# print('Best Accuracy : {:.3f}%'.format(best_score))
# print('Best Parameters : ',best_params)
# print('Best Estimator : {}'.format(grid.best_estimator_))



model14 = LGBMClassifier(
    boosting_type='goss',
    n_estimators=359,
    learning_rate=0.01,
    max_depth  = 20,
    num_leaves=90,
    subsample=0.4,
    colsample_bytree=0.7,
    n_jobs=-1
)
loss_lists = []
num_classes = pd.Series(y_train).nunique()
ypredtrain = np.empty((len(y_train)), dtype=np.dtype('<U8'))
ytrainprobs = np.empty((len(y_train), num_classes))
ypredtest = np.empty((len(X_test)), dtype= np.dtype('<U8'))
ytestprobs = np.empty((len(X_test), num_classes))

scores = []
i = 1
for train_indices, test_indices in skfold.split(X_train, y_train):
    
    # fit model to training fold
    trainX, testX = X_train[train_indices], X_train[test_indices]
    trainy, testy = y_train[train_indices], y_train[test_indices]

    scaler = StandardScaler()
    scaler.fit_transform(trainX)
    scaler.transform(testX)

    start = time()
    model14.fit(trainX, trainy, eval_set=[(testX, testy)])
    # print(model14.evals_result_['valid_0']['multi_logloss'])
    loss_lists.append(model14.evals_result_['valid_0']['multi_logloss'])
    end = time()
    # get predictions
    ypredtrain[test_indices] = model14.predict(testX)
    ytrainprobs[test_indices] = model14.predict_proba(testX)
    # score the model on validation fold
    scores.append(model14.score(testX, testy))
    print('Score on Fold {} : {:.3f}%, Time taken = {:.3f}s'.format(i, scores[-1]*100, end-start))
    i += 1
print(f'\nMean score of KFold CV for {type(model14).__name__}: {100*np.mean(scores):.3f}% ± {100*np.std(scores):.3f}%')

# loss_lists = np.array(loss_lists)
# print(np.mean(loss_lists, axis=0))
# print(np.argmin(np.mean(loss_lists, axis=0)), np.min(np.mean(loss_lists, axis=0)))


print('\nTraining on train data and predicting for test...')
model14.fit(X_train, y_train, verbose=True)
print('Finished Training')
ypredtest = model14.predict(X_test)
ytestprobs = model14.predict_proba(X_test)


train_df = pd.DataFrame(data = ytrainprobs, columns = ["m14_pred_"+str(i) for i in range(1, num_classes+1)])
train_df['m14_predictions'] = ypredtrain

test_df = pd.DataFrame(data=ytestprobs, columns = ["m14_pred_"+str(i) for i in range(1, num_classes+1)])
test_df['m14_predictions'] = ypredtest
test_df['filename'] = test_files

train_df.to_csv('../predictions/train_predictions/model14.csv', index = None)
test_df.to_csv('../predictions/test_predictions/model14.csv', index = None)

