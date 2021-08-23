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
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

np.random.seed(420)

X_train = np.load('../dataset/features_train/features1.npy')
y_train = np.load('../dataset/features_train/emotions1.npy')
X_test = np.load('../dataset/features_test/features1test.npy')
test_files = np.load('../dataset/features_test/testfiles1.npy')


skfold = StratifiedKFold(
    n_splits=6, 
    shuffle=True
)

model = XGBClassifier( 
    learning_rate =0.1, 
    n_estimators=1000, 
)

param_grid = {
    'max_depth':range(3,10,2),
    'min_child_weight':range(1,6,2),
    'gamma':[i/10.0 for i in range(0,5)],
    'subsample':[i/10.0 for i in range(6,10)],
    'colsample_bytree':[i/10.0 for i in range(6,10)],
    'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
              

grid = RandomizedSearchCV(estimator=model, 
                            param_distributions=param_grid, 
                            n_iter = 500, scoring='accuracy', 
                            cv = skfold, 
                            verbose=3,
                            n_jobs= 1
)

# grid = GridSearchCV(estimator=model,
#                         param_grid=param_grid,
#                         cv = skfold,
#                         scoring= 'accuracy',
#                         verbose=5,
#                         n_jobs=1
# )

grid.fit(X_train, y_train, eval_metric = 'mlogloss', verbose = True)
best_score = grid.best_score_
best_params = grid.best_params_
print('Best Accuracy : {:.3f}%'.format(best_score))
print('Best Parameters : ',best_params)



# model4 = XGBClassifier( 
#     learning_rate = 5e-2, 
#     n_estimators=5000, 
#     max_depth=7,
#     objective= 'multi:softprob',
#     min_child_weight=2, 
#     gamma=0.1, 
#     subsample=0.9, 
#     colsample_bytree=0.6,
#     reg_alpha = 1e-5
#     # scale_pos_weight=1,
# )

# num_classes = pd.Series(y_train).nunique()
# ypredtrain = np.empty((len(y_train)), dtype=np.dtype('<U8'))
# ytrainprobs = np.empty((len(y_train), num_classes))
# ypredtest = np.empty((len(X_test)), dtype= np.dtype('<U8'))
# ytestprobs = np.empty((len(X_test), num_classes))

# scores = []
# i = 1
# for train_indices, test_indices in skfold.split(X_train, y_train):
    
#     # fit model to training fold
#     trainX, testX = X_train[train_indices], X_train[test_indices]
#     trainy, testy = y_train[train_indices], y_train[test_indices]

#     scaler = StandardScaler()
#     scaler.fit_transform(trainX)
#     scaler.transform(testX)

#     start = time()
#     model4.fit(trainX, trainy, eval_set=[(testX, testy)], early_stopping_rounds=80, eval_metric= 'merror', verbose= True)
#     end = time()
#     # get predictions
#     ypredtrain[test_indices] = model4.predict(testX)
#     ytrainprobs[test_indices] = model4.predict_proba(testX)
#     # score the model on validation fold
#     scores.append(model4.score(testX, testy))
#     print('Score on Fold {} : {:.3f}%, Time taken = {:.3f}s'.format(i, scores[-1]*100, end-start))
#     i += 1
# print(f'\nMean score of KFold CV for {type(model4).__name__}: {100*np.mean(scores):.3f}% ± {100*np.std(scores):.3f}%')


# print('\nTraining on train data and predicting for test...')
# model4.fit(X_train, y_train)
# print('Finished Training')
# ypredtest = model4.predict(X_test)
# ytestprobs = model4.predict_proba(X_test)


# train_df = pd.DataFrame(data = ytrainprobs, columns = ["m4_pred_"+str(i) for i in range(1, num_classes+1)])
# train_df['m4_predictions'] = ypredtrain

# test_df = pd.DataFrame(data=ytestprobs, columns = ["m4_pred_"+str(i) for i in range(1, num_classes+1)])
# test_df['m4_predictions'] = ypredtest
# test_df['filename'] = test_files

# train_df.to_csv('../predictions/train_predictions/model4.csv', index = None)
# test_df.to_csv('../predictions/test_predictions/model4.csv', index = None)

