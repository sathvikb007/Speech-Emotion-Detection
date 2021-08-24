import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from time import time
import imblearn.over_sampling as ovs
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier, EasyEnsembleClassifier

np.random.seed(420)

X_train = np.load('../dataset/features_train/features2.npy')
y_train = np.load('../dataset/features_train/emotions2.npy')
X_test = np.load('../dataset/features_test/features2test.npy')
test_files = np.load('../dataset/features_test/testfiles2.npy')


skfold = StratifiedKFold(
    n_splits=6, 
    shuffle=True
)

# model = GaussianNB()

# param_grid = {
#     'var_smoothing': [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000, 100000, 1000000]
# }

# grid = RandomizedSearchCV(estimator=model, 
#                             param_distributions=param_grid, 
#                             n_iter = 100, scoring='accuracy', 
#                             cv = skfold, 
#                             verbose=3)

# grid = GridSearchCV(estimator=model,
#                         param_grid=param_grid,
#                         cv = skfold,
#                         verbose=3,
#                         n_jobs= -1
# )

# grid.fit(X_train, y_train)
# best_score = grid.best_score_
# best_params = grid.best_params_
# print('Best Accuracy : {:.3f}%'.format(best_score))
# print('Best Parameters : ',best_params)



model10 = GaussianNB(
    var_smoothing= 8e-3
    )
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

    # scaler = MaxAbsScaler()
    # trainX = scaler.fit_transform(trainX)
    # testX = scaler.transform(testX)

    start = time()
    model10.fit(trainX, trainy)
    end = time()
    # get predictions
    ypredtrain[test_indices] = model10.predict(testX)
    ytrainprobs[test_indices] = model10.predict_proba(testX)
    # score the model on validation fold
    scores.append(model10.score(testX, testy))
    print('Score on Fold {} : {:.3f}%, Time taken = {:.3f}s'.format(i, scores[-1]*100, end-start))
    i += 1
print(f'\nMean score of KFold CV for {type(model10).__name__}: {100*np.mean(scores):.3f}% ± {100*np.std(scores):.3f}%')


print('\nTraining on train data and predicting for test...')
# scaler.fit_transform(X_train)
# scaler.transform(X_test)
model10.fit(X_train, y_train)
print('Finished Training')
ypredtest = model10.predict(X_test)
ytestprobs = model10.predict_proba(X_test)


train_df = pd.DataFrame(data = ytrainprobs, columns = ["m10_pred_"+str(i) for i in range(1, num_classes+1)])
train_df['m10_predictions'] = ypredtrain

test_df = pd.DataFrame(data=ytestprobs, columns = ["m10_pred_"+str(i) for i in range(1, num_classes+1)])
test_df['m10_predictions'] = ypredtest
test_df['filename'] = test_files



train_df.to_csv('../predictions/train_predictions/model10.csv', index = None)
test_df.to_csv('../predictions/test_predictions/model10.csv', index = None)

