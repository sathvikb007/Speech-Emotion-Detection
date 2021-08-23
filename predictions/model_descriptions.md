### Model 1
MaxAbsScaler

model1 = KNeighborsClassifier(
    n_neighbors=27,
    weights='distance',
    algorithm='kd_tree',
    leaf_size=10,
    p=1,
    n_jobs= -1
)
Mean score of KFold CV for KNeighborsClassifier: 57.359% ± 0.401%

### Model 2
StandardScaler

model2 = LogisticRegression(
    C=0.01
)
Mean score of KFold CV for LogisticRegression: 51.530% ± 0.429%

### Model 3
StandardScaler

model3 = SVC(
    kernel= 'linear',
    probability=True,
    C=0.01
)
Mean score of KFold CV for SVC: 51.444% ± 0.419%

### Model 4

model4 = XGBClassifier( 
    learning_rate = 5e-2, 
    n_estimators=5000, 
    max_depth=7,
    objective= 'multi:softprob',
    min_child_weight=2, 
    gamma=0.1, 
    subsample=0.9, 
    colsample_bytree=0.6,
    reg_alpha = 1e-5
    # scale_pos_weight=1,
)


### Model 6
features 2
MinMaxScaler

model6 = KNeighborsClassifier(
    n_neighbors=31,
    weights='distance',
    algorithm='kd_tree',
    leaf_size=10,
    p=1,
    n_jobs= -1
)

Mean score of KFold CV for KNeighborsClassifier: 57.411% ± 0.741%

### Model 7
features 2
Mean score of KFold CV for Classifier_CNN: 51.496% ± 0.620%

