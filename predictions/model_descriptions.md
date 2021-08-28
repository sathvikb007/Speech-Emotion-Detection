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

### Model 8
features 2
StandarScaler

model8 = XGBClassifier( 
    learning_rate = 0.2, 
    n_estimators=5000,
    max_depth = 11,
    min_child_weight = 1,
    gamma = 0.0,
    subsample = 0.9,
    colsample_bytree = 0.6,
    reg_alpha = 1e-2
)

### Model 9
features 2
RobustScaler

model9 = RandomForestClassifier(
    n_estimators = 500, 
    max_depth = 30,
    max_features = 'sqrt'
    )

score was 58 ish

### Model 10
features 2
No scaling

model10 = GaussianNB(
    var_smoothing= 8e-3
    )

Mean score of KFold CV for GaussianNB: 50.000% ± 0.400%

### Model 11
features 3
StandardScaler

model11 = KNeighborsClassifier(
    n_neighbors= 35,
    weights='distance',
    leaf_size=10,
    p=2,
    n_jobs= -1
)

Mean score of KFold CV for KNeighborsClassifier: 57.015% ± 0.924%

### Model 12
features 3
StandardScaler

model12 = LogisticRegression(
    C=0.1
)

Mean score of KFold CV for LogisticRegression: 51.307% ± 0.445%

### Model 13
features 3
RobustScaler

model13 = ExtraTreesClassifier(
    n_estimators = 1000, 
    max_depth = 30,
    max_features = None,
    min_samples_leaf= 1,
    min_samples_split= 5,
    n_jobs=-1
)

Mean score of KFold CV for ExtraTreesClassifier: 58.683% ± 0.801%
