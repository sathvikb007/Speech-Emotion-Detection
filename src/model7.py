import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, accuracy_score
import warnings; warnings.filterwarnings('ignore')
from time import time
import tensorflow as tf

np.random.seed(420)

X_train = np.load('../dataset/features_train/features2.npy')
y_train = np.load('../dataset/features_train/emotions2.npy')
X_test = np.load('../dataset/features_test/features2test.npy')
test_files = np.load('../dataset/features_test/testfiles2.npy')
emotions = ['surprise', 'neutral', 'fear', 'joy', 'sadness', 'anger', 'disgust']
emot_to_int = {emotions[0]: 0, emotions[1]: 1, emotions[2]: 2, emotions[3]: 3, emotions[4]: 4, emotions[5]: 5, emotions[6]: 6}
int_to_emot = {0: emotions[0], 1: emotions[1], 2: emotions[2], 3: emotions[3], 4: emotions[4], 5: emotions[5], 6: emotions[6]}

# y_train = np.vectorize(emot_to_int.get)(y_train)



class Classifier_CNN:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, load_weights=False):
        self.output_directory = output_directory
        self.best_weights = os.path.join(output_directory, 'best_model7.hdf5')
        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                print(self.model.summary())
            self.verbose = verbose
        return

    def build_model(self, input_shape, nb_classes):
        padding = 'valid'
        input_layer = tf.keras.layers.Input(input_shape)

        if input_shape[0] < 60: # for italypowerondemand dataset
            padding = 'same'

        conv1 = tf.keras.layers.Conv1D(filters=64,kernel_size=7,padding=padding,activation='relu')(input_layer)
        conv1 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv1)

        conv2 = tf.keras.layers.Conv1D(filters=32,kernel_size=5,padding=padding,activation='relu')(conv1)
        conv2 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv2)

        flatten_layer = tf.keras.layers.Flatten()(conv2)

        output_layer = tf.keras.layers.Dense(units=nb_classes,activation='softmax')(flatten_layer)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='sparse_categorical_crossentropy', 
                        optimizer=tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9, nesterov=True), metrics=['accuracy'])


        self.reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5, verbose=1)

        # file_path = os.path.join(self.output_directory, 'best_model.hdf5')

        self.model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath= self.best_weights,  monitor='val_loss',
                                                           save_best_only=True)

        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

        self.callbacks = [self.reduce_lr, self.model_checkpoint, self.early_stopping]

        return model

    def fit(self, x_train, y_train, batch_size, epochs, x_val=None, y_val=None):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training
        nb_epochs = epochs

        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        if x_val is not None:
            hist = self.model.fit(x_train, y_train, 
                                batch_size=mini_batch_size, 
                                epochs=nb_epochs,
                                verbose=1, 
                                validation_data=(x_val, y_val), 
                                callbacks=self.callbacks
            )
        else:
            hist = self.model.fit(x_train, y_train, 
                                batch_size=mini_batch_size, 
                                epochs=nb_epochs,
                                verbose=1
            )

        # duration = time.time() - start_time

        # self.model.save(os.path.join(self.output_directory, 'last_model.hdf5'))

        # y_pred = self.predict(x_val, y_true, x_train, y_train, y_val,
                            #   return_df_metrics=False)

        # save predictions
        # np.save(self.output_directory + 'y_pred.npy', y_pred)

        # convert the predicted from binary to integer
        # y_pred = np.argmax(y_pred, axis=1)

        # df_metrics = save_logs(self.output_directory, hist, y_pred, y_true, duration)

        tf.keras.backend.clear_session()

        return hist


    def predict(self, X, y_true = None):
        if y_true is not None:
            self.model.load_weights(self.best_weights)
        y_probs = self.model.predict(X)
        y_pred = np.vectorize(int_to_emot.get)(np.argmax(y_probs, axis=1))
        
        if y_true is not None:
            score = accuracy_score(y_true, np.argmax(y_probs, axis=1))
            return y_pred, y_probs, score
        return y_pred, y_probs


skfold = StratifiedKFold(
    n_splits=6, 
    shuffle=True
)

num_classes = pd.Series(y_train).nunique()
ypredtrain = np.empty((len(y_train)), dtype=np.dtype('<U8'))
ytrainprobs = np.empty((len(y_train), num_classes))
ypredtest = np.empty((len(X_test)), dtype= np.dtype('<U8'))
ytestprobs = np.empty((len(X_test), num_classes))

scores = []
lens = []
i = 1
for train_indices, test_indices in skfold.split(X_train, y_train):
    
    # fit model to training fold
    trainX, testX = X_train[train_indices], X_train[test_indices]
    trainy, testy = y_train[train_indices], y_train[test_indices]

    scaler = StandardScaler()
    trainX = scaler.fit_transform(trainX)
    testX = scaler.transform(testX)

    trainy = np.vectorize(emot_to_int.get)(trainy)
    testy = np.vectorize(emot_to_int.get)(testy)

    start = time()
    model7 = Classifier_CNN('../weights/', (trainX.shape[1], 1), num_classes, verbose=False)
    hist = model7.fit(trainX, trainy, batch_size=32, epochs=1000, x_val = testX, y_val = testy)
    end = time()
    lens.append(len(hist.history['loss']))
    # get predictions
    ypredtrain[test_indices], ytrainprobs[test_indices], fold_score = model7.predict(testX, testy)
    # score the model on validation fold
    scores.append(fold_score)
    print('Score on Fold {} : {:.3f}%, Time taken = {:.3f}s'.format(i, scores[-1]*100, end-start))
    i += 1
print(f'\nMean score of KFold CV for {type(model7).__name__}: {100*np.mean(scores):.3f}% ± {100*np.std(scores):.3f}%')
print(f'Num epochs = {lens}')
test_epochs = np.max(lens)

print('\nTraining on train data and predicting for test...')
tf.keras.backend.clear_session()
scaler.fit_transform(X_train)
scaler.transform(X_test)
y_train = np.vectorize(emot_to_int.get)(y_train)
history = model7.fit(X_train, y_train, batch_size = 32, epochs = test_epochs)
print('Finished Training')
ypredtest, ytestprobs = model7.predict(X_test)


train_df = pd.DataFrame(data = ytrainprobs, columns = ["m7_pred_"+str(i) for i in range(1, num_classes+1)])
train_df['m7_predictions'] = ypredtrain

test_df = pd.DataFrame(data=ytestprobs, columns = ["m7_pred_"+str(i) for i in range(1, num_classes+1)])
test_df['m7_predictions'] = ypredtest
test_df['filename'] = test_files



train_df.to_csv('../predictions/train_predictions/model7.csv', index = None)
test_df.to_csv('../predictions/test_predictions/model7.csv', index = None)

