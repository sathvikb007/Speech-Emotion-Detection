# Get the critical imports out of the way
from operator import mod
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import librosa.display
import soundfile
import os
import seaborn as sns
import tqdm
# matplotlib complains about the behaviour of librosa.display, so we'll ignore those warnings:
import warnings; warnings.filterwarnings('ignore')
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

X_train = np.load('../../dataset/features_train/features1.npy')
y_train = np.load('../../dataset/features_train/emotions1.npy')
X_test = np.load('../../dataset/features_test/features1test.npy')
test_files = np.load('../../dataset/features_test/testfiles1.npy')

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

input_seq = tf.keras.layers.Input(shape=(X_train.shape[1]))

x = tf.keras.layers.Dense(90, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l2(1e-6))(input_seq)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(45, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=tf.keras.regularizers.l2(1e-6))(x)
x = tf.keras.layers.BatchNormalization()(x)
encoded = tf.keras.layers.Dense(30, activation='relu', kernel_initializer='he_uniform', name = 'encoded', kernel_regularizer=tf.keras.regularizers.l2(1e-6))(x)
x = tf.keras.layers.BatchNormalization()(encoded)
x = tf.keras.layers.Dense(45, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l2(1e-6))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(90, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l2(1e-6))(x)
x = tf.keras.layers.BatchNormalization()(x)
decoded = tf.keras.layers.Dense(180, kernel_initializer='he_uniform', activation='sigmoid')(x)


# x = tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same')(input_seq)
# x = tf.keras.layers.MaxPooling1D(2, padding='same')(x)
# x = tf.keras.layers.Conv1D(8, 3, activation='relu', padding='same')(x)
# x = tf.keras.layers.MaxPooling1D(2, padding='same')(x)
# x = tf.keras.layers.Conv1D(8, 3, activation='relu', padding='same')(x)
# x = tf.keras.layers.MaxPooling1D(2, padding='same')(x)
# x = tf.keras.layers.Conv1D(1, 3, activation='relu', padding='same')(x)
# encoded = tf.keras.layers.MaxPooling1D(2, padding='same', name = 'encoded')(x)

# # at this point the representation is (4, 4, 8) i.e. 128-dimensional

# x = tf.keras.layers.Conv1D(1, 3, activation='relu', padding='same')(encoded)
# x = tf.keras.layers.UpSampling1D(2)(x)
# x = tf.keras.layers.Conv1D(8, 3, activation='relu', padding='same')(x)
# x = tf.keras.layers.UpSampling1D(2)(x)
# x = tf.keras.layers.Conv1D(8, 3, activation='relu', padding='same')(x)
# x = tf.keras.layers.UpSampling1D(2)(x)
# x = tf.keras.layers.Conv1D(16, 3, activation='relu')(x)
# x = tf.keras.layers.UpSampling1D(2)(x)
# decoded = tf.keras.layers.Conv1D(1, 3, activation='sigmoid', padding='same')(x)




# x = tf.keras.layers.Conv1D(16, 3, activation="relu", padding="same")(input_seq) # 10 dims
# #x = BatchNormalization()(x)
# x = tf.keras.layers.MaxPooling1D(2, padding="same")(x) # 5 dims
# x = tf.keras.layers.Conv1D(1, 3, activation="relu", padding="same")(x) # 5 dims
# #x = BatchNormalization()(x)
# encoded = tf.keras.layers.MaxPooling1D(2, padding="same", name = "encoded")(x) # 3 dims

# encoder = tf.keras.Model(input_seq, encoded)

# # 3 dimensions in the encoded layer

# x = tf.keras.layers.Conv1D(1, 3, activation="relu", padding="same")(encoded) # 3 dims
# #x = BatchNormalization()(x)
# x = tf.keras.layers.UpSampling1D(2)(x) # 6 dims
# x = tf.keras.layers.Conv1D(16, 3, activation='relu')(x) # 5 dims
# #x = BatchNormalization()(x)
# x = tf.keras.layers.UpSampling1D(2)(x) # 10 dims
# decoded = tf.keras.layers.Conv1D(1, 3, activation='sigmoid', padding='same')(x) # 10 dims

encoder = tf.keras.Model(input_seq, encoded)
autoencoder = tf.keras.Model(input_seq, decoded)
opt = tf.keras.optimizers.RMSprop(lr=1e-3, rho=0.9, epsilon=1e-07, decay=0.1)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=0.0001)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='../../weights/autoencoder.h5', monitor='val_loss', save_best_only=True, verbose=1)

autoencoder.fit(X_train, X_train,
                epochs=1000,
                batch_size=128,
                shuffle=True,
                validation_split=0.2,
                callbacks=[tf.keras.callbacks.TensorBoard(log_dir='/tmp/autoencoder'), early_stopping, model_checkpoint])

autoencoder.load_weights('../../weights/autoencoder.h5')

X_train = encoder.predict(X_train)
X_test = encoder.predict(X_test)


np.save('../../dataset/features_test/features3test.npy', X_test)
np.save('../../dataset/features_test/testfiles3.npy', np.array(test_files))

np.save('../../dataset/features_train/features3.npy', X_train)
np.save('../../dataset/features_train/emotions3.npy', y_train)
