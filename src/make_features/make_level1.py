import numpy as np 
import pandas as pd
import sys
import os

train_labels = np.load('../../dataset/features_train/emotions3.npy')

def load_files(path):
    """Load all csv files from path and join them as dataframes"""
    dfs = []
    for file in os.listdir(path):
        if file.endswith(".csv"):
            d = pd.read_csv(os.path.join(path,file))
            # print(d.head())
            dfs.append(d)
            # print(file)
    return pd.concat(dfs, axis = 1)

df_train = load_files('../../predictions/train_predictions')
df_test = load_files('../../predictions/test_predictions')

np.save('../../dataset/features_test/lvl1_emotions.npy', train_labels)
df_train.to_csv('../../dataset/features_train/lvl1_features.csv', index = False)
df_test.to_csv('../../dataset/features_test/lvl1_featurestest.csv', index = False)

