# Get the critical imports out of the way
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
from IPython.core.display import HTML 
# Center matplotlib figures...
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")

train_dist = pd.read_csv('../../dataset/train.csv', index_col = 'filename')
test_dist = pd.read_csv('../../dataset/test.csv', index_col = 'filename')

import librosa

def feature_chromagram(waveform, sample_rate):
    # STFT computed here explicitly; mel spectrogram and MFCC functions do this under the hood
    stft_spectrogram=np.abs(librosa.stft(waveform))
    # Produce the chromagram for all STFT frames and get the mean of each column of the resulting matrix to create a feature array
    chromagram=np.mean(librosa.feature.chroma_stft(S=stft_spectrogram, sr=sample_rate).T,axis=0)
    return chromagram

def feature_melspectrogram(waveform, sample_rate):
    # Produce the mel spectrogram for all STFT frames and get the mean of each column of the resulting matrix to create a feature array
    # Using 8khz as upper frequency bound should be enough for most speech classification tasks
    melspectrogram=np.mean(librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=32, fmax=8000).T,axis=0)
    return melspectrogram

def feature_mfcc(waveform, sample_rate):
    # Compute the MFCCs for all STFT frames and get the mean of each column of the resulting matrix to create a feature array
    # 40 filterbanks = 40 coefficients
    mfc_coefficients=np.mean(librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13).T, axis=0) 
    return mfc_coefficients

def get_features(file):
    # load an individual soundfile
    waveform, sample_rate = librosa.load(file, mono=True)
    chromagram = feature_chromagram(waveform, sample_rate)
    melspectrogram = feature_melspectrogram(waveform, sample_rate)
    mfc_coefficients = feature_mfcc(waveform, sample_rate)

    feature_matrix=np.array([])
    # use np.hstack to stack our feature arrays horizontally to create a feature matrix
    feature_matrix = np.hstack((chromagram, melspectrogram, mfc_coefficients))

    return feature_matrix

import os, glob

def load_train_data():
    X,y=[],[]
    count = 0
    for file in tqdm.tqdm(glob.glob("../../dataset/TrainAudioFiles/*")):
        file_name=os.path.basename(file)
        emotion= train_dist.loc[file_name].emotion
        features = get_features(file)
        X.append(features)
        y.append(emotion)
        count += 1
        # '\r' + end='' results in printing over same line
        #print('\r' + f' Processed {count} audio samples',end=' ')
    # Return arrays to plug into sklearn's cross-validation algorithms
    return np.array(X), np.array(y)

def load_test_data():
    X = []
    file_list = []
    count = 0
    for file in tqdm.tqdm(glob.glob("../../dataset/TestAudioFiles/*")):
        file_name=os.path.basename(file)
#         emotion= train_dist.loc[file_name].emotion
        features = get_features(file)
        X.append(features)
        file_list.append(file_name)
        count += 1
        # '\r' + end='' results in printing over same line
        #print('\r' + f' Processed {count} audio samples',end=' ')
    # Return arrays to plug into sklearn's cross-validation algorithms
    return np.array(X), file_list

features, emotions = load_train_data()
test_features, test_files = load_test_data()

np.save('../../dataset/features_test/features2test.npy', test_features)
np.save('../../dataset/features_test/testfiles2.npy', np.array(test_files))

np.save('../../dataset/features_train/features2.npy', features)
np.save('../../dataset/features_train/emotions2.npy', emotions)

print(f'\nAudio samples represented: {features.shape[0]}')
print(f'Numerical features extracted per sample: {features.shape[1]}')