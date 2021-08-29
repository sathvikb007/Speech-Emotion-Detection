# Speech Emotion Recognition Ensemble

Emotion Recognition from Audio Files.

The audio files are speech files of mono type from various speakers.

## Feature Extraction
Three types of features from the audio files are extracted:
- **MFCC**: Mel Frequency Cepstral Coefficients.
- **Chromograms**: STFT is performed on the audio file and the chromogram is extracted.
- **Spectrograms**: Mel Spectrograms from the STFT frames

Two feature sets are created by varying the number of features as listed above.

The third feature set is created by using an autoencoder to extract useful features from the audio features


## Stacking - Level 0 Models

A stacking ensemble of various models is used. The description of the various models used in level 0 is given in the predictions/ folder under the name model_description.md.

Various models use different feature sets, algorithms, and/or hyperparameters

## Stacking - Level 1 Models

- Model 1: Bagging Classifier of XGBoost
- Model 2: Bagging Classifier of Extra Trees

## Final Predictions

Weighted average of the predictions from the level 1 models