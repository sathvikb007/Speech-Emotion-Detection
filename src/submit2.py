import numpy as np
import pandas as pd

df = pd.read_csv('../predictions/test_predictions/model9.csv')
print(df.head())
submit_df = df[['filename', 'm9_predictions']].rename(columns={'m9_predictions': 'emotion'})
print(submit_df.head())
submit_df.to_csv('../submission2.csv', index = None)