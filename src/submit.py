import numpy as np
import pandas as pd

m1_df = pd.read_csv('../predictions/test_predictions/lvl1_model1.csv')
m2_df = pd.read_csv('../predictions/test_predictions/lvl1_model2.csv')

final_probs = pd.DataFrame()

# print(m1_df['l1_m1_predictions'].unique())
# print(m1_df[m1_df['l1_m1_predictions']=='sadness'].head())

weights = np.array([0.3, 0.7])

for i in range(1, 8):
    final_probs["pred"+str(i)] = weights[0]*m1_df["l1_m1_pred_"+str(i)] + weights[1]*m2_df["l1_m2_pred_"+str(i)]

# final_preds = np.argmax(final_probs, axis=0)

# print(final_probs.tail())

final_preds = np.argmax(final_probs.values, axis = 1)

emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
emot_to_int = {emotions[0]: 0, emotions[1]: 1, emotions[2]: 2, emotions[3]: 3, emotions[4]: 4, emotions[5]: 5, emotions[6]: 6}
int_to_emot = {0: emotions[0], 1: emotions[1], 2: emotions[2], 3: emotions[3], 4: emotions[4], 5: emotions[5], 6: emotions[6]}
# print(final_preds[:10])

submission = pd.DataFrame()
submission['filename'] = m1_df['filename']
submission['emotion'] = [int_to_emot[emot_to_int[emotions[int(final_preds[i])]]] for i in range(len(final_preds))]

submission.to_csv('final_submission.csv', index=False)

