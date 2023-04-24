import numpy as np
import pandas as pd
import googletrans
import time
import tqdm
from tqdm import tqdm

train_data = pd.read_csv('data/train.csv')
print(type(train_data))
val_data = pd.read_csv('data/dev.csv')
train_data = pd.concat([train_data, val_data], axis = 0)
#분리 작업(전처리 후에는 문장 위치가 사라지니 미리 나눔)
num = len(train_data)
train_data, val_data, test_data = train_data.iloc[0:int(num*0.7), :], train_data.iloc[int(num*0.7):int(num*0.85), :], train_data.iloc[int(num*0.85):, :]

translator = googletrans.Translator()
count = 0
def ko2ko(text, translator):
    global count

    try:
        eng = translator.translate(text, dest = 'en', src = 'ko')
        back = translator.translate(eng.text, dest = 'ko', src = 'en')
        return back.text
    
    except:
        count+=1
        return text
    

train_data = train_data.iloc[:, :]
df_only1 = train_data.copy()
df_only2 = train_data.copy()
df_both12 = train_data.copy()

tqdm.pandas()
df_only1['sentence_1'] = df_only1['sentence_1'].progress_apply(ko2ko, translator = translator)
print(f'first error number = {count}')
df_only2['sentence_2'] = df_only2['sentence_2'].progress_apply(ko2ko, translator = translator)
print(f'total error number = {count}')
df_both12['sentence_1'] = df_only1['sentence_1']
df_both12['sentence_2'] = df_only2['sentence_2']

train_data = pd.concat([train_data, df_only1, df_only2, df_both12])

train_data2 = train_data.copy()
train_data2.rename(columns={'sentence_1':'sentence_2', 'sentence_2':'sentence_1'}, inplace = True)
train_data  = pd.concat([train_data, train_data2])

train_data.drop_duplicates(keep='first',ignore_index=True, inplace = True)


train_data.to_csv('translated_train_data.csv')
val_data.to_csv('val_data.csv')
test_data.to_csv('real_test.csv')