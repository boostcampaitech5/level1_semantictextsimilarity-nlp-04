import pandas as pd
import transformers
import matplotlib.pyplot as plt

df = pd.read_csv('./train_data_after_preprocessing.csv')
print(df.head())

sentences = [df.loc[i, 'sentence_1'] + '[SEP]' + df.loc[i, 'sentence_2'] for i in range(len(df))]
print(len(sentences) == len(df))

tokenizer = transformers.AutoTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')
print(tokenizer(sentences[1])['input_ids'])
a = max(len(tokenizer(sentence)['input_ids']) for sentence in sentences)
print(a)
b = sum(len(tokenizer(sentence)['input_ids']) for sentence in sentences) / len(sentences)
print(b)
c = [len(tokenizer(sentence)['input_ids']) for sentence in sentences]
print(c)
plt.hist(c)
plt.show()