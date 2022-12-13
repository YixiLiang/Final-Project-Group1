'''
@Project ：test_nlp 
@File    ：Final_project_clean_data_add_summary.py
@Author  ：Yixi Liang
@Date    ：2022/12/8 23:01 
'''
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import pipeline
import pandas as pd

label_dic = {'Claim': 5, 'Evidence': 4, 'Position': 2, 'Concluding Statement': 6,
       'Lead': 0, 'Counterclaim': 1, 'Rebuttal': 3}

num = 0
for i in range(20):
    print(50*'*')
    print(num)
    df_raw_train = pd.read_csv(f'train.csv')
    df_raw_train = df_raw_train[['discourse_text', 'discourse_type']]
    df_train = df_raw_train[num:num+1000]

    label_set = set(df_train['discourse_type'])
    NUM_LABEL = len(label_set)

    df_train['label'] = df_train['discourse_type'].apply(lambda x: label_dic[x])
    summarizer = pipeline("summarization",  model="t5-base", tokenizer="t5-base")
    df_train['summary'] = df_train['discourse_text'].apply(lambda x: summarizer(x, min_length=5, max_length=30)[0]['summary_text'])

    df_train.to_csv(f'summary_{num}_{num+1000}.csv')
    num += 1000

final_dataset = pd.read_csv(f'summary_0_1000.csv')
num = 0
for i in range(1, 20):
    cur_dataset = pd.read_csv(f'summary_{num}_{num+1000}.csv')
    final_dataset = pd.concat([final_dataset, cur_dataset], ignore_index=True)
    num += 1000

final_dataset = pd.DataFrame({'text':final_dataset['discourse_text'], 'label': final_dataset['label'], 'summary':final_dataset['summary']})
train_balanced, test_balanced = train_test_split(final_dataset, test_size=0.1)

train_balanced.to_csv('train_balanced.csv', index=False)
test_balanced.to_csv('test_balanced.csv', index=False)
print('Finish')


