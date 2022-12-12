'''
@Project ：test_nlp 
@File    ：Final_project_clean_data_add_summary.py
@Author  ：Yixi Liang
@Date    ：2022/12/8 23:01 
'''
from tqdm import tqdm
from transformers import pipeline
import pandas as pd

num = 0
for i in range(20):
    print(50*'*')
    print(num)
    df_raw_train = pd.read_csv(f'train.csv')
    df_raw_train = df_raw_train[['discourse_text', 'discourse_type']]
    # df_raw_test = pd.read_csv(f'sample_submission.csv')
    df_train = df_raw_train[num:num+1000]
    # df_test = df_raw_test

    label_set = set(df_train['discourse_type'])
    NUM_LABEL = len(label_set)
    label_dic = {}
    for index, label in enumerate(label_set):
        label_dic[label] = index

    df_train['label'] = df_train['discourse_type'].apply(lambda x: label_dic[x])
    summarizer = pipeline("summarization",  model="t5-base", tokenizer="t5-base")
    df_train['summary'] = df_train['discourse_text'].apply(lambda x: summarizer(x, min_length=5, max_length=30)[0]['summary_text'])

    df_train.to_csv(f'test_summary_{num}_{num+1000}.csv')
    num += 1000
