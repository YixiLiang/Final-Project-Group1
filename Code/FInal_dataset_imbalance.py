'''
@Project ：test_nlp 
@File    ：FInal_dataset_imbalance.py
@Author  ：Yixi Liang
@Date    ：2022/12/9 11:23 
'''
import pandas as pd

final_res = pd.read_csv(f'final_res.csv')
final_res.drop(columns=['label'], inplace=True)
label_dic = {'Claim': 5, 'Evidence': 4, 'Position': 2, 'Concluding Statement': 6,
       'Lead': 0, 'Counterclaim': 1, 'Rebuttal': 3}
final_res['label'] = final_res['discourse_type'].apply(lambda x: label_dic[x])



df_train = pd.read_csv('train.csv')


df_train = df_train[25000:]
df_train['label'] = df_train['discourse_type'].apply(lambda x: label_dic[x])

df_train_2 = df_train.loc[df_train['label']==2, : ].sample(n=456)
df_train_6 = df_train.loc[df_train['label']==6, : ].sample(n=(3000-2257))
df_train_0 = df_train.loc[df_train['label']==0, : ].sample(n=(3000-1937))
df_train_1 = df_train.loc[df_train['label']==1, : ].sample(n=(3000-988))
df_train_3 = df_train.loc[df_train['label']==3, : ].sample(n=(3000-800))

df_unsum = pd.concat([df_train_2,df_train_6,df_train_0,df_train_1,df_train_3])
df_unsum.to_csv('unsum_train.csv')

res = pd.concat([final_res, df_unsum])
res = pd.DataFrame({'text':res['discourse_text'], 'label':res['label']})
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(res, test_size=0.1)
train_df.to_csv('train_balanced.csv')
test_df.to_csv('test_balanced.csv')

