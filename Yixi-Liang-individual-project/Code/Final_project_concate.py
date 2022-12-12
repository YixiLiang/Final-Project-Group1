'''
@Project ：test_nlp 
@File    ：Final_project_concate.py
@Author  ：Yixi Liang
@Date    ：2022/12/9 11:09 
'''
import pandas as pd
from sklearn.model_selection import train_test_split

final_res = pd.read_csv(f'final_res.csv')
final_res = pd.DataFrame({'text':final_res['discourse_text'], 'label': final_res['label'], 'summary':final_res['summary']})
# final_res.drop(columns=['label'], inplace=True)
# label_dic = {'Claim': 5, 'Evidence': 4, 'Position': 2, 'Concluding Statement': 6,
#        'Lead': 0, 'Counterclaim': 1, 'Rebuttal': 3}
# final_res['label'] = final_res['discourse_type'].apply(lambda x: label_dic[x])
# final_res.to_csv('final_res.csv')

final_res_0_2 = pd.read_csv(f'final_sum_0_2000.csv')
final_res_0_2 = pd.DataFrame({'text':final_res_0_2['text'], 'label': final_res_0_2['label'], 'summary':final_res_0_2['summary']})
final_res_2_4 = pd.read_csv(f'final_sum_2000_4000.csv')
final_res_2_4 = pd.DataFrame({'text':final_res_2_4['text'], 'label': final_res_2_4['label'], 'summary':final_res_2_4['summary']})
final_res_4_6 = pd.read_csv(f'final_sum_4000_6474.csv')
final_res_4_6 = pd.DataFrame({'text':final_res_4_6['text'], 'label': final_res_4_6['label'], 'summary':final_res_4_6['summary']})
final_res = pd.concat([final_res,final_res_0_2,final_res_2_4,final_res_4_6], ignore_index=True)

train_balanced, test_balanced = train_test_split(final_res, test_size=0.1)

train_balanced.to_csv('train_balanced.csv', index=False)
test_balanced.to_csv('test_balanced.csv', index=False)
print('Finish')

