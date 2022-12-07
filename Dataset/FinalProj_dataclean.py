import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re
import nltk


# ------------------------preprocessing------------------------
# ----------load data
dataset = pd.read_csv('train.csv')
data = dataset[['discourse_text', 'discourse_type']].copy()
# ----------check null values
print('missing values exists:\n', data.isnull().any())
# ----------set numeric label
labels = enumerate(set(data['discourse_type']))
label2int = {}
for i, l in labels:
    label2int[l] = i

data['label'] = data['discourse_type'].apply(lambda x: label2int[x])
print(data.head())


# ----------clean text
def clean_text(x):
    # ----------lowercase
    x = x.lower()
    # ----------remove url
    match_url = re.compile(r'https?://(www\.)?(\w+)(\.\w+)')
    x = re.sub(match_url, "", x)
    return x

data['text'] = data['discourse_text'].astype(str).apply(lambda x: clean_text(x))

# ----------remove stop words
stop_words = nltk.corpus.stopwords.words('english')

def remove_stop_words(corpus):
    result = []
    corp = corpus.split(' ')
    result = [w for w in corp if w not in stop_words]
    result = " ".join(result)

    return result

data['text'] = data['text'].apply(lambda x: remove_stop_words(x))

# ----------lemmatize
lemma = nltk.WordNetLemmatizer()
data['text'] = data.text.apply(lambda x: lemma.lemmatize(x))

# ----------tokenize
from transformers import AutoTokenizer
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
data['text'] = data['text'].apply(lambda x:tokenizer.tokenize(x, truncation=True, return_tensors="pt"))
print(data.head())

# data['sent_token'] = data['text'].apply(lambda x: sent_tokenize(x))
# print(data.sent_token.head())

# ----------savedata
df = data.filter(['text', 'discourse_text', 'label'])
df.to_csv('clean_train.csv')


