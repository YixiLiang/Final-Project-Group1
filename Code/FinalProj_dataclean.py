import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re
import nltk
import os
from unidecode import unidecode
import contractions


# ------------------------preprocessing------------------------
# ----------load data
# dataset = pd.read_csv('train.csv')
dataset = pd.read_csv('../train.csv.zip')
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
print(f'Text contains non-ASCII characters, for example: {data.discourse_text.iloc[144290].encode()}')
def clean_text(x):
    # ----------convert text to ASCII form
    x = unidecode(x)
    # ----------lowercase
    x = x.lower()
    # remove contraction
    x = contractions.fix(x)
    # ----------remove url(not-well formatted)
    # match_url = re.compile(r'http\S+')
    match_url = re.compile(r'https?://(www\.)?([-_\w\s\.\/]*)')
    x = re.sub(match_url, "", x)
    # ----------remove consecutive letter 3ormore
    x = re.sub(r'([^\W\d_])\1{2,}', r'\1\1', x)
    # ----------remove parenthesis
    # x = re.sub(re.compile(r'\([^\)]*\)'), "", x)
    x = re.sub(re.compile(r'[()]'), "", x)
    return x

data['text'] = data['discourse_text'].astype(str).apply(clean_text)

print(f'after cleaning: {data.text.iloc[144290]}')
# ----------remove stop words
stop_words = nltk.corpus.stopwords.words('english')

def remove_stop_words(corpus):
    result = []
    corp = corpus.split(' ')
    result = [w for w in corp if w not in stop_words]
    result = " ".join(result).strip()

    return result

data['text'] = data['text'].apply(remove_stop_words)

# ----------lemmatize
lemma = nltk.WordNetLemmatizer()
data['text'] = data.text.apply(lemma.lemmatize)

# ----------remove non-sense text
print('Dataset has text with no sense:')
print(data[data.text==""])
data = data[data.text!=""].reset_index()

# ----------savedata
df = data.filter(['text', 'discourse_text', 'discourse_type', 'label'])
path = os.getcwd()
df.to_csv(f'{path}/clean_train.csv')


