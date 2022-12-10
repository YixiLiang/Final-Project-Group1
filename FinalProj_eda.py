import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AdamW, BertForSequenceClassification, BertTokenizer, DataCollatorWithPadding
import torch
from torch import nn
import seaborn as sns
import re
import nltk
from collections import Counter
import contractions
import spacy
from wordcloud import WordCloud
from rake_nltk import Rake

# ------------------------------Preprocessing------------------------------
dataset = pd.read_csv('Dataset/clean_train.csv', index_col=[0])
data = dataset.filter(['text', 'discourse_type', 'label'])

# ----------word tokenization
def nltk_tokenization(text, remove_punc=False):
    # remove contraction
    text = contractions.fix(text)
    # remove punc
    if remove_punc:
        text = re.sub(r'[^\w\s]','',text)
    text = nltk.word_tokenize(text)

    return text

data['token'] = data.text.apply(nltk_tokenization, remove_punc=True)


# ----------words frequency
print('number of samples with each label:')
print(data[['discourse_type', 'label']].value_counts())


# Lead(label 0)
df_lead = data.token[data.label==0]
# Position(label 1)
df_pos = data.token[data.label==1]
# Claim(label 3)
df_claim = data.token[data.label==3]
# Counter Claim(6)
df_counter = data.token[data.label==6]
# Rebuttal(5)
df_rebut = data.token[data.label==5]
# Evidence(2)
df_evidence = data.token[data.label==2]
# Concluding Statement(4)
df_conclude = data.token[data.label==4]

def get_bag_of_words(list_of_words, counter):
    counter.update(list_of_words)

def get_most_n(df, n):
    ct = Counter()
    df.apply(get_bag_of_words, counter=ct)
    return ct.most_common(n)


# lead = pd.DataFrame(get_most_n(df_lead, 100)[6:], columns=['word', 'frequency'])
# position = pd.DataFrame(get_most_n(df_pos, 100)[5:],columns=['word', 'frequency'])
# claim = pd.DataFrame(get_most_n(df_claim, 100)[4:], columns=['word', 'frequency'])
# counterc = pd.DataFrame(get_most_n(df_counter, 100)[5:], columns=['word', 'frequency'])
# rebut = pd.DataFrame(get_most_n(df_rebut, 100)[3:], columns=['word', 'frequency'])
# evidence = pd.DataFrame(get_most_n(df_evidence, 100)[3:], columns=['word', 'frequency'])
# conclusion = pd.DataFrame(get_most_n(df_conclude, 100)[4:], columns=['word', 'frequency'])
lead = dict(get_most_n(df_lead, 80)[6:])
position = dict(get_most_n(df_pos, 80)[5:])
claim = dict(get_most_n(df_claim, 80)[4:])
counterc = dict(get_most_n(df_counter, 80)[5:])
rebut = dict(get_most_n(df_rebut, 80)[3:])
evidence = dict(get_most_n(df_evidence, 80)[5:])
conclusion = dict(get_most_n(df_conclude, 80)[4:])

# ----------phrase detection
# nlp = spacy.load('en_core_web_sm')
#
# def phrase_tokenization(text):
#     spacy_text = nlp(text)
#     return [chunk for chunk in spacy_text.noun_chunks]
#
# data['word_chunk'] = data.text.apply(phrase_tokenization)

def phrase_tokenization(text):
    r = Rake()
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()

data['word_chunk'] = dataset.discourse_text.apply(phrase_tokenization)


# ----------phrase frequency

# Lead(label 0)
df_lead_ph = data.word_chunk[data.label==0]
# Position(label 1)
df_pos_ph = data.word_chunk[data.label==1]
# Claim(label 3)
df_claim_ph = data.word_chunk[data.label==3]
# Counter Claim(6)
df_counter_ph = data.word_chunk[data.label==6]
# Rebuttal(5)
df_rebut_ph = data.word_chunk[data.label==5]
# Evidence(2)
df_evidence_ph = data.word_chunk[data.label==2]
# Concluding Statement(4)
df_conclude_ph = data.word_chunk[data.label==4]


def get_bag_of_phrase(list_of_words, counter, phrase_len):
    phrase = [p for p in list_of_words if len(p.split())>=phrase_len]
    counter.update(phrase)


def get_most_n_ph(df, n, phrase_len):
    ct = Counter()
    df.apply(get_bag_of_phrase, counter=ct, phrase_len=phrase_len)
    return ct.most_common(n)


lead_ph = dict(get_most_n_ph(df_lead_ph, 80, 2)[3:])
position_ph = dict(get_most_n_ph(df_pos_ph, 80, 2)[3:])
claim_ph = dict(get_most_n_ph(df_claim_ph, 80, 2)[4:])
counterc_ph = dict(get_most_n_ph(df_counter_ph, 80, 2)[3:])
rebut_ph = dict(get_most_n_ph(df_rebut_ph, 80, 2)[3:])
evidence_ph = dict(get_most_n_ph(df_evidence_ph, 80, 2)[3:])
conclusion_ph = dict(get_most_n_ph(df_conclude_ph, 80, 2)[3:])

# ------------------------------Visualization------------------------------
# ----------target distribution
temp_df = pd.DataFrame(data['discourse_type'].value_counts())
fig = plt.figure(figsize=(14, 6))
plt.barh(temp_df.index, temp_df.discourse_type)
plt.title('Discourse Element Type Distribution')
plt.xlabel('count')
plt.show()

# ----------frequent word in dataset
df_all_word = pd.DataFrame(get_most_n(data.token, 15), columns=['word', 'frequency'])
fig = plt.figure(figsize=(10, 6))
sns.barplot(data=df_all_word, x='frequency', y='word', color='steelblue')
plt.title('Top 15 Frequency Words')
plt.show()

# # ----------frequent phrase in dataset
df_all_phrase = pd.DataFrame(get_most_n_ph(data.word_chunk, 20, 2), columns=['word', 'frequency'])
fig = plt.figure(figsize=(16, 6))
sns.barplot(data=df_all_phrase, x='frequency', y='word', color='steelblue')
plt.title('Top 15 Frequency Phrase')
plt.show()

# ----------word cloud for each element
word_all_df = [lead, position, claim, counterc, rebut, evidence, conclusion]
phrase_all_df = [lead_ph, position_ph, claim_ph, counterc_ph, rebut_ph, evidence_ph, conclusion_ph]
df_label = ['Lead Statement', 'Position Statement', 'Claim', 'Counterclaim',
            'Rebuttal Statement', 'Evidence', 'Conclusion']

# for z in range(len(df_label)):
#     fig, ax = plt.subplots(1, 2, figsize=(12, 12))
fig, ax = plt.subplots(2, 2, figsize=(12, 12))
z = 0
for i in range(2):
    for j in range(2):
        # w = " ".join(word_all_df[z].word.values)
        wordcloud = WordCloud(width=900, height=900,
                              background_color='white',
                              min_font_size=10, random_state=12).generate_from_frequencies(word_all_df[z])

        plt.figure(figsize=(8, 8), facecolor=None)
        ax[i, j].imshow(wordcloud)
        ax[i, j].set_title(f'Most Frequent Word in {df_label[z]}', fontsize=15)
        z+=1
plt.tight_layout(pad=0)
plt.show()

fig, ax = plt.subplots(3, 1, figsize=(6, 12))
z = 4
for i in range(3):
        # w = " ".join(word_all_df[z].word.values)
        wordcloud = WordCloud(width=900, height=900,
                              background_color='white',
                              min_font_size=10, random_state=12).generate_from_frequencies(word_all_df[z])

        plt.figure(figsize=(8, 8), facecolor=None)
        ax[i].imshow(wordcloud)
        ax[i].set_title(f'Most Frequent Word in {df_label[z]}', fontsize=15)
        z+=1
plt.tight_layout(pad=0)
plt.show()


# ----------phrase cloud for each element



fig, ax = plt.subplots(2, 2, figsize=(12, 12))
z = 0
for i in range(2):
    for j in range(2):
        wordcloud = WordCloud(width=900, height=900,
                              background_color='white',
                              min_font_size=10, random_state=12).generate_from_frequencies(phrase_all_df[z])

        plt.figure(figsize=(8, 8), facecolor=None)
        ax[i, j].imshow(wordcloud, interpolation='bilinear')
        ax[i, j].set_title(f'Most Frequent Phrase in {df_label[z]}', fontsize=15)
        z+=1
plt.tight_layout(pad=0)
plt.show()

fig, ax = plt.subplots(3, 1, figsize=(6, 12))
z = 4
for i in range(3):
        wordcloud = WordCloud(width=900, height=900,
                              background_color='white',
                              min_font_size=10, random_state=12).generate_from_frequencies(phrase_all_df[z])

        plt.figure(figsize=(8, 8), facecolor=None)
        ax[i].imshow(wordcloud)
        ax[i].set_title(f'Most Frequent Phrase in {df_label[z]}', fontsize=15)
        z+=1
plt.tight_layout(pad=0)
plt.show()

# ------------------------------Modeling------------------------------
# #============== train test split #==============
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(data.word_chunk, data.label, test_size=0.75, random_state=42)
#
# def get_corpus(text, corpus):
#     """
#
#     :param text:
#     :param corpus:
#     :return:
#     """
#     corpus.append(" ".join(text))
#
# corpus_train=[]
# x_train.apply(lambda x:corpus_train.append(" ".join(x)))
# corpus_test = []
# x_test.apply(lambda x:corpus_test.append(" ".join(x)))
#
# from sklearn.feature_extraction.text import CountVectorizer
# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(corpus_train).toarray()
# X_test_counts = count_vect.transform(corpus_test).toarray()
#
# #============== TFIDF #==============
# from sklearn.feature_extraction.text import TfidfTransformer
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# X_test_tfidf = tfidf_transformer.transform(X_test_counts)
#
# #============== Bayes #==============
# from sklearn.naive_bayes import MultinomialNB
# model_bayes = MultinomialNB()
# clf = model_bayes.fit(X_train_tfidf, y_train)
#
# model_bayes.score(X_test_tfidf, y_test)
# predicted = clf.predict(X_test_tfidf)
# print('Bayes score:', model_bayes.score(X_test_tfidf, y_test))
#
# #============== LogisticR #==============
# from sklearn.linear_model import LogisticRegression
# model_logi = LogisticRegression()
# lg = model_logi.fit(X_train_tfidf, y_train)
# model_logi.score(X_test_tfidf, y_test)
# predicted_lg = lg.predict(X_test_tfidf)
# print('Logistic Regression score:', model_logi.score(X_test_tfidf, y_test))