# Logistic, LSA input Logistic, Naive Bayes
from nltk import word_tokenize
import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD

import warnings
warnings.filterwarnings('ignore')

# load the data
df_train = pd.read_csv('train_balanced.csv')
df_test = pd.read_csv('test_balanced.csv')

### Lower
body_text_list = df_train['text'].tolist()
body_text_list_t = df_test['text'].tolist()
text_lower = [str(i).lower() for i in body_text_list]
text_lower_t = [str(i).lower() for i in body_text_list_t]

### Punctuations
text_no_punc = [re.sub(r'[^\w\s]', '', i) for i in text_lower]
text_no_punc_t = [re.sub(r'[^\w\s]', '', i) for i in text_lower_t]

### Stem
def stem(phrase):
    return ' '.join([re.findall('^(.*ss|.*?)(s)?$', word)
                     [0][0].strip("'") for word in phrase.lower().split()])
body_text_list_stemmed = [stem(i) for i in text_no_punc]
body_text_list_stemmed_t = [stem(i) for i in text_no_punc_t]

### Token
token_words_list = []
for i in body_text_list_stemmed:
    w = word_tokenize(i)
    token_words_list.append(w)
token_words_list_t = []
for i in body_text_list_stemmed_t:
    w = word_tokenize(i)
    token_words_list_t.append(w)

### Stopwords
stopword = stopwords.words('english')
list_no_stop = []
list_no_stop_t = []
for i in token_words_list:
    s = []
    for j in i:
        if j not in stopword:
            s.append(j)
    list_no_stop.append(s)
for i in token_words_list_t:
    s = []
    for j in i:
        if j not in stopword:
            s.append(j)
    list_no_stop_t.append(s)

### Lemma
lemmatizer = WordNetLemmatizer()
lemmatized_list = []
for i in list_no_stop:
    s = []
    for ii in i:
        s.append(lemmatizer.lemmatize(ii))
    lemmatized_list.append([lemmatizer.lemmatize(j) for j in i])
lemmatized_list_t = []
for i in list_no_stop_t:
    s = []
    for ii in i:
        s.append(lemmatizer.lemmatize(ii))
    lemmatized_list_t.append([lemmatizer.lemmatize(j) for j in i])

### Clean text list and save
list_text = [' '.join(i) for i in lemmatized_list]
list_text_t = [' '.join(i) for i in lemmatized_list_t]
df_train_new = pd.DataFrame({'text': list_text,
                         'label': df_train['label']})
df_test_new = pd.DataFrame({'text': list_text_t,
                         'label': df_test['label']})

### Vectorizer TFIDF
tfidf_vector = TfidfVectorizer()
tfidf_vector.fit(df_train_new['text'])
X_train_tfidf = tfidf_vector.transform(df_train_new['text'])
X_test_tfidf  = tfidf_vector.transform(df_test_new['text'])



### Logistic no LSA
clf_lo = LogisticRegression().fit(X_train_tfidf, df_train_new['label'])
predict_lo = clf_lo.predict(X_test_tfidf)
report_lo = classification_report(df_test_new['label'], predict_lo,
                                          target_names = sorted([str(i) for i in df_train_new['label'].unique()]),
                                          output_dict=True)
### Logistic + LSA
lsa = TruncatedSVD(n_components = 500, n_iter = 100)
lsa.fit(X_train_tfidf)
X_train_lsa = lsa.transform(X_train_tfidf)
X_test_lsa = lsa.transform(X_test_tfidf)
clf_l = LogisticRegression().fit(X_train_lsa, df_train_new['label'])
predict_l = clf_l.predict(X_test_lsa)
report_l_lsa = classification_report(df_test_new['label'], predict_l,
                                             target_names = sorted([str(i) for i in df_train_new['label'].unique()]),
                                             output_dict=True)
### Naive Bayes
clf_n = MultinomialNB().fit(X_train_tfidf, df_train_new['label'])
predict_n = clf_n.predict(X_test_tfidf)
report_nb = classification_report(df_test_new['label'], predict_n,
                                          target_names = sorted([str(i) for i in df_train_new['label'].unique()]),
                                          output_dict=True)

# da = pd.DataFrame({'text': df_test_new.text, 'label': df_test_new['label'],'log_prediction': predict_lo})
# print(da.head())
# print()
# db = pd.DataFrame({'text': df_test_new.text, 'label': df_test_new['label'],'log_lsa_prediction': predict_l})
# print(db.head())
# print()
# dc = pd.DataFrame({'text': df_test_new.text, 'label': df_test_new['label'],'bayes_prediction': predict_n})
# print(dc.head())
# print()

df_lo= pd.DataFrame(report_lo).transpose()
print(df_lo)
print()
df_l_lsa = pd.DataFrame(report_l_lsa).transpose()
print(df_l_lsa)
print()
df_nb = pd.DataFrame(report_nb).transpose()
print(df_nb)