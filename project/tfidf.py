from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import pymorphy2
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import time

vectorizer = TfidfVectorizer()
morph = pymorphy2.MorphAnalyzer()

df_processed = pd.read_csv('all.csv', index_col=0)
df = pd.read_csv('data.csv', index_col=0)


def docterm(corpus):
    X = vectorizer.fit_transform(corpus)
    return X


def query_vetr(query):
    query_lemm = ''
    for word in query.split():
        if word not in stopwords.words('russian'):
            lemm = morph.parse(word)[0].normal_form
            query_lemm += lemm + ' '
    query_vec = vectorizer.transform([query_lemm])
    return query_vec


def main(q):
    startTime = time.time()
    matrx = docterm(df_processed['all'].values.astype('U'))
    query = query_vetr(q)

    cos_sim = cosine_similarity(matrx, query)

    indx_val = np.argsort(cos_sim, axis=0)
    sorted_texts = []
    for i in range(len(indx_val) - 1, len(indx_val) - 12, -1):
        indx = indx_val[i]
        sorted_texts.append({'question': df.loc[[indx[0]], ['question']].values[0][0],
                             'comment': df.loc[[indx[0]], ['comment']].values[0][0],
                             'answer': df.loc[[indx[0]], ['answers']].values[0][0]})
    executionTime = (time.time() - startTime)
    return sorted_texts, executionTime
