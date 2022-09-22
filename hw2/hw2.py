from sklearn.feature_extraction.text import TfidfVectorizer
import pymorphy2
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np
import argparse

vectorizer = TfidfVectorizer()
morph = pymorphy2.MorphAnalyzer()

parser = argparse.ArgumentParser()
parser.add_argument('file')
parser.add_argument('query')
args = parser.parse_args()


def load(file):
    with open(file, encoding='utf-8') as f:
        j_file = json.load(f)
    return j_file


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


def main(f, q):
    file = load(f)
    doc_names = []
    doc_texts = []
    for key, value in file.items():
        doc_names.append(key)
        doc_texts.append(value)

    matrx = docterm(doc_texts)
    query = query_vetr(q)

    cos_sim = cosine_similarity(matrx, query)

    indx_val = np.argsort(cos_sim, axis=0)

    sorted_names = []
    for i in range(1, 165):
        indx = np.where(indx_val == i)[0][0]
        sorted_names.append(doc_names[indx])

    print(sorted_names)


if __name__ == "__main__":
    main(f=args.file, q=args.query)
