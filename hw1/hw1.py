import os
import pymorphy2
from nltk.corpus import stopwords
import string
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import Counter

vectorizer = CountVectorizer(analyzer='word')
morph = pymorphy2.MorphAnalyzer()

mon = [['моника', 'мона'], 'Моника']
rach = [['рэйчел', 'рейч'], 'Рэйчел']
chand = [['чендлера', 'чэндлера', 'чен'], 'Чендлер']
phoebe = [['фиби', 'фибс'], 'Фиби']
ross = [['росс'], 'Росс']
joey = [['джоуя', 'джой', 'джо'], 'Джоуи']

curr_dir = os.getcwd()
friends_dir = os.path.join(curr_dir, 'friends-data')


def clean_text(episode):
    episode = episode.translate(str.maketrans('', '', string.punctuation + '–—«»…'))
    episode = episode.lower()
    lemm_ep = ''
    for word in episode.split():
        if word not in stopwords.words('russian'):
            lemm = morph.parse(word)[0].normal_form
            lemm_ep += lemm + ' '
    return lemm_ep


def text_to_matrix(corpus):
    texts = list(corpus.values())
    matrix = vectorizer.fit_transform(texts)
    return matrix


def text_to_dict(corpus):
    index = defaultdict(list)
    for key, value in corpus.items():
        freq = Counter(value.split())
        for k, v in freq.items():
            index[k].append([v, key])
    return dict(index)


def name_freq_mtrx(names, freq_dict):
    total = 0
    for i in names:
        total += freq_dict[i]
    return total


def name_freq_dict(names, dict_index):
    total = 0
    for i in names:
        for ep in dict_index[i]:
            total += ep[0]
    return total


def main():
    ep_list = {}

    for root, dirs, files in os.walk(friends_dir):
        pbar = tqdm(files)
        for name in pbar:
            pbar.set_description("Processing %s" % name)
            with open(os.path.join(root, name), encoding='utf-8-sig') as f:
                episode = f.read().replace('\n\n', '\n')
                ep_list[name] = clean_text(episode)

    print('Индексирование с помощью матрицы\n')

    mtrx = text_to_matrix(ep_list)
    sum_words = mtrx.sum(axis=0)
    words_freq_mtrx = {word: sum_words[0, idx] for word, idx in vectorizer.vocabulary_.items()}
    words_freq_mtrx = dict(sorted(words_freq_mtrx.items(), key=lambda item: item[1], reverse=True))
    print('Самое частотное слово: ', list(words_freq_mtrx.items())[0][0])
    print('Самое редкое слово: ', list(words_freq_mtrx.items())[-1][0])

    names = [mon, rach, chand, phoebe, ross, joey]
    name_freq = []
    for name in names:
        name_freq.append([name[1], name_freq_mtrx(name[0], words_freq_mtrx)])
    name_freq = sorted(name_freq, key=lambda x: x[1], reverse=True)
    print('Самый популярный персонаж:', name_freq[0][0], ',количество употреблений: ', name_freq[0][1])

    print('\nИндексирование с помощью словаря\n')

    dict_indx = text_to_dict(ep_list)
    words_freq = []
    for key, val in dict_indx.items():
        freq = 0
        for ep in val:
            freq += ep[0]
        words_freq.append((key, freq))
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    print('Самое частотное слово: ', words_freq[0][0])
    print('Самое редкое слово: ', words_freq[-1][0])

    all_docs = []
    for key, value in dict_indx.items():
        if len(value) == 165:
            all_docs.append(key)
    print('во всех документах встречаются:\n', all_docs)

    name_freq = []
    for name in names:
        name_freq.append([name[1], name_freq_dict(name[0], dict_indx)])
    name_freq = sorted(name_freq, key=lambda x: x[1], reverse=True)
    print('Самый популярный персонаж:', name_freq[0][0], ',количество употреблений: ', name_freq[0][1])


if __name__ == "__main__":
    main()
