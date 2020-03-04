import re
from nltk import word_tokenize
from nltk import pos_tag
from collections import Counter
import numpy as np
import pandas as pd
from nltk.corpus import reuters
from nltk.corpus import wordnet as wn

DOCUMENT_COUNT = {}

functionwords = ['about', 'across', 'against', 'along', 'around', 'at',
                 'behind', 'beside', 'besides', 'by', 'despite', 'down',
                 'during', 'for', 'from', 'in', 'inside', 'into', 'near', 'of',
                 'off', 'on', 'onto', 'over', 'through', 'to', 'toward',
                 'with', 'within', 'without', 'anything', 'everything',
                 'anyone', 'everyone', 'ones', 'such', 'it', 'itself',
                 'something', 'nothing', 'someone', 'the', 'some', 'this',
                 'that', 'every', 'all', 'both', 'one', 'first', 'other',
                 'next', 'many', 'much', 'more', 'most', 'several', 'no', 'a',
                 'an', 'any', 'each', 'no', 'half', 'twice', 'two', 'second',
                 'another', 'last', 'few', 'little', 'less', 'least', 'own',
                 'and', 'but', 'after', 'when', 'as', 'because', 'if', 'what',
                 'where', 'which', 'how', 'than', 'or', 'so', 'before', 'since',
                 'while', 'although', 'though', 'who', 'whose', 'can', 'may',
                 'will', 'shall', 'could', 'be', 'do', 'have', 'might', 'would',
                 'should', 'must', 'here', 'there', 'now', 'then', 'always',
                 'never', 'sometimes', 'usually', 'often', 'therefore',
                 'however', 'besides', 'moreover', 'though', 'otherwise',
                 'else', 'instead', 'anyway', 'incidentally', 'meanwhile']


def load_data():
    train_documents, train_categories = zip(
        *[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if i.startswith('training/')])
    test_documents, test_categories = zip(
        *[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if i.startswith('test/')])

    return train_documents, train_categories, test_documents, test_categories


def pos_tag_text(text):
    return pos_tag(text)


def preprocess_text(text):
    pattern_1 = re.compile(r'[^A-Za-z\s]*')
    text = pattern_1.sub('', text)

    text = text.lower()

    text = word_tokenize(text)

    text = pos_tag_text(text)

    text = [wn.morphy(x[0]) for x in text if (x[0] not in functionwords) and ('NN' in x[1] or 'JJ' in x[1] or 'VB' in x[1])]

    return text


def compute_term_frequency(text):
    text = preprocess_text(text)
    c = Counter(text)
    total_terms = sum(c.values())

    for key in c:
        c[key] = c[key] / total_terms
        DOCUMENT_COUNT[key] = DOCUMENT_COUNT[key] + 1 if key in DOCUMENT_COUNT else 1

    return dict(c)


def compute_idf(term_freq, N):
    idf = {}
    for key in term_freq:
        n = DOCUMENT_COUNT[key]
        idf[key] = term_freq[key] * np.log(N/n)

    return idf


if __name__ == '__main__':
    train_documents, train_categories, test_documents, test_categories = load_data()

    df_train = pd.DataFrame({'Document': train_documents, 'Category': [x[0] for x in train_categories]})
    df_test = pd.DataFrame({'Document': test_documents, 'Category': [x[0] for x in test_categories]})

    cats_to_consider = df_train.Category.value_counts().index[:5]

    df_train = df_train.loc[df_train.Category.isin(cats_to_consider)]
    df_test = df_test.loc[df_test.Category.isin(cats_to_consider)]

    df_train['term_frequency'] = df_train.Document.apply(compute_term_frequency)

    N = df_train.shape[0]

    df_train['idf'] = df_train.term_frequency.apply(compute_idf, N=N)

    print(df_train.head())