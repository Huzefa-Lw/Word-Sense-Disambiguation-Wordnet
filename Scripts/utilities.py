import re
import pandas as pd
import numpy as np
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn


def preprocess_mail_body(body_dict):
    """
    Convert the input email body dictionary to string. Non word characters, Emails and URLs are removed during
    preprocessing.

    :param body_dict: dict
        Email body dictionary
    :return: string
        Preprocessed email.
    """
    mail_body =  body_dict['Mail_1']

    if 'Mail_2' in body_dict.keys():
        mail_body = mail_body + ' ' + body_dict['Mail_2']

    pattern_1 = re.compile(r'[\w\.-_]+@[\w\.-_]+')

    text = pattern_1.sub('', mail_body)

    pattern_2 = re.compile(r'(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+')

    text = pattern_2.sub('', text)

    text = ' '.join(word_tokenize(text))

    # pattern_3 = re.compile(r'[^A-Za-z\s]*')
    #
    # text = pattern_3.sub('', text)

    text = ' '.join(x for x in text.split() if not any(c.isdigit() for c in x))

    text = text.lower()

    return text


def preprocess_text(text):
    """
    Preprocess and POS tag input sentence.

    :param text: string
        Sentence to preprocess.
    :return: list
        List of POS tagged tokens.
    """
    # pattern_1 = re.compile(r'[^A-Za-z\s]*')
    # text = pattern_1.sub('', text)

    text = ' '.join(x for x in text.split() if not any(c.isdigit() for c in x))

    text = text.lower()

    text = word_tokenize(text)

    text = pos_tag(text)

    return text


def get_all_senses(word):
    """
    Determine all the senses of the input word by using WordNet.

    :param word: string
        Token for which senses are to be determined.
    :return: list
        List of all the senses of the input word.
    """
    return wn.synsets(word, pos=[wn.NOUN, wn.VERB])


def get_all_hypernyms(sense):
    """
    Determine all the hypernyms (is-a relationaship) for the input synset using WordNet.

    :param sense: wn.Syset
        Sense of a word for which all the hypernyms are to be determined.
    :return: list
        List of all hypernyms of the input synset.
    """
    return sense.hypernyms()


def sign(a, b):
    """
    1 if a> b else -1
    :param a: int
    :param b: int
    :return: int
    """
    return 1 if a >= b else -1


def compute_feature_contribution(njk, nj, nk, N):
    """
    Compute feature importance for a category using Chi-Square Statistic. Here features are individual tokens
    in the vocabulary.

    :param njk: int
        Count of term j in category k
    :param nj: int
        Total count of term j across all categories
    :param nk: int
        Total count of all terms in category k
    :param N: int
        Tatal count of all the terms across all the categories
    :return: float
        Feature importance of term j for category k
    """
    fjk = njk / N
    fjfk = nj * nk / N ** 2

    X2 = (fjk - fjfk) ** 2 / (fjfk) * sign(fjk, fjfk)

    return X2


def convert_to_tfidf(df):
    """
    Convert the count term document matrix to tfidf matrix.

    :param df: pd.Dataframe
        Term document count dataframe where documents are in rows and terms are in columns.
    :return: pd.Dataframe
        TfIdf transformed input dataframe.
    """
    df = pd.DataFrame(df.values / df.sum(axis=1).values.reshape(-1, 1), columns=df.columns, index=df.index)
    df = pd.DataFrame(df.values * np.log(df.shape[0] / (df > 0).sum()).values.reshape(1, -1), columns=df.columns,
                      index=df.index)

    return df


def generate_weighted_vector_cat(df_term_cat, df_chi_sq, k):
    """
    Filter the top K terms for a category along with their weights.

    :param df_term_cat: pd.Dataframe
        Term category TfIdf dataframe where categories are in rows and terms are in columns.
    :param df_chi_sq: pd.Dataframe
        Dataframe containing importances of a term for a category.
    :param k:  int
        Number of features to be fitered for each category.
    :return: dict
        Dictionary containing top terms and their weights for each category.
    """
    cat_vec = {}

    for cat in df_term_cat.index:
        top_k_cats = df_chi_sq[cat].sort_values(ascending=False)[:k].index.tolist()
        cat_vec[cat] = (top_k_cats, df_term_cat.loc[cat, top_k_cats].values.tolist())

    return cat_vec

