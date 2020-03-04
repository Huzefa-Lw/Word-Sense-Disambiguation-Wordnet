from nltk.corpus import reuters, stopwords
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk.stem import PorterStemmer
from nltk import word_tokenize
import pandas as pd
import numpy as np
import re
import spacy
nlp = spacy.load('en_core_web_sm')
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


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

    return text


def compute_overlap_score(synset, sentence):
    gloss = set(word_tokenize(synset.definition()))

    for i in synset.examples():
        gloss.union(i)

    gloss = gloss.difference(functionwords)

    if isinstance(sentence, str):
        sentence = set(sentence.split(" "))

    elif isinstance(sentence, list):
        sentence = set(sentence)

    elif isinstance(sentence, set):
        pass

    else:
        return

    sentence = sentence.difference(functionwords)

    return len(gloss.intersection(sentence))


def lesk(word, sentence):
    best_sense = None
    max_overlap = 0
    word = wn.morphy(word) if wn.morphy(word) is not None else word

    for sense in wn.synsets(word):
        overlap = compute_overlap_score(sense, sentence)

        for hyponym in sense.hyponyms():
            overlap += compute_overlap_score(hyponym, sentence)

        for hypernym in sense.hypernyms():
            overlap += compute_overlap_score(hypernym, sentence)

        for meronym in sense.part_meronyms():
            overlap += compute_overlap_score(meronym, sentence)

        for meronym in sense.substance_meronyms():
            overlap += compute_overlap_score(meronym, sentence)

        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense

    return best_sense


def get_all_senses(word):
    return wn.synsets(word)


def get_all_hypernyms(sense):
    return sense.hypernyms()


def merge_terms(pos_text):
    text = []

    for pos_word in pos_text:
        if 'NN' in pos_word[1] or 'JJ' in pos_word[1] or 'VB' in pos_word[1]:
            text.append(pos_word[0].lower())

    text = ' '.join(text)

    doc_text = nlp(text)

    text = [x.lemma_ for x in doc_text]

    # --------- Using Lesk Algorithm to find best sense of every word ------------

    word_sense_dict = {x: lesk(x, text) for x in text}

    text = np.array(text)

    # ------------- Merging terms with commons meanings --------------------------

    for i in range(len(text)-1):
        if word_sense_dict[text[i]] is not None:
            for j in range(i+1, len(text)):
                if text[i] != text[j]:
                    if word_sense_dict[text[i]] in get_all_senses(text[j]):
                        # print(f'Merged...{text[i]} and {text[j]}')
                        text = np.where(text == text[j], text[i], text)

    # ------------------- Merging terms with Hypernyms ---------------------------

    for i in range(len(text)-1):
        try:
            if word_sense_dict[text[i]] is not None:
                for j in range(i+1, len(text)):
                    try:
                        if (text[i] != text[j]) and (word_sense_dict[text[j]] is not None):
                            word_sense_i = word_sense_dict[text[i]]
                            word_sense_j = word_sense_dict[text[j]]

                            hypernyms_i = get_all_hypernyms(word_sense_i)
                            hypernyms_j = get_all_hypernyms(word_sense_j)

                            if word_sense_i in hypernyms_j:
                                # print(f'{text[i]} is a Hypernym of {text[j]}')
                                text = np.where(text == text[j], text[i], text)

                            elif word_sense_j in hypernyms_i:
                                # print(f'{text[j]} is a Hypernym of {text[i]}')
                                text = np.where(text == text[i], text[j], text)

                            elif len(set(hypernyms_i).intersection(set(hypernyms_j)))>0:
                                hypernym_lemma = set(hypernyms_i).intersection(set(hypernyms_j)
                                                                               ).pop().lemmas()[0].name()

                                # print(f'{text[i]} and {text[j]} have common hypernyms: {hypernym_lemma}')

                                text = np.where((text == text[j]) | (text == text[i]), hypernym_lemma, text)
                    except KeyError as ke:
                        continue

        except KeyError as ke:
            continue

    return ' '.join(text)





def build_tfidf_matrix(corpus):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus)

    feature_names = vectorizer.get_feature_names()

    df_matrix = pd.DataFrame(matrix.todense(), columns=feature_names)

    print(df_matrix)


if __name__ == '__main__':
    train_documents, train_categories, test_documents, test_categories = load_data()

    df_train = pd.DataFrame({'Document': train_documents, 'Category': [x[0] for x in train_categories]})
    df_test = pd.DataFrame({'Document': test_documents, 'Category': [x[0] for x in test_categories]})

    cats_to_consider = df_train.Category.value_counts().index[:5]

    df_train = df_train.loc[df_train.Category.isin(cats_to_consider)]
    df_test = df_test.loc[df_test.Category.isin(cats_to_consider)]

    df_train = df_train.head(1000)

    print('-'*50, 'Preprocessing Text', '-'*50)

    df_train['pos_text'] = df_train.Document.apply(preprocess_text)

    print('-' * 50, 'Merging Terms', '-' * 50)

    df_train['preprocessed_text'] = df_train.pos_text.apply(merge_terms)

    # build_tfidf_matrix(df_train.head().Document)

    tfidf1 = TfidfVectorizer()
    print(tfidf1.fit_transform(df_train.Document).todense().shape)

    tfidf2 = TfidfVectorizer()
    print(tfidf2.fit_transform(df_train.preprocessed_text).todense().shape)

