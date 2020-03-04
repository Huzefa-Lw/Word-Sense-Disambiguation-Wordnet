"""
This module is implemented using the below mentioned research papers:
1.) Using WordNet for Text Categorization - http://iajit.org/PDF/vol.5%2Cno.1/3-37.pdf
2.) Efficient Email Classification Approach Based On Semantic Methods - https://www.sciencedirect.com/science/article/pii/S2090447918300455
3.) Semantic Feature Selection Using WordNet - https://ieeexplore.ieee.org/document/1410799
4.) Measuring Similarity Between Sentences - https://www.coursehero.com/file/31214030/WordNetDotNet-Semantic-Similaritypdf/
"""

import sqlite3
import json
import spacy
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.exceptions import NotFittedError
from utilities import preprocess_mail_body, preprocess_text, get_all_hypernyms, get_all_senses
from utilities import compute_feature_contribution, convert_to_tfidf, generate_weighted_vector_cat
nlp = spacy.load('en_core_web_sm')


class WordNetFeatures:
    """
    Computes most significant features to classify each category using WordNet and ChiSquare Statistics. WordNet is used
    to extract the semantic information from the text and merge the terms having similar meanings. ChiSquare statistic
    measures the importance of each term for a category.

    :param feature_count: int, optional (default: 800)
        Number of significant terms to represent each category.
    :param function_words: list, optional (default: nltk.corpus.stopwords.words("english"))
        Stopwords to remove during text preprocessing.
    """
    def __init__(self, feature_count=800, function_words=stopwords.words('english')):
        self.feature_count = feature_count
        self.function_words = function_words
        self.df_train = None
        self.cats_to_consider = None
        self.model_is_fit = False
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.cat_encoder = None
        self.df_chi_sq = None
        self.cat_vecs = None

    def __compute_overlap_score(self, synset, sentence):
        """
        Compute the overlap score between the glossary of input synset with the input sentence.

        :param synset: wn.Synset
            WordNet defined specific sense of a word.
        :param sentence: string
            The sentence of which the original form of input synset is a token.
        :return: int
            Overlap score between the glossary of input synset with the input sentence.
        """
        gloss = set(word_tokenize(synset.definition()))

        for i in synset.examples():
            gloss = gloss.union(i)

        gloss = gloss.difference(self.function_words)

        if isinstance(sentence, str):
            sentence = set(sentence.split(" "))

        elif isinstance(sentence, list):
            sentence = set(sentence)

        elif isinstance(sentence, set):
            pass

        else:
            return

        sentence = sentence.difference(self.function_words)

        return len(gloss.intersection(sentence))

    def __lesk(self, word, sentence):
        """
        Implementation of Lesk Algorithm. This algorithm is used in Word Sense Disambiguation.

        :param word: string
            Token which is to be disambiguated.
        :param sentence: string
            The sentence of which input word is a token.
        :return: wn.Synset
            The best sense of the input word.
        """
        best_sense = None
        max_overlap = 0
        word = wn.morphy(word) if wn.morphy(word) is not None else word

        for sense in wn.synsets(word):
            overlap = self.__compute_overlap_score(sense, sentence)

            for hyponym in sense.hyponyms():
                overlap += self.__compute_overlap_score(hyponym, sentence)

            for hypernym in sense.hypernyms():
                overlap += self.__compute_overlap_score(hypernym, sentence)

            for meronym in sense.part_meronyms():
                overlap += self.__compute_overlap_score(meronym, sentence)

            for meronym in sense.substance_meronyms():
                overlap += self.__compute_overlap_score(meronym, sentence)

            if overlap > max_overlap:
                max_overlap = overlap
                best_sense = sense

        return best_sense

    def __merge_terms(self, pos_text):
        """
        Merge the tokens with overlapping synset signatures. Hypernym relations between individual token are also
        exploited to increase the vocabulary of each category.

        :param pos_text: dict
            POS tagged input sentence of form {token: pos_tag}
        :return: string
            Sentence with similar terms being merged.
        """
        text = []

        for pos_word in pos_text:
            if ('NN' in pos_word[1] or 'JJ' in pos_word[1] or 'VB' in pos_word[1]) and (pos_word[0]
                                                                                        not in self.function_words):
                text.append(pos_word[0].lower())

        text = ' '.join(text)

        doc_text = nlp(text)

        text = [x.lemma_ for x in doc_text]

        # --------- Using Lesk Algorithm to find best sense of every word ------------

        word_sense_dict = {x: self.__lesk(x, text) for x in text}

        text = np.array(text)

        # ------------- Merging terms with commons meanings --------------------------

        for i in range(len(text) - 1):
            if word_sense_dict[text[i]] is not None:
                for j in range(i + 1, len(text)):
                    if text[i] != text[j]:
                        if word_sense_dict[text[i]] in get_all_senses(text[j]):

                            text = np.where(text == text[j], text[i], text)

                            text = np.array(list(text) + [x.name() for x in word_sense_dict[text[i]].lemmas()])

        # ------------------- Merging terms with Hypernyms ---------------------------

        for i in range(len(text) - 1):
            try:
                if word_sense_dict[text[i]] is not None:
                    for j in range(i + 1, len(text)):
                        try:
                            if (text[i] != text[j]) and (word_sense_dict[text[j]] is not None):
                                word_sense_i = word_sense_dict[text[i]]
                                word_sense_j = word_sense_dict[text[j]]

                                hypernyms_i = get_all_hypernyms(word_sense_i)
                                hypernyms_j = get_all_hypernyms(word_sense_j)

                                if word_sense_i in hypernyms_j:
                                    text = np.where(text == text[j], text[i], text)

                                    # Appending all the lemmas of the matched hypernym to the original text
                                    text = np.array(list(text) + [x.name() for x in word_sense_i.lemmas()])

                                elif word_sense_j in hypernyms_i:
                                    text = np.where(text == text[i], text[j], text)

                                    text = np.array(list(text) + [x.name() for x in word_sense_j.lemmas()])

                                elif len(set(hypernyms_i).intersection(set(hypernyms_j))) > 0:

                                    hypernym_lemmas = set(hypernyms_i).intersection(set(hypernyms_j)
                                                                                    ).pop().lemmas()

                                    hypernym_lemmas = [x.name() for x in hypernym_lemmas]

                                    text = np.array(list(text) + hypernym_lemmas)

                        except KeyError:
                            continue

            except KeyError:
                continue

        return ' '.join(text)

    def __build_term_document_matrix(self, corpus):
        """
        Build the Count and TfIdf matrix of the input corpus.

        :param corpus: iterable
            An iterable which yields either str, unicode or file objects.
        :return: None
        """
        self.tfidf_vectorizer = TfidfVectorizer()
        self.count_vectorizer = CountVectorizer()

        self.tfidf_vectorizer.fit(corpus)
        self.count_vectorizer.fit(corpus)

        return

    def __transform_text_to_term_frequency(self, text):
        """
        Convert the input text corpus to Count and TfIdf Matrices.

        :param text: iterable
            An iterable which yields either str, unicode or file objects.
        :return: tuple
            A tuple of  dataframes with documents as rows and frequency of individual tokens a columns.
        """
        tfidf_matrix = self.tfidf_vectorizer.transform(text)
        count_matrix = self.count_vectorizer.transform(text)

        tfidf_feature_names = self.tfidf_vectorizer.get_feature_names()
        count_feature_names = self.count_vectorizer.get_feature_names()

        df_tfidf_matrix = pd.DataFrame(tfidf_matrix.todense(), columns=tfidf_feature_names)
        df_count_matrix = pd.DataFrame(count_matrix.todense(), columns=count_feature_names)

        return df_tfidf_matrix, df_count_matrix

    def __build_category_encoder(self, cats_to_encode):
        """
        Build label encoder for input categories.

        :param cats_to_encode: list
            Input categories to be label encoded.
        :return: None
        """
        self.cat_encoder = LabelEncoder()
        self.cat_encoder.fit(cats_to_encode)
        return

    def __compute_features(self, email_vec, return_prediction):
        """
        Compute the cosine similarity between the input email vector and vector of each category. These are used as
        features. This method can also return the predicted class for the input email vector.

        :param email_vec: np.ndarray
            Vector having tfidf frequencies of individual tokens.
        :param return_prediction: bool
            Return the predicted category for the input email vector.
        :return: pd.Series
            Features for the input email vector.
        """
        prediction = None

        features = {}

        best_score = -1

        for cat in self.cat_vecs.keys():
            cat_name = self.cat_encoder.inverse_transform([cat])[0]
            top_terms = self.cat_vecs[cat][0]
            cat_vec = np.array(self.cat_vecs[cat][1]).reshape(1, -1)

            x_cat = email_vec[top_terms].values.reshape(1, -1)

            score = cosine_similarity(x_cat, cat_vec)

            features[cat_name] = score.flatten()[0]

            if score > best_score:
                best_score = score
                prediction = cat_name

        if return_prediction:
            features['prediction'] = prediction

        return pd.Series(features)

    def fit(self, df_train, cats_to_consider):
        """
        Fit the train data to determine significant terms for each category and compute their feature vectors.

        :param df_train: pd.Dataframe
            Training set to fit the model.
        :param cats_to_consider: list
            Categories to consider to fit the model.
        :return: None
        """
        self.df_train = df_train
        self.cats_to_consider = cats_to_consider

        self.df_train = self.df_train.loc[self.df_train.CLASS.isin(self.cats_to_consider)]

        self.__build_category_encoder(self.df_train.CLASS)
        self.df_train.CLASS = self.cat_encoder.transform(self.df_train.CLASS)

        self.df_train['BODY'] = self.df_train.BODY.apply(preprocess_mail_body)

        self.df_train['text'] = self.df_train.SUBJECT + ' ' + self.df_train.BODY
        self.df_train['pos_text'] = self.df_train.text.apply(preprocess_text)

        self.df_train['preprocessed_text'] = self.df_train.pos_text.apply(self.__merge_terms)

        self.__build_term_document_matrix(self.df_train.preprocessed_text)

        _, df_count_matrix = self.__transform_text_to_term_frequency(self.df_train.preprocessed_text)

        df_count_matrix['Category'] = self.df_train.CLASS.values

        df_term_category = df_count_matrix.groupby('Category').sum().T

        n = df_term_category.sum().sum()
        n_j_dot = df_term_category.sum(axis=1)
        n_dot_k = df_term_category.sum(axis=0)

        self.df_chi_sq = pd.DataFrame(index=df_term_category.index, columns=df_term_category.columns)

        for term in df_term_category.index:
            for cat in df_term_category.columns:
                self.df_chi_sq[cat][term] = compute_feature_contribution(df_term_category[cat][term], n_j_dot[term],
                                                                         n_dot_k[cat], n)

        df_term_category = convert_to_tfidf(df_term_category.T)

        self.cat_vecs = generate_weighted_vector_cat(df_term_category, self.df_chi_sq, self.feature_count)

        self.model_is_fit = True

        return

    def get_train_merged_text(self):
        """
        Return complete preprocessed and merged text of the training data. This method is kept to avoid
        preprocessing and merging when transforming the training data.

        :return: pd.Series
            Preprocessed training data.
        """
        if not self.model_is_fit:
            raise NotFittedError("This WordNetFeatures instance is not fitted yet. Call 'fit' with appropriate"
                                 " arguments before using this method.")

        return self.df_train.preprocessed_text

    def transform(self, df_email, return_prediction=False, preprocess_email=True):
        """
        Compute the features for input emails.

        :param df_email: pd.Dataframe, pd.Series
            Emails to compute the features.
        :param return_prediction: bool
            Return the predicted category for an email
        :param preprocess_email: bool
            Set it to False when computing features for training set.
        :return: pd.Dataframe
            Features for input emails.
        """
        if not self.model_is_fit:
            raise NotFittedError("This WordNetFeatures instance is not fitted yet. Call 'fit' with appropriate"
                                 " arguments before using this method.")

        if preprocess_email:
            df_email['BODY'] = df_email.BODY.apply(preprocess_mail_body)
            df_email['text'] = df_email.SUBJECT + ' ' + df_email.BODY
            df_email['pos_text'] = df_email.text.apply(preprocess_text)
            df_email['preprocessed_text'] = df_email.pos_text.apply(self.__merge_terms)

            df_tfidf, _ = self.__transform_text_to_term_frequency(df_email.preprocessed_text)

            df_features = df_tfidf.apply(self.__compute_features, axis=1, return_prediction=return_prediction)

        else:
            df_tfidf, _ = self.__transform_text_to_term_frequency(df_email)

            df_features = df_tfidf.apply(self.__compute_features, axis=1, return_prediction=return_prediction)

        return df_features


# ------------------------Example of how to use this module-------------------------------------

if __name__ == '__main__':

    wordnet_1 = WordNetFeatures()

    def load_json(df):
        for col in df.columns:
            df[col] = df[col].apply(lambda x: json.loads(x))
        return df


    def load_data():
        db_path = '../data/DB.sqlite'
        conn = sqlite3.connect(db_path)

        df_emails = pd.read_sql('SELECT * FROM Data', con=conn).drop('index', axis=1).reset_index(drop=True)
        df_emails = load_json(df_emails)

        return df_emails

    df_email = load_data()

    df_email = df_email.head(1000)
    df_test = df_email.tail(10)

    cats_to_consider = ['1_Class_Add_Invoice', '2_Class_Payment_Query']

    wordnet_1.fit(df_email, cats_to_consider)

    print(wordnet_1.transform(wordnet_1.get_train_merged_text(), return_prediction=True, preprocess_email=False))

    print(wordnet_1.transform(df_test, return_prediction=True, preprocess_email=True))



    
















