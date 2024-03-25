# -*- coding: utf-8 -*-
"""
This file contains the functions needed to build the features matrix for the Fake News Recognizer.
"""

from nltk import word_tokenize, download
import numpy as np
from numpy import ndarray
from nltk.corpus import stopwords
from scipy.sparse import dok_matrix

# Download NLTK packages needed
download('punkt')
download('stopwords')

# Retrieve english stop words used to filter words
ENGLISH_STOP_WORDS = stopwords.words("english")


def tokenize_text(corpus: ndarray):
    """
    Function that tokenizes all news in the dataset and filters unwanted words
    Args:
        corpus(ndarray): list of news loaded from dataset

    Returns:
        (list, set): (list of sentences which are a list of words, set of unique words across all sentences)
    """
    # Initialize relevant lists
    sentences = []  # List of lists containing the tokenized words of each text
    word_set = []  # Unique words found on all texts

    # Loop through the complete dataset
    for text in corpus:
        # Retrieve list of tokenized words
        tokenized_words = word_tokenize(text)
        # Exclude any stop words as they don't add meaning
        tokenized_words = list(filter(lambda word: word not in ENGLISH_STOP_WORDS and len(word) > 2, tokenized_words))
        # Add list of tokenized words to sentences list
        sentences.append(tokenized_words)
        # Add new words found in the sentence
        word_set.extend([word for word in tokenized_words if word not in word_set])

    return sentences, set(word_set)


def count_dict(sentences: list, words: list):
    """
    Function that loops over the sentences list and counts the absolute frequency of each word in the sentence
    Args:
        sentences(list): list of sentences which are a list of words
        words(list): set of unique words across all sentences

    Returns:
        dict: a dictionary that maps each word to its absolute frequency across all news
    """
    # Initialize all words with an absolute frequency of 0
    words_dict = {word: 0 for word in words}
    for sentence in sentences:
        for word in sentence:
            # The word could not be found as when testing new words may appear that didn't were taken into account on
            # training, so that's why the try except is needed
            try:
                words_dict[word] += 1
            except Exception as e:
                continue

    return words_dict


def term_frequency(document, word):
    """
    Function that computes the frequency of a word in a document, its part of the TF-IDF vectorization process
    Args:
        document(list): list of words in a news article
        word(str): word to look for in the document

    Returns:
        float: term frequency
    """
    n = len(document)
    occurrences = document.count(word)
    return occurrences / n


def inverse_document_frequency(word, word_count, total_docs):
    """
    Function that computes the inverse document frequency of a word. it's part of the TF-IDF vectorization process
    Args:
        word(str): word to take into account calculating the IDF
        word_count(dict): dictionary containing the absolute frequency of a word in the whole corpus
        total_docs(int): length of the corpus

    Returns:
        float: inverse document frequency
    """
    return np.log(total_docs / (word_count[word] + 1))


def tf_idf(sentence, words, words_count, corpus_len, word_index):
    """
    Function that computes the TF-IDF vector for each given news
    Args:
        sentence(list): list of words from the news text
        words(list): set of unique words in the corpus
        words_count(dict): dictionary of absolute frequency of words in the corpus
        corpus_len(int): size of the corpus
        word_index(dict): mapping dictionary that maps a word to its index in the features matrix

    Returns:
        ndarray: features matrix of shape (1, len(word_set))
    """
    empty_vec = np.zeros((len(words),))
    for word in sentence:
        # If the word wasn't taken into account while training so it should be ignored
        if word in words:
            tf = term_frequency(sentence, word)
            idf = inverse_document_frequency(word, words_count, corpus_len)
            empty_vec[word_index[word]] = tf * idf
    return empty_vec


def apply_tf_idf(corpus: ndarray, words_index: dict = None, words: list = None):
    """
    Function that applies the TF-IDF vectorization to the whole corpus to create the features matrix
    Args:
        corpus(ndarray): list of sentences from the news text
        words_index(dict): mapping dictionary that maps a word to its index in the features matrix
        words(list): list of unique words in the corpus

    Returns:
        scipy.sparse.tok_matrix: sparse matrix of shape (len(corpus), len(words))
        dict: mapping dictionary that maps a word to its index in the features matrix
        list: list of unique words in the corpus
    """
    corpus_len = len(corpus)
    sentences, words_set = tokenize_text(corpus)
    # If no words list is provided means that a new one must be created from the corpus
    if not words:
        words = list(words_set)
        del words_set
    # If no words index dictionary is provided means that a new one must be created from the words list
    if not words_index:
        words_index = dict(zip(words, range(len(words))))

    # Creating a dictionary containing the absolute frequency of each word in the whole corpus
    words_count = count_dict(sentences, words)

    # Initialize sparse matrix which will be our feature's matrix. PSA: I use a DOK sparse matrix as it can be easily
    # constructed on the go, but if you have the complete matrix you could turn it into a COO one.
    features_matrix = dok_matrix((corpus_len, len(words)), dtype=np.float32)
    for i, sentence in enumerate(sentences):
        features = tf_idf(sentence, words, words_count, corpus_len, words_index)
        for j, feature in enumerate(features):
            # Add the TF-IDF rate to the matrix if it's different from 0 as sparse matrices don't store zeros
            if feature != 0:
                features_matrix[i, j] = feature

    return features_matrix, words_index, words
