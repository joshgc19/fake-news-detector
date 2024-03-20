import numpy as np
import pandas as pd
from nltk import word_tokenize
from numpy import ndarray


def tokenize_text(corpus: ndarray):
    # Initialize relevant lists
    sentences = []  # List of lists containing the tokenized words of each text
    word_set = []  # Unique words found on all texts

    # Loop through the complete dataset
    for text in corpus:
        # Retrieve list of tokenized words
        tokenized_words = word_tokenize(text)
        # Add list of tokenized words to sentences list
        sentences.append(tokenized_words)
        # Add new words found in the sentence
        word_set.extend([word for word in tokenized_words if word not in word_set])
        # TODO: LO QUE HACE COUNT DICT PUEDE SER INCLUIDO AQUI Y AHORRARSE UN CICLO

    return sentences, set(word_set)


def count_dict(sentences, word_set):
    words_dict = {word: 0 for word in word_set}
    for sentence in sentences:
        for word in sentence:
            words_dict[word] += 1

    return words_dict


def term_frequency(document, word):
    n = len(document)
    occurrences = filter(lambda token: token == word, document)
    return occurrences / n


def inverse_document_frequency(word, word_count, total_docs):
    # TODO: PUEDE SER SIMPLIFICADO PASANDO EL WORD COUNT ENVES DE EL DICCIONARIO Y LA LLAVE, PERO LO USAN CON TRY
    #  ENTONCES HAY QUE PROBAR
    return np.log(total_docs / (word_count[word] + 1))


def tf_idf(sentence, word_set, words_count, corpus_len, word_index):
    empty_vec = np.zeros((len(word_set),))
    for word in sentence:
        tf = term_frequency(sentence, word)
        idf = inverse_document_frequency(word, words_count, corpus_len)
        empty_vec[word_index[word]] = tf * idf
    return empty_vec


def apply_tf_idf(corpus: ndarray):
    corpus_len = len(corpus)
    sentences, word_set = tokenize_text(corpus)
    word_index = dict(zip(word_set, range(len(word_set))))
    words_count = count_dict(sentences, word_set)
    vectors = [tf_idf(sentence, word_set, words_count, corpus_len, word_index) for sentence in sentences]
    return vectors

