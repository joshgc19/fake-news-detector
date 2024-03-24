from nltk import word_tokenize, download
import numpy as np
from numpy import ndarray
from nltk.corpus import stopwords
from scipy.sparse import dok_matrix

download('punkt')
download('stopwords')

ENGLISH_STOP_WORDS = stopwords.words("english")


def tokenize_text(corpus: ndarray):
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
        # TODO: LO QUE HACE COUNT DICT PUEDE SER INCLUIDO AQUI Y AHORRARSE UN CICLO

    return sentences, set(word_set)


def count_dict(sentences, word_set):
    words_dict = {word: 0 for word in word_set}
    for sentence in sentences:
        for word in sentence:
            try:
                words_dict[word] += 1
            except Exception as e:
                continue

    return words_dict


def term_frequency(document, word):
    n = len(document)
    occurrences = document.count(word)
    # list(filter(lambda token: token == word, document))
    return occurrences / n


def inverse_document_frequency(word, word_count, total_docs):
    # TODO: PUEDE SER SIMPLIFICADO PASANDO EL WORD COUNT ENVES DE EL DICCIONARIO Y LA LLAVE, PERO LO USAN CON TRY
    #  ENTONCES HAY QUE PROBAR
    return np.log(total_docs / (word_count[word] + 1))


def tf_idf(sentence, word_set, words_count, corpus_len, word_index):
    empty_vec = np.zeros((len(word_set),))
    for word in sentence:
        if word in word_index.keys():
            tf = term_frequency(sentence, word)
            idf = inverse_document_frequency(word, words_count, corpus_len)
            empty_vec[word_index[word]] = tf * idf
    return empty_vec


def apply_tf_idf(corpus: ndarray, words_index: dict = None, words: list = None):
    corpus_len = len(corpus)
    sentences, words_set = tokenize_text(corpus)
    if not words:
        words = list(words_set)
        del words_set
    if not words_index:
        words_index = dict(zip(words, range(len(words))))
    words_count = count_dict(sentences, words)

    features_matrix = dok_matrix((corpus_len, len(words)), dtype=np.float32)
    for i, sentence in enumerate(sentences):
        features = tf_idf(sentence, words, words_count, corpus_len, words_index)
        for j, feature in enumerate(features):
            if feature != 0:
                features_matrix[i, j] = feature

    # vectors = [tf_idf(sentence, word_set, words_count, corpus_len, word_index) for sentence in sentences]
    return features_matrix, words_index, words
