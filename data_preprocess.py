from typing import List, Tuple
from utils.utils import GLOVE_PATH, files_paths

def read_train_dev_data(train: bool = True) -> Tuple[List[List[str]], List[List[int]]]:
    # Parse train and dev data.
    # Returns a tuple - first entry for the words, second entry for labels
    # each entry holds a list of Sentences, each sentence - a list of words (either the word itself or it's label)

    file = open(files_paths['train'] if train else files_paths['dev'], "r")
    sentences_words = list()
    sentences_labels = list()

    sentence_words = list()
    sentence_labels = list()

    for line in file:
        word_label = line.split("\t")
        if word_label == ["\n"]:  # a sentence ended.
            sentences_words.append(sentence_words)
            sentences_labels.append(sentence_labels)
            sentence_words = list()
            sentence_labels = list()
            continue
        word = word_label[0].lower()
        label = word_label[1].rstrip()
        label = 0 if label == 'O' else 1
        sentence_words.append(word)
        sentence_labels.append(label)
    file.close()
    return sentences_words, sentences_labels


def read_test_data() -> List[List[str]]:
    # Parse test data.
    # Returns a list of Sentences, each sentence - a list of the words
    # TODO - if not using it for training the rep model, then this is redundent. remove
    file = open(files_paths['test'], "r")
    sentences_words = list()
    sentence_words = list()

    for line in file:
        if line == "\n":  # a sentence ended.
            sentences_words.append(sentence_words)
            sentence_words = list()
            continue
        word = line.lower().rstrip()
        sentence_words.append(word)
    file.close()
    return sentences_words


def get_data_from_files():
    # parse data files. See each function to understand structure
    train_data = read_train_dev_data(train=True)
    dev_data = read_train_dev_data(train=False)
    test_data = read_test_data()
    return train_data, dev_data, test_data