from typing import List, Tuple
from utils.utils import files_paths
import re

def preprocess_url(word):
    is_url = re.search('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', word)
    if not is_url:
        return word
    return 'url_'


def find_repeating_letters(word):
    rep_letters = re.findall(r'((\w)\2{2,})', word)
    for sequence, single in rep_letters:
        word = word.replace(sequence, single)
    return word


def preprocess(word):
    word = word.lower().strip()
    word = preprocess_url(word)
    word = find_repeating_letters(word)
    return word

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
        word = preprocess(word_label[0])
        label = word_label[1].rstrip()
        label = 0 if label == 'O' else 1
        sentence_words.append(word)
        sentence_labels.append(label)
    file.close()
    return sentences_words, sentences_labels


def get_data_from_files():
    # parse data files. See each function to understand structure
    train_data = read_train_dev_data(train=True)
    dev_data = read_train_dev_data(train=False)
    return train_data, dev_data