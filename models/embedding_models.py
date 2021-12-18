import numpy as np

import os.path

from gensim import downloader
from gensim.models import Word2Vec, KeyedVectors, keyedvectors, word2vec

from datetime import datetime

from utils.embedding import pretrained_embedding_file_path, embedding_file_path
from utils import utils
from utils import embedding


def get_pretrained_rep_model() -> keyedvectors.KeyedVectors:
    # Download and return pretrained model
    print(f"downloading glove (on {datetime.now().time()})")
    if os.path.isfile(pretrained_embedding_file_path):
        glove = KeyedVectors.load(pretrained_embedding_file_path)
    else:
        glove = downloader.load(utils.GLOVE_PATH)
        glove.save(pretrained_embedding_file_path)
    print(f"Finished downloading glove (on {datetime.now().time()})")
    return glove


def train_representation_model(train_data) -> word2vec.Word2Vec:
    # train a Word2Vec model from given data
    training_data = train_data[0]
    model = Word2Vec(sentences=training_data, vector_size=embedding.vector_size, window=embedding.window_size,
                     min_count=embedding.min_count, workers=embedding.num_workers)
    model.train(training_data, total_examples=len(training_data), epochs=embedding.num_epochs)
    model.wv.save_word2vec_format(embedding_file_path, binary=True)
    return model


def produce_representation_vector_per_word(word: str, glove: keyedvectors.KeyedVectors,
                                           representation_model: word2vec.Word2Vec) -> np.ndarray:
    # For each word, returns the embedding vector with length utils.embedding.vector_size
    in_pretrained = word in glove.key_to_index
    in_trained = word in representation_model.wv.key_to_index
    if not in_pretrained and not in_trained:
        # not an existing word in any of the models")
        vec = np.zeros(200)
    else:
        if in_pretrained and in_trained:
            # word exists in both, will use weighted averaged embedding vector
            vec = embedding.pretrained_weight * glove[word] + (1-embedding.pretrained_weight) * representation_model.wv.get_vector(word, norm=True)
        elif in_pretrained:
            # word exists only in pretrained embedding model
            vec = glove[word]
        else:
            # word exists only in trained embedding model
            vec = representation_model.wv.get_vector(word, norm=True)
    return vec


def produce_representation_vectors(dataset, glove: keyedvectors.KeyedVectors, representation_model: word2vec.Word2Vec):
    representation = []
    for sentence in dataset[0]:
        for word in sentence:
            vec = produce_representation_vector_per_word(word, glove, representation_model)
            representation.append(vec)

    labels = [item for sublist in dataset[1] for item in sublist]
    return np.asarray(representation), np.asarray(labels)
