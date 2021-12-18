from copy import copy
from sklearn import svm
from datetime import datetime

from utils.utils import files_paths
from utils.nn_classifier import m2_nn, m2_file_path
from utils.simple_classifier import m1_file_path
from embedding_models import produce_representation_vector_per_word

import pickle
import torch


def load_models():
    with open(m1_file_path, 'rb') as f:
        m1 = pickle.load(f)
    state_dict = torch.load(m2_file_path)
    m2 = copy(m2_nn)
    m2.load_state_dict(state_dict)
    m3 = m1
    return m1, m2, m3


def generate_files(model_object, model_name):
    read_f = open(files_paths['test'], "r")
    write_f = open(f'comp_{model_name}_207108820.tagged', 'w')

    for line in read_f:
        if line == "\n":  # a sentence ended.
            write_f.write(line)
            continue

        # Pre process
        word = line.lower().rstrip()
        # TODO IF THERE ARE MORE PREPROCESS, IT SHOULD BE DONE HERE ALSO!!!

        # Get representation vector
        vec = produce_representation_vector_per_word(word, glove, representation_model)

        # Run Inference
        if type(model_object) == svm._classes.SVC:
            m_pred = model_object.predict(vec.reshape(1, -1))
        else:
            x = torch.tensor(vec).view(1,
                                       -1)  # artificially setting a batch of size = 1, containing only that single sample
            model_object.eval()
            logps = model_object(x.float())
            probabilities = torch.exp(logps)
            m_pred = torch.argmax(probabilities, dim=1)

        # Write to file
        m_pred = 'O' if m_pred[0] == 0 else 'X'
        write_f.write(f"{line.rstrip()}\t{m_pred}\n")

    read_f.close()
    write_f.close()


def main():
    m1, m2, m3 = load_models()
    for i, m in enumerate([m1, m2, m3]):
        model_name = f'm{i + 1}'
        print(f"Generating file for model {model_name} (on {datetime.now().time()})\n")
        generate_files(m, model_name)
