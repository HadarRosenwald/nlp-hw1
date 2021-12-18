import pickle
from datetime import datetime

from sklearn import svm
from sklearn.metrics import f1_score

from embedding_models import produce_representation_vectors
from utils.simple_classifier import m1_file_path
from utils.utils import set_seed


def train_simple_model(train_data, glove, representation_model):
    set_seed()
    ### Training ###

    # Each word will get a representation vector:
    rep, labels = produce_representation_vectors(train_data, glove, representation_model)
    # rep: type: np.ndarray, shape: (46469, 200). 46469 is the # words in training data
    # labels: type: np.ndarray, shape: (46469,).
    print(
        f"Produced representation for the words in the training data set. {rep.shape[0]} words with vector len {rep.shape[1]}\n")

    # Training simple model:
    print(f"Starting model fit on training data (on {datetime.now().time()})")

    svm_model = svm.SVC().fit(rep, labels)
    print(f"Model fitting completed (on {datetime.now().time()})\n")

    with open(m1_file_path, 'wb') as f: # TODO SHOULD WE CLOSE?
        pickle.dump(svm_model, f)

    # TODO do we want to add inference + evaluation for train F1 result?


def evaluate_simple_model(dev_data, glove, representation_model, svm_model) -> float:
    ### Evaluation ###

    # Representing the words from dev data set:
    dev_rep, dev_labels = produce_representation_vectors(dev_data, glove, representation_model)
    # dev_rep: type: np.ndarray, shape: (16261, 200). 16261 is the # words in dev data
    # dev_labels: type: np.ndarray, shape: (16261,).
    print(
        f"Produced representation for the words in the dev data set. {dev_rep.shape[0]} words with vector len {dev_rep.shape[1]}\n")

    # Inference:
    print(f"Starting inference on dev data set (on {datetime.now().time()})")
    dev_pred = svm_model.predict(dev_rep)
    # dev_pred: type: np.ndarray, shape: (16261,). 16261 is the # words in dev data
    print(f"Inference completed (on {datetime.now().time()})\n")

    # Performance evaluations:
    m1_f1_score = f1_score(dev_labels, dev_pred, average='binary')
    print(f"Model's F1 binary score: {m1_f1_score}")
    # models_f1_score: type: float

    return m1_f1_score


def load_simple_model():
    with open(m1_file_path, 'rb') as f:
        svm_model = pickle.load(f)
    # TODO SHOULD WE CLOSE?
    return svm_model


def simple_model(train_data, dev_data, glove, representation_model) -> (svm._classes.SVC, float):
    train_simple_model(train_data, glove, representation_model)
    svm_model = load_simple_model()
    m_1_f1_score = evaluate_simple_model(dev_data, glove, representation_model, svm_model)
    return svm_model, m_1_f1_score
