from preprocess.data_preprocess import get_data_from_files
from models.embedding_models import get_pretrained_rep_model, train_representation_model
from models.simple_classification_model import simple_model
from models.nn_classifier_model import nn_model
from utils.nn_classifier import m2_nn


def main():
    print("### Get data from files ###")
    train_data, dev_data = get_data_from_files()

    print("\n### Create embedding models ###")
    _ = get_pretrained_rep_model()
    train_representation_model(train_data)

    print("\n### Create classification models ###")
    print("### m1 ###")
    m1, m1_f1_score = simple_model(train_data, dev_data)
    print("### m2 ###")
    m2, m2_f1_score = nn_model(train_data, dev_data, m2_nn)

    print("\n### Evaluation results ###")
    print(f"F1 score for m1: {m1_f1_score}")
    print(f"F1 score for m2: {m2_f1_score}")



if __name__ == '__main__':
    main()
