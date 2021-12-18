from preprocess.data_preprocess import get_data_from_files
from models.embedding_models import get_pretrained_rep_model, train_representation_model
from models.simple_classification_model import simple_model
from models.nn_classifier_model import nn_model
from utils.nn_classifier import m2_nn


def main():
    # Get data from files
    train_data, dev_data = get_data_from_files()

    # Create embedding models
    glove = get_pretrained_rep_model()
    representation_model = train_representation_model(train_data)

    # Create classification models
    m1, m1_f1_score = simple_model(train_data, dev_data, glove, representation_model)
    m2, m2_f1_score = nn_model(train_data, dev_data, glove, representation_model, m2_nn)



if __name__ == '__main__':
    main()
