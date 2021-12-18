import torch
from torch.utils.data import Dataset

from tqdm import tqdm
from sklearn.metrics import f1_score

from models.embedding_models import produce_representation_vectors
from utils.utils import set_seed
from utils.nn_classifier import batch_size, dataloader_shuffle, get_m2_optimizer, num_epochs, criterion, m2_file_path


class TrainDevDataset(Dataset):
    def __init__(self, data):
        self.data = data
        rep, labels = produce_representation_vectors(self.data)
        self.rep = rep
        self.labels = torch.from_numpy(labels)
    def __len__(self):
        return self.rep.shape[0]
    def __getitem__(self, idx):
        return self.rep[idx], self.labels[idx]

def evaluate(model, dev_loader):
    set_seed()
    y_pred_list = []
    for i, x_y in tqdm(enumerate(dev_loader)):
        x, y = x_y
        logps = model(x.float())
        probabilities = torch.exp(logps)
        y_pred = torch.argmax(probabilities, dim=1)
        y_pred_list.append(y_pred)
    return [item for sublist in y_pred_list for item in sublist]

def get_data_set_loader(data):
    dataset = TrainDevDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=dataloader_shuffle)
    return dataset, loader


def train_evaluate_nn_model(m2_nn, train_loader, dev_loader, dev_dataset, train_dataset):
    set_seed()

    print(m2_nn)

    optimizer = get_m2_optimizer(m2_nn.parameters())
    best_f1 = 0.0

    for epoch in tqdm(range(num_epochs)):
        y_pred_list = []
        for i, x_y in enumerate(train_loader):
            x, y = x_y

            # forward
            logps = m2_nn(x.float())
            loss = criterion(logps, y.squeeze().long())

            # backward
            optimizer.zero_grad()
            loss.backward()

            # step
            optimizer.step()
            probabilities = torch.exp(logps)

            #train eval
            y_pred = torch.argmax(probabilities, dim=1)
            y_pred_list.append(y_pred)


        #eval
        m2_nn.train(False)
        y_pred_dev = torch.stack(evaluate(m2_nn, dev_loader)).tolist()
        dev_f1_score = f1_score(dev_dataset.labels, y_pred_dev, average="binary")
        y_pred_train = torch.stack([item for sublist in y_pred_list for item in sublist]).tolist()
        train_f1_score = f1_score(train_dataset.labels, y_pred_train, average="binary")
        #TODO print loss
        print(f'train_f1_score: {train_f1_score}, dev_f1_score: {dev_f1_score}')
        if dev_f1_score > best_f1:
            print('model saved')
            torch.save(m2_nn.state_dict(), m2_file_path)
            best_f1 = dev_f1_score
        m2_nn.train(True)
    return m2_nn, best_f1


def nn_model(train_data, dev_data, m2_nn):
    set_seed()

    train_dataset, train_loader = get_data_set_loader(train_data)
    dev_dataset, dev_loader = get_data_set_loader(dev_data)

    m2_nn, best_f1 = train_evaluate_nn_model(m2_nn, train_loader, dev_loader, dev_dataset, train_dataset)

    return m2_nn, best_f1