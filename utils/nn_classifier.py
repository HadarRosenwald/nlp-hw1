import torch
from torch import nn
from torch import optim

from utils import embedding


# This files holds the NN hyper params and variables

m2_file_path = 'm2.pth'


## hyperparameters: ##

# dataloader
batch_size = 64
dataloader_shuffle = False

#trainig
class_weights = torch.FloatTensor([1, 5])
criterion = nn.NLLLoss(weight=class_weights)
lr = 0.001
def get_m2_optimizer(m2_nn_parameters): return optim.Adam(m2_nn_parameters, lr=lr)
num_epochs = 30

#archtiecture
dp_p = 0.25
hidden_layers = [100, 50]

layers_dim = [embedding.vector_size] + hidden_layers
num_layers = len(layers_dim)
nn_layers = [[nn.Linear(layers_dim[i], layers_dim[i+1]), nn.ReLU(), nn.Dropout(dp_p)] for i in range(0,len(layers_dim)-1)]
output_layer = [nn.Linear(layers_dim[-1], 2), nn.LogSoftmax(dim=1)]

layers_flattened = [item for sublist in nn_layers for item in sublist] + output_layer
m2_nn = nn.Sequential(*layers_flattened)

