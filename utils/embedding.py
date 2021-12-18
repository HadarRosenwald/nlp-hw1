
# This files holds Embedding hyper params

vector_size = 200 # Dimensionality of the word vectors.
window_size = 5 # Maximum distance between the current and predicted word within a sentence.
min_count = 1 # Ignores all words with total frequency lower than this.
num_workers = 4 # Use these many worker threads to train the model (=faster training with multicore machines).
num_epochs = 10

pretrained_weight = 0.8