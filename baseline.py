import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchtext.vocab import build_vocab_from_iterator
from sklearn.model_selection import ParameterGrid

class MyRNN(nn.Module):
    def __init__(self, model_w2v, hidden_size, num_classes):
        super(MyRNN, self).__init__()
        self.vocab_size = len(model_w2v.wv)
        self.emb_size = model_w2v.vector_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.emb = nn.Embedding(self.vocab_size, self.emb_size)
        self.rnn = nn.RNN(self.emb_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, X):
        embedded = self.emb(X)
        _, hidden = self.rnn(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        output = self.fc(hidden)
        return output

    def parameters(self):
        for name, param in self.named_parameters():
            if name != 'emb.weight':
                yield param

def accuracy(model, dataset, max=1000):
    """
    Estimate the accuracy of `model` over the `dataset`.
    We will take the **most probable class**
    as the class predicted by the model.

    Parameters:
        `model`   - An object of class nn.Module
        `dataset` - A dataset of the same type as `train_data`.
        `max`     - The max number of samples to use to estimate
                    model accuracy

    Returns: a floating-point value between 0 and 1.
    """

    correct, total = 0, 0
    dataloader = DataLoader(dataset,
                            batch_size=1,  # use batch size 1 to prevent padding
                            collate_fn=collate_batch)
    for i, (x, t) in enumerate(dataloader):
        z = model(x)
        y = torch.argmax(z, axis=1)
        correct += int(torch.sum(t == y))
        total   += 1
        if i >= max:
            break
    return correct / total

def train_model(model,                # an instance of MLPModel
                train_data,           # training data
                val_data,             # validation data
                learning_rate=0.001,
                batch_size=100,
                num_epochs=10,
                plot_every=5,        # how often (in # iterations) to track metrics
                plot=True):           # whether to plot the training curve
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               collate_fn=collate_batch,
                                               shuffle=True) # reshuffle minibatches every epoch
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # these lists will be used to track the training progress
    # and to plot the training curve
    iters, train_loss, train_acc, val_acc = [], [], [], []
    iter_count = 0 # count the number of iterations that has passed

    try:
        for e in range(num_epochs):
            model.train()
            for i, (texts, labels) in enumerate(train_loader):
                texts = torch.tensor(texts).long()
                z = model(texts) 
                loss = criterion(z, labels) 

                loss.backward() # propagate the gradients
                optimizer.step() # update the parameters
                optimizer.zero_grad() # clean up accumualted gradients

                iter_count += 1
                if iter_count % plot_every == 0:
                    iters.append(iter_count)
                    ta = accuracy(model, train_data)
                    va = accuracy(model, val_data)
                    train_loss.append(float(loss))
                    train_acc.append(ta)
                    val_acc.append(va)
                    print(iter_count, "Loss:", float(loss), "Train Acc:", ta, "Val Acc:", va)
    finally:
        # This try/finally block is to display the training curve
        # even if training is interrupted
        if plot:
            plt.figure()
            plt.plot(iters[:len(train_loss)], train_loss)
            plt.title("Loss over iterations")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")

            plt.figure()
            plt.plot(iters[:len(train_acc)], train_acc)
            plt.plot(iters[:len(val_acc)], val_acc)
            plt.title("Accuracy over iterations")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.legend(["Train", "Validation"])
            plt.savefig("test.png")


def convert_indices(data, model_w2v):
    result = []
    for words, label in data:
        indices = []
        for w in words:
            if w in model_w2v.wv.key_to_index:
                indices.append(model_w2v.wv.key_to_index.get(w))
            else:
                # this is a bit of a hack, but we will repurpose *last* word
                # (least common word) appearing in the w2v vocabluary as our
                # '<pad>' token
                indices.append(len(model_w2v.wv)-1)
        result.append((indices, label),)
    return result


def collate_batch(batch):
    """
    Returns the input and target tensors for a batch of data

    Parameters:
        `batch` - An iterable data structure of tuples (emb, label),
                  where `emb` is a sequence of word embeddings, and
                  `label` is either 1 or 0.

    Returns: a tuple `(X, t)`, where
        - `X` is a PyTorch tensor of shape (batch_size, sequence_length)
        - `t` is a PyTorch tensor of shape (batch_size)
    where `sequence_length` is the length of the longest sequence in the batch
    """
    text_list = []  
    label_list = [] 
    for (text_indices, label) in batch:
        text_list.append(torch.tensor(text_indices))
        label_list.append(label)

    # TODO CONFIRM THIS PADDING VALUE!!!!   
    X = pad_sequence(text_list, padding_value=len(model_w2v.wv)-1).transpose(0, 1)
    t = torch.tensor(label_list, dtype=torch.long)
    return X, t

if __name__ == "__main__":
    human_token = pd.read_csv("human_token.csv", index_col=0)
    ai_token = pd.read_csv("ai_token.csv", index_col=0)
    model_w2v = Word2Vec.load("w2vmodel.model")

    # Combine human token and ai token
    token = pd.concat([human_token, ai_token], ignore_index=True)

    # Shuffle and split the data
    X_train, X_temp, y_train, y_temp = train_test_split(token["text"], token["human_wrote"], test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    train_data = list(zip(X_train, y_train))
    val_data = list(zip(X_val, y_val))
    test_data = list(zip(X_test, y_test))

    # Convert to indices
    train_data_indices = convert_indices(train_data, model_w2v)
    val_data_indices = convert_indices(val_data, model_w2v)
    test_data_indices = convert_indices(test_data, model_w2v)

    model = MyRNN(model_w2v,
              hidden_size=64,
              num_classes=2)

    train_model(model, train_data_indices[:20], val_data_indices[:20], batch_size=5, num_epochs=4)

#     param_grid = {
#     'batch_size': [32, 64, 128],
#     'learning_rate': [0.001, 0.0001],
#     'num_epochs': [5, 10]
# }
#     grid = list(ParameterGrid(param_grid))
#     for params in grid:
#     train_model(model, train_data, val_data, 
#                 learning_rate=params['learning_rate'], 
#                 batch_size=params['batch_size'], 
#                 num_epochs=params['num_epochs'])