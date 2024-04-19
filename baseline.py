import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse

class MyRNN(nn.Module):
    def __init__(self, model_w2v, hidden_size, num_classes):
        super(MyRNN, self).__init__()
        self.vocab_size = len(model_w2v.wv)+1
        self.emb_size = model_w2v.vector_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.emb = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=0)
        # Add an extra zero-initialized vector as the padding vector
        self.emb.weight.data.copy_(torch.from_numpy(np.vstack((np.zeros((1, self.emb_size)), model_w2v.wv.vectors))))
        self.emb = nn.Embedding.from_pretrained(torch.from_numpy(model_w2v.wv.vectors), freeze=True)
        self.rnn = nn.RNN(self.emb_size, hidden_size, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, X):
        embedded = self.emb(X)
        _, hidden = self.rnn(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        dropped = self.dropout(hidden)
        output = self.fc(dropped)
        return output

    def parameters(self):
        for name, param in self.named_parameters():
            if name != 'emb.weight':
                yield param


class MyLSTM(nn.Module):
    def __init__(self, model_w2v, hidden_size, num_classes):
        super(MyLSTM, self).__init__()
        self.vocab_size = len(model_w2v.wv) + 1
        self.emb_size = model_w2v.vector_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.emb = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=0)
        # Add an extra zero-initialized vector as the padding vector
        self.emb.weight.data.copy_(torch.from_numpy(np.vstack((np.zeros((1, self.emb_size)), model_w2v.wv.vectors))))
        self.lstm = nn.LSTM(self.emb_size, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.dropout_layer = nn.Dropout(p=0.2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, X):
        embedded = self.emb(X)
        outputs, (ht, ct) = self.lstm(embedded)
        output = self.dropout_layer(torch.cat((ht[-2], ht[-1]), dim=1))
        output = self.fc(output)
        return output

    def parameters(self):
        for name, param in self.named_parameters():
            if name != 'emb.weight':
                yield param


def accuracy(model, dataset):
    """
    Estimate the accuracy of `model` over the `dataset`.
    We will take the **most probable class**
    as the class predicted by the model.

    Parameters:
        `model`   - An object of class nn.Module
        `dataset` - A dataset of the same type as `train_data`.

    Returns: a floating-point value between 0 and 1.
    """
    model.eval()
    correct, total = 0, 0
    dataloader = DataLoader(dataset,
                            batch_size=20, 
                            collate_fn=collate_batch)
    for i, (x, t) in enumerate(dataloader):
        x, t = x.to(device), t.to(device)
        z = model(x)
        y = torch.argmax(z, axis=1)
        correct += int(torch.sum(t == y))
        total += t.size(0)

    model.train()
    return correct / total

def train_model(model,                # an instance of MLPModel
                train_data,           # training data
                val_data,             # validation data
                learning_rate=0.0001,
                batch_size=314,
                num_epochs=10,
                plot=True):           # whether to plot the training curve
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               collate_fn=collate_batch,
                                               shuffle=True) # reshuffle minibatches every epoch
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # these lists will be used to track the training progress
    # and to plot the training curve
    train_loss, train_acc, val_acc = [], [], []

    try:
        for e in range(num_epochs):
            model.train()
            for i, (texts, labels) in enumerate(train_loader):
                texts = texts.to(device)
                labels = labels.to(device)
                z = model(texts) 
                loss = criterion(z, labels) 

                loss.backward() # propagate the gradients
                optimizer.step() # update the parameters
                optimizer.zero_grad() # clean up accumualted gradients

            ta = accuracy(model, train_data)
            va = accuracy(model, val_data)
            train_loss.append(float(loss))
            train_acc.append(ta)
            val_acc.append(va)
            print(e, "Loss:", float(loss), "Train Acc:", ta, "Val Acc:", va)

    finally:
        # This try/finally block is to display the training curve
        # even if training is interrupted
        if plot:
            plt.figure()
            plt.plot(range(num_epochs), train_loss)
            plt.title("Loss over iterations")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.savefig("loss.png")
            plt.close()

            plt.figure()
            plt.plot(range(num_epochs), train_acc)
            plt.plot(range(num_epochs), val_acc)
            plt.title("Accuracy over iterations")
            plt.xlabel("Iterations")
            plt.ylabel("Accuracy")
            plt.legend(["Train", "Validation"])
            plt.savefig("acc.png")
            plt.close()


def convert_indices(data, model_w2v):
    result = []
    for words, label in data:
        indices = []
        for w in words:
            if w in model_w2v.wv.key_to_index:
                # Shift indices up by one since the padding token is at index 0
                indices.append(model_w2v.wv.key_to_index.get(w) + 1) 
            else:
                indices.append(0)
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

    X = pad_sequence(text_list, padding_value=0, batch_first=True)
    t = torch.tensor(label_list, dtype=torch.long)
    return X, t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='RNN') # Choose RNN or LSTM
    args = parser.parse_args()  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(22)

    human_token = pd.read_csv("human_token.csv", index_col=0)
    ai_token = pd.read_csv("ai_token.csv", index_col=0)
    model_w2v = Word2Vec.load("w2vmodel.model")

    # Combine human token and ai token
    token = pd.concat([human_token, ai_token], ignore_index=True)

    # Shuffle and split the data
    X_train, X_temp, y_train, y_temp = train_test_split(token["text"], token["human_wrote"], test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    train_data = list(zip(X_train, y_train))
    val_data = list(zip(X_val, y_val))
    test_data = list(zip(X_test, y_test))

    # Convert to indices
    train_data_indices = convert_indices(train_data, model_w2v)
    val_data_indices = convert_indices(val_data, model_w2v)
    test_data_indices = convert_indices(test_data, model_w2v)

    if args.model == "RNN":
        model = MyRNN(model_w2v, hidden_size=300, num_classes=2).to(device)
        train_model(model, train_data_indices, val_data_indices, batch_size=400)
    elif args.model == "LSTM":
        model = MyLSTM(model_w2v, hidden_size=256, num_classes=2).to(device)
        train_model(model, train_data_indices, val_data_indices, batch_size=314)
    
    print(f"test accuracy: {accuracy(model, test_data_indices)}")
    torch.save(model.state_dict(), f"models/best_{args.model}.pth")
        

    
