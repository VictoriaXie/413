import torch.nn as nn

class MyRNN(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_classes):
        super(MyRNN, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.emb = nn.Embedding(vocab_size, emb_size)
        # TODO: Your code here
        self.rnn = nn.RNN(emb_size, hidden_size, batch_first = True, bidirectional = True)
        self.fully_connected = nn.Linear(hidden_size*2, num_classes)
    def forward(self, X):
        # TODO: Your code here
        out = self.emb(X)
        h, o = self.rnn(out)
        out = (1/ h.shape[1]) * torch.sum(h, 1)
        out = self.fully_connected(torch.reshape(out, (h.shape[0], h.shape[2])))
        return out
