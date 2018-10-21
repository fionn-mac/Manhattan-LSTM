import torch
import torch.nn as nn
from torch import Tensor
from torch import optim
import torch.nn.functional as F

class Manhattan_LSTM(nn.Module):
    def __init__(self, data_name, hidden_size, embedding, use_embedding=False, train_embedding=True):
        super(Manhattan_LSTM, self).__init__()
        self.data_name = data_name
        self.use_cuda = torch.cuda.is_available()
        self.hidden_size = hidden_size

        if use_embedding:
            self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
            self.embedding.weight = nn.Parameter(embedding)
            self.input_size = embedding.shape[1] # V - Size of embedding vector

        else:
            self.embedding = nn.Embedding(embedding[0], embedding[1])
            self.input_size = embedding[1]

        self.embedding.weight.requires_grad = train_embedding

        self.lstm_1 = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, bidirectional=False)
        self.lstm_2 = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, bidirectional=False)

    def exponent_neg_manhattan_distance(self, x1, x2):
        ''' Helper function for the similarity estimate of the LSTMs outputs '''
        return torch.exp(-torch.sum(torch.abs(x1 - x2), dim=1))

    def forward(self, input, hidden):
        '''
        input           -> (2 x Max. Sequence Length (per batch) x Batch Size)
        hidden          -> (2 x Num. Layers * Num. Directions x Batch Size x Hidden Size)
        '''
        embedded_1 = self.embedding(input[0]) # L, B, V
        embedded_2 = self.embedding(input[1]) # L, B, V

        batch_size = embedded_1.size()[1]

        outputs_1, hidden_1 = self.lstm_1(embedded_1, hidden)
        outputs_2, hidden_2 = self.lstm_2(embedded_2, hidden)

        similarity_scores = self.exponent_neg_manhattan_distance(hidden_1[0].permute(1, 2, 0).view(batch_size, -1),
                                                                 hidden_2[0].permute(1, 2, 0).view(batch_size, -1))

        if self.data_name == 'sick': return similarity_scores*5.0
        else: return similarity_scores

    def init_weights(self):
        ''' Initialize weights of lstm 1 '''
        for name_1, param_1 in self.lstm_1.named_parameters():
            if 'bias' in name_1:
                nn.init.constant_(param_1, 0.0)
            elif 'weight' in name_1:
                nn.init.xavier_normal_(param_1)

        ''' Set weights of lstm 2 identical to lstm 1 '''
        lstm_1 = self.lstm_1.state_dict()
        lstm_2 = self.lstm_2.state_dict()

        for name_1, param_1 in lstm_1.items():
            # Backwards compatibility for serialized parameters.
            if isinstance(param_1, torch.nn.Parameter):
                param_1 = param_1.data

            lstm_2[name_1].copy_(param_1)

    def init_hidden(self, batch_size):
        # Hidden dimensionality : 2 (h_0, c_0) x Num. Layers * Num. Directions x Batch Size x Hidden Size
        result = torch.zeros(2, 1, batch_size, self.hidden_size)

        if self.use_cuda: return result.cuda()
        else: return result
