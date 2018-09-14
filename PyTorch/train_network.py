import random

import torch
import torch.nn.utils.rnn as rnn

class Train_Network(object):
    def __init__(self, manhattan_lstm, index2word):
        self.manhattan_lstm = manhattan_lstm
        self.index2word = index2word
        self.use_cuda = torch.cuda.is_available()

    def train(self, input_sequences, similarity_scores, criterion, model_optimizer=None, evaluate=False):

        sequences_1 = [sequence[0] for sequence in input_sequences]
        sequences_2 = [sequence[1] for sequence in input_sequences]
        batch_size = len(sequences_1)

        '''
        Pad all tensors in this batch to same length.
        PyTorch pad_sequence method doesn't take pad length, making this step problematic.
        Therefore, lists concatenated, padded to common length, and then split.
        '''
        temp = rnn.pad_sequence(sequences_1 + sequences_2)
        sequences_1 = temp[:, :batch_size]
        sequences_2 = temp[:, batch_size:]

        ''' No need to send optimizer in case of evaluation. '''
        if model_optimizer: model_optimizer.zero_grad()
        loss = 0.0

        hidden = self.manhattan_lstm.init_hidden(batch_size)
        output_scores = self.manhattan_lstm([sequences_1, sequences_2], hidden).view(-1)

        loss += criterion(output_scores, similarity_scores)

        if not evaluate:
            loss.backward()
            model_optimizer.step()

        return loss.item(), output_scores
