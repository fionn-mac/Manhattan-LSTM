import time
import random

import torch
import torch.nn as nn
from torch import optim
import numpy as np

from nltk import bleu_score

from helper import Helper

class Run_Iterations(object):
    def __init__(self, data_name, model, x_train, y_train, index2word, batch_size, num_iters,
                 learning_rate, tracking_pair=False, x_val=[], y_val=[], print_every=1, plot_every=1):
        self.use_cuda = torch.cuda.is_available()
        self.data_name = data_name
        self.model = model
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()

        self.tracking_pair = tracking_pair
        self.print_every = print_every
        self.plot_every = plot_every

        self.index2word = index2word
        ''' Lists that will contain data in the form of tensors. '''
        # Training data.
        self.x_train = x_train
        self.y_train = y_train
        self.train_samples = len(self.x_train)

        # Development data.
        self.x_val = x_val
        self.y_val = y_val
        self.val_samples = len(self.x_val)

        self.help_fn = Helper()

    def train_iters(self):
        start = time.time()
        plot_losses = []
        print_loss_total = 0.0  # Reset every self.print_every
        plot_loss_total = 0.0  # Reset every self.plot_every

        model_trainable_parameters = list(filter(lambda p: p.requires_grad, self.model.manhattan_lstm.parameters()))
        model_optimizer = optim.Adam(model_trainable_parameters, lr=self.learning_rate)

        print('Beginning Model Training.\n')

        for epoch in range(1, self.num_iters + 1):
            for i in range(0, self.train_samples, self.batch_size):
                input_variables = self.x_train[i : i + self.batch_size] # Batch Size x Sequence Length
                similarity_scores = self.y_train[i : i + self.batch_size] # Batch Size

                loss, _ = self.model.train(input_variables, similarity_scores, self.criterion, model_optimizer)
                print_loss_total += loss
                plot_loss_total += loss

            if epoch % self.print_every == 0:
                print_loss_avg = print_loss_total / self.print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (self.help_fn.time_slice(start, epoch / self.num_iters),
                                             epoch, epoch / self.num_iters * 100, print_loss_avg))

            if epoch % self.plot_every == 0:
                plot_loss_avg = plot_loss_total / self.plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

            print('Validation Accuracy: %f Validation Precision: %f Validation Recall: %f Validation Loss: %f' % self.get_accuracy())
            print('\n')

            if epoch % 5 == 0:
                self.learning_rate *= 0.80
                model_optimizer = optim.Adam(model_trainable_parameters, lr=self.learning_rate)

        # self.help_fn.show_plot(plot_losses)

    def evaluate(self, seqs, scores):
        loss, similarity_scores = self.model.train(seqs, scores, self.criterion, evaluate=True)
        return loss, similarity_scores

    def evaluate_specific(self, seqs, score, name='tracking_pair'):
        sequence1 = [self.index2word[j.item()] for j in seqs[0].view(-1).data]
        sequence2 = [self.index2word[j.item()] for j in seqs[1].view(-1).data]
        print('>', sequence1)
        print('>', sequence2)
        print('=', score.item())

        _, similarity_score = self.evaluate([seqs], score)
        print('<', similarity_score.item())

    def evaluate_randomly(self, n=10):
        for i in range(n):
            ind = random.randrange(self.val_samples)
            self.evaluate_specific(self.x_val[ind], self.y_val[ind], name=str(i))

    def get_accuracy(self):
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        total_loss = 0

        accuracy = 0.0
        precision = 0.0
        recall = 0.0

        scale = 1.0
        if self.data_name == 'sick': scale *= 5.0

        for i in range(0, self.val_samples, self.batch_size):
            input_variables = self.x_val[i : i + self.batch_size] # Batch Size x Sequence Length
            actual_scores = self.y_val[i : i + self.batch_size] # Batch Size

            loss, predicted_scores = self.model.train(input_variables, actual_scores, self.criterion, evaluate=True)
            total_loss += loss

            for actual, predict in zip(actual_scores, predicted_scores):
                if actual.item()/scale < 0.5 and predict.item() < 0.5:
                    true_negative += 1

                if actual.item()/scale < 0.5 and predict.item() >= 0.5:
                    false_positive += 1

                elif actual.item()/scale >= 0.5 and predict.item() >= 0.5:
                    true_positive += 1

                if actual.item()/scale >= 0.5 and predict.item() < 0.5:
                    false_negative += 1

        accuracy = (true_positive + true_negative)*100/len(self.x_val)
        if true_positive + false_positive > 0: precision = true_positive*100/(true_positive + false_positive)
        if true_positive + false_negative > 0: recall = true_positive*100/(true_positive + false_negative)

        return accuracy, precision, recall, total_loss
