from re import sub

import csv
import itertools
import random
from random import shuffle

import pandas as pd
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split as split_data

import keras
from keras.preprocessing.sequence import pad_sequences
import numpy as np

class Data(object):
    def __init__(self, path, maxlen=0):
        self.path = path
        self.maxlen = maxlen
        self.vocab_size = 1
        self.questions_cols = ['question1', 'question2']
        self.x_train = list()
        self.y_train = list()
        self.x_val = list()
        self.y_val = list()
        self.x_test = list()
        self.y_test = list()
        self.vocab = set()
        self.word_to_id = dict()
        self.id_to_word = dict()
        self.run()

    def text_to_word_list(self, text):
        ''' Pre process and convert texts to a list of words '''
        text = str(text)
        text = text.lower()

        # Clean the text
        text = sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = sub(r"what's", "what is ", text)
        text = sub(r"\'s", " ", text)
        text = sub(r"\'ve", " have ", text)
        text = sub(r"can't", "cannot ", text)
        text = sub(r"n't", " not ", text)
        text = sub(r"i'm", "i am ", text)
        text = sub(r"\'re", " are ", text)
        text = sub(r"\'d", " would ", text)
        text = sub(r"\'ll", " will ", text)
        text = sub(r",", " ", text)
        text = sub(r"\.", " ", text)
        text = sub(r"!", " ! ", text)
        text = sub(r"\/", " ", text)
        text = sub(r"\^", " ^ ", text)
        text = sub(r"\+", " + ", text)
        text = sub(r"\-", " - ", text)
        text = sub(r"\=", " = ", text)
        text = sub(r"'", " ", text)
        text = sub(r"(\d+)(k)", r"\g<1>000", text)
        text = sub(r":", " : ", text)
        text = sub(r" e g ", " eg ", text)
        text = sub(r" b g ", " bg ", text)
        text = sub(r" u s ", " american ", text)
        text = sub(r"\0s", "0", text)
        text = sub(r" 9 11 ", "911", text)
        text = sub(r"e - mail", "email", text)
        text = sub(r"j k", "jk", text)
        text = sub(r"\s{2,}", " ", text)

        text = text.split()

        return text

    def load_quora(self):
        # Load training and test set
        train_df = pd.read_csv(self.path + 'train.csv')
        # test_df = pd.read_csv(self.path + 'test.csv')

        stops = set(stopwords.words('english'))

        # Iterate over the questions only of both training and test datasets
        for dataset in [train_df]:
            for index, row in dataset.iterrows():
                # Iterate through the text of both questions of the row
                for question in self.questions_cols:
                    q2n = []  # Question Numbers Representation
                    for word in self.text_to_word_list(row[question]):
                        # Remove unwanted words
                        if word in stops:
                            continue

                        if word not in self.vocab:
                            self.vocab.add(word)
                            self.word_to_id[word] = self.vocab_size
                            q2n.append(self.vocab_size)
                            self.id_to_word[self.vocab_size] = word
                            self.vocab_size += 1
                        else:
                            q2n.append(self.word_to_id[word])

                    # Replace questions as word to question as number representation
                    dataset.set_value(index, question, q2n)

        return train_df

    def pad_sequences(self):
        if self.maxlen == 0:
            self.maxlen = max(max(len(seq) for seq in self.x_train[0]),
                              max(len(seq) for seq in self.x_train[1]),
                              max(len(seq) for seq in self.x_val[0]),
                              max(len(seq) for seq in self.x_val[1]))

        # Zero padding
        for dataset, side in itertools.product([self.x_train, self.x_val], [0, 1]):
            dataset[side] = pad_sequences(dataset[side], maxlen=self.maxlen)

    def run(self):
        print('Loading data and building vocabulary.')
        train_df = self.load_quora()

        # Split to train validation
        validation_size = 2000
        training_size = len(train_df) - validation_size

        X = train_df[self.questions_cols]
        Y = train_df['is_duplicate']

        self.x_train, self.x_val, self.y_train, self.y_val = split_data(X, Y, test_size=validation_size)

        # Split to lists
        self.x_train = [self.x_train.question1, self.x_train.question2]
        self.x_val = [self.x_val.question1, self.x_val.question2]
        # self.x_test = [test_df.question1, test_df.question2]

        # Convert labels to their numpy representations
        self.y_train = self.y_train.values
        self.y_val = self.y_val.values

        print('Padding Sequences.')
        self.pad_sequences()
