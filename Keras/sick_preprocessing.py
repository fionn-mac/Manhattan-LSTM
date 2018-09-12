from re import sub

import csv
import itertools
import random
from random import shuffle

import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split as split_data

import keras
from keras.preprocessing.sequence import pad_sequences
import numpy as np

class Data(object):
    def __init__(self, data_dir, max_len=0):
        self.data_dir = data_dir
        self.max_len = max_len
        self.vocab_size = 1
        self.sentence_cols = ['sentence_A', 'sentence_B']
        self.x_train = list()
        self.y_train = list()
        self.x_val = list()
        self.y_val = list()
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

    def load_sick(self):
        # Load training set
        train_df = pd.read_csv(self.data_dir + 'SICK.txt', sep='\t')

        stops = set(stopwords.words('english'))

        # Iterate over the sentences only of training dataset
        for index, row in train_df.iterrows():
            # Iterate through the text of both sentences of the row
            for sentence in self.sentence_cols:
                q2n = []  # sentence Numbers Representation
                for word in self.text_to_word_list(row[sentence]):
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

                # Replace sentences as word to sentence as number representation
                train_df.set_value(index, sentence, q2n)

        return train_df

    def pad_sequences(self):
        if self.max_len == 0:
            self.max_len = max(max(len(seq) for seq in self.x_train[0]),
                              max(len(seq) for seq in self.x_train[1]),
                              max(len(seq) for seq in self.x_val[0]),
                              max(len(seq) for seq in self.x_val[1]))

        # Zero padding
        for dataset, side in itertools.product([self.x_train, self.x_val], [0, 1]):
            dataset[side] = pad_sequences(dataset[side], max_len=self.max_len)

    def run(self):
        print('Loading data and building vocabulary.')
        train_df = self.load_sick()

        # Split to train validation
        validation_size = int(len(train_df)*0.2)
        training_size = len(train_df) - validation_size

        X = train_df[self.sentence_cols]
        Y = train_df['relatedness_score']

        self.x_train, self.x_val, self.y_train, self.y_val = split_data(X, Y, test_size=validation_size)

        # Split to lists
        self.x_train = [self.x_train.sentence_A, self.x_train.sentence_B]
        self.x_val = [self.x_val.sentence_A, self.x_val.sentence_B]

        # Convert labels to their numpy representations
        self.y_train = self.y_train.values
        self.y_val = self.y_val.values

        print('Padding Sequences.')
        self.pad_sequences()
