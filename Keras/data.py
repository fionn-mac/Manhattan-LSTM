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
    def __init__(self, data_name, data_file, train_ratio=0.8, max_len=None,
                 vocab_limit=None, sentence_cols=None, score_col=None):
        self.data_file = data_file
        self.train_ratio = train_ratio
        self.max_len = max_len
        self.vocab_size = 1
        self.vocab_limit = vocab_limit

        if data_name.lower() == 'sick':
            self.score_col = 'relatedness_score'
            self.sequence_cols = ['sentence_A', 'sentence_B']

        elif data_name.lower() == 'quora':
            self.score_col = 'is_duplicate'
            self.sequence_cols = ['question1', 'question2']

        else:
            self.score_col = score_col
            self.sequence_cols = questions_cols

        self.x_train = list()
        self.y_train = list()
        self.x_val = list()
        self.y_val = list()
        self.vocab = set('PAD')
        self.word_to_id = {'PAD':0}
        self.id_to_word = {0:'PAD'}
        self.word_to_count = dict()
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

    def load_data(self):
        stops = set(stopwords.words('english'))

        # Load data set
        data_df = pd.read_csv(self.data_file, sep='\t')

        # Iterate over required sequences of provided dataset
        for index, row in data_df.iterrows():
            # Iterate through the text of both questions of the row
            for sequence in self.sequence_cols:
                s2n = []  # Sequences with words replaces with indices
                for word in self.text_to_word_list(row[sequence]):
                    # Remove unwanted words
                    if word in stops:
                        continue

                    if word not in self.vocab:
                        self.vocab.add(word)
                        self.word_to_id[word] = self.vocab_size
                        self.word_to_count[word] = 1
                        s2n.append(self.vocab_size)
                        self.id_to_word[self.vocab_size] = word
                        self.vocab_size += 1
                    else:
                        self.word_to_count[word] += 1
                        s2n.append(self.word_to_id[word])

                # Replace |sequence as word| with |sequence as number| representation
                data_df.at[index, sequence] = s2n

        return data_df

    def pad_sequences(self):
        if self.max_len == 0:
            self.max_len = max(max(len(seq) for seq in self.x_train[0]),
                               max(len(seq) for seq in self.x_train[1]),
                               max(len(seq) for seq in self.x_val[0]),
                               max(len(seq) for seq in self.x_val[1]))

        # Zero padding
        for dataset, side in itertools.product([self.x_train, self.x_val], [0, 1]):
            if self.max_len: dataset[side] = pad_sequences(dataset[side], maxlen=self.max_len)
            else : dataset[side] = pad_sequences(dataset[side])

    def run(self):
        # Loading data and building vocabulary.
        data_df = self.load_data()
        data_size = len(data_df)

        X = data_df[self.sequence_cols]
        Y = data_df[self.score_col]

        self.x_train, self.x_val, self.y_train, self.y_val = split_data(X, Y, train_size=self.train_ratio)

        # Split to lists
        self.x_train = [self.x_train[column] for column in self.sequence_cols]
        self.x_val = [self.x_val[column] for column in self.sequence_cols]

        # Convert labels to their numpy representations
        self.y_train = self.y_train.values
        self.y_val = self.y_val.values

        # Padding Sequences.
        self.pad_sequences()
