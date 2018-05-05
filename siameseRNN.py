import os
from sys import exit
from time import time
import argparse
import datetime
from math import exp

import tensorflow as tf

import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Activation
from keras.layers import Embedding, Input, Merge
from keras.layers import LSTM, Lambda, concatenate
from keras import regularizers

import numpy as np

# import matplotlib.pyplot as plt

from sick_preprocessing import Data
from embedding_google import GetEmbedding

def exponent_neg_manhattan_distance(x):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(x[:,:50] - x[:,50:]), axis=1, keepdims=True))

def exponent_neg_cosine_distance(x):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    leftNorm = K.l2_normalize(x[:,:50], axis=-1)
    rightNorm = K.l2_normalize(x[:,50:], axis=-1)
    return K.exp(K.sum(K.prod([leftNorm, rightNorm], axis=0), axis=1, keepdims=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", "--data", type=str, help="Path to dataset", default="./SICK/")
    parser.add_argument("--trp", type=int, help="Percentage of samples in training set", default=80)
    parser.add_argument("--maxlen", type=int, help="Maximum number of words in a sentence", default=20)
    parser.add_argument("-dwrd", "--word_embed_size", type=int, help="Word embedding size", default=300)
    parser.add_argument("-lstmu", "--word_lstm_units", type=int, help="Number of Units in LSTM layer", default=50)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch Size", default=64)
    parser.add_argument("--num_iters", type=int, help="Number of iterations", default=3)

    args = parser.parse_args()

    embedding_sizes = [50, 100, 200, 300]
    if args.word_embed_size not in embedding_sizes:
        raise Exception('Invalid embedding size - Allowed sizes are 50, 100, 200, and 300.')
        exit()

    data = Data(args.data)

    x_train = data.x_train
    y_train = data.y_train
    x_test = data.x_val
    y_test = data.y_val
    vocab_size = data.vocab_size
    maxlen = data.maxlen

    print(len(x_train[0]))
    print(len(x_test[0]))
    print(maxlen)

    embedding = GetEmbedding(data.word_to_id, vocab_size, args.word_embed_size)

    print('Building model')
 
    seq_1 = Input(shape=(maxlen,), dtype='int32', name='sequence1')
    seq_2 = Input(shape=(maxlen,), dtype='int32', name='sequence2')

    embed_layer = Embedding(output_dim=args.word_embed_size, input_dim=vocab_size, input_length=maxlen, trainable=False)
    embed_layer.build((None,))
    embed_layer.set_weights([embedding.embedding_matrix])

    input_1 = embed_layer(seq_1)
    input_2 = embed_layer(seq_2)

    l1 = LSTM(units=args.word_lstm_units)

    l1_out = l1(input_1)
    l2_out = l1(input_2)

    concats = concatenate([l1_out, l2_out], axis=-1)

    main_output = Lambda(exponent_neg_cosine_distance, output_shape=(1,))(concats)

    model = Model(inputs=[seq_1, seq_2], outputs=[main_output])

    opt = keras.optimizers.Adadelta(clipnorm=1.25)

    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
    model.summary()

    training_start_time = time()

    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        epochs=args.num_iters, batch_size=args.batch_size, verbose=1)

    print("Training time finished.\n{} epochs in {}".format(args.num_iters,
                                                            datetime.timedelta(seconds=time()-training_start_time)))

    #  "Accuracy"
    # plt.figure(1)
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('Model Accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['Training', 'Validation'], loc='upper left')
    # plt.show()
    # 
    # # "Loss"
    # plt.figure(2)
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model Loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['Training', 'Validation'], loc='upper left')
    # plt.show()
