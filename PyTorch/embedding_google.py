import numpy as np
from gensim.models import KeyedVectors

import torch

class Get_Embedding(object):
    def __init__(self, file_path, word_index):
        self.use_cuda = torch.cuda.is_available()
        self.embedding_size = 300 # Dimensionality of Google News' Word2Vec
        self.embedding_matrix = self.create_embed_matrix(file_path, word_index)

    def create_embed_matrix(self, file_path, word_index):
        word2vec = KeyedVectors.load_word2vec_format(file_path, binary=True)

        # Prepare Embedding Matrix.
        embedding_matrix = np.zeros((len(word_index)+1, self.embedding_size))

        for word, i in word_index.items():
            # words not found in embedding index will be all-zeros.
            if word not in word2vec.vocab:
                continue
            embedding_matrix[i] = word2vec.word_vec(word)

        del word2vec

        embedding_matrix = torch.FloatTensor(embedding_matrix)
        if self.use_cuda: embedding_matrix = embedding_matrix.cuda()

        return embedding_matrix
