import numpy as np
from gensim.models import KeyedVectors

class Get_Embedding(object):
    def __init__(self, file_path, word_index):
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
        return embedding_matrix
