import numpy as np
from gensim.models import KeyedVectors

class GetEmbedding(object):
    def __init__(self, word_index, vocab_size=10000, embedding_size=300):
        self.embedding_matrix = self.create_embed_matrix(word_index, vocab_size, embedding_size)

    def create_embed_matrix(self, word_index, vocab_size, embedding_size):
        print('Preparing Embedding Matrix.')

        file_name = './GoogleNews/GoogleNews-vectors-negative300.bin.gz'
        word2vec = KeyedVectors.load_word2vec_format(file_name, binary=True)
        # prepare embedding matrix
        num_words = min(vocab_size, len(word_index) + 1)
        embedding_matrix = np.zeros((num_words, embedding_size))

        for word, i in word_index.items():
            # words not found in embedding index will be all-zeros.
            if i >= vocab_size or word not in word2vec.vocab:
                continue
            embedding_matrix[i] = word2vec.word_vec(word)

        del word2vec
        return embedding_matrix
