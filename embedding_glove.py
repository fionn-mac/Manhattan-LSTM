import numpy as np

class GetEmbedding(object):
    def __init__(self, word_index, glove_dir='./Glove/', vocab_size=10000, embedding_size=300):
        self.embeddings_index = dict()
        self.create_embed_ind(glove_dir, embedding_size)
        self.embedding_matrix = self.create_embed_matrix(word_index, vocab_size, embedding_size)

    def create_embed_ind(self, glove_dir, embedding_size):
        print('Creating Embedding Index.')
        file_name = 'glove.6B.' + str(embedding_size) + 'd.txt'
        with open(os.path.join(glove_dir, file_name)) as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = coefs

        print('Found %s word vectors.' % len(self.embeddings_index))

    def create_embed_matrix(self, word_index, vocab_size, embedding_size):
        print('Preparing Embedding Matrix.')

        # prepare embedding matrix
        num_words = min(vocab_size, len(word_index))
        embedding_matrix = np.zeros((num_words, embedding_size))

        for word, i in word_index.items():
            if i >= vocab_size:
                continue
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        return embedding_matrix
