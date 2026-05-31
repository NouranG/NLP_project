from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
from .base import BaseVectorizer


class Embeddings(BaseVectorizer):
    def __init__(self, embedding_type: str, embedding_path: str,
                vector_size=100, window=5, min_count=2, workers=4):

        self.embedding_type = embedding_type
        self.embedding_path = embedding_path
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers

        self.model = None

    def fit(self, X):

        tokenized = [doc.split() for doc in X]

        if self.embedding_type == "word2vec":
                self.model = Word2Vec(
                    tokenized,
                    vector_size=self.vector_size,
                    window=self.window,
                    min_count=self.min_count,
                    workers=self.workers
                )

        elif self.embedding_type == "glove":
                word2vec_file = self.embedding_path + ".word2vec"

                glove2word2vec(self.embedding_path, word2vec_file)

                self.model = KeyedVectors.load_word2vec_format(word2vec_file)

            
        self.vector_size = self.model.vector_size
                    

    def get_embedding(self, word):
            if word in self.model:
                return self.model[word]

            return None

    def most_similar(self, word: str, topn=5):

        if self.model is None:
            raise ValueError(
                "Load embeddings first."
            )

        return self.model.most_similar(
            word,
            topn=topn
        )
    #document pooling
    
    def _doc_vector(self, text):
        vectors = []

        for word in text.split():
            vec = self.get_embedding(word)

            if vec is not None:
                vec = np.asarray(vec, dtype=np.float32)

                # safety check
                if vec.shape[0] != self.vector_size:
                    continue

                vectors.append(vec)

        if len(vectors) == 0:
            return np.zeros(self.vector_size, dtype=np.float32)

        return np.mean(vectors, axis=0)
    

    def transform(self, texts):
        vectors = [self._doc_vector(t) for t in texts]
        return np.vstack(vectors).astype(np.float32)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)