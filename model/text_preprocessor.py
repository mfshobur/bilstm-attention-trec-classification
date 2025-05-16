from collections import Counter
import gensim
import gensim.downloader
import re
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

class TextPreprocessor:
    def __init__(
            self,
            max_vocab_size=10000,
            max_seq_length=100,
            embedding_dim=300,
            stopwords: set = None,
            ):
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.word_to_index = {"<PAD>": 0, "<UNK>": 1}
        self.index_to_word = {0: "<PAD>", 1: "<UNK>"}
        self.word_counts = Counter()
        self.vocab_size = 2  # Starting with PAD and UNK tokens
        self.stopwords = stopwords
        if self.stopwords:
            self.stopwords_intervention()
        self.embedding_matrix = None

        # Initialize NLP tools
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
    
    def get_vocab_from_texts(self, texts, use_stopwords=True, use_lemmatization=False, use_stemming=False):
        counter = Counter()
        for text in texts:
            clean_text = self.clean_text(
                text,
                use_stopwords=use_stopwords,
                use_lemmatization=use_lemmatization,
                use_stemming=use_stemming
            )
            tokens = clean_text.split()
            counter.update(tokens)

        vocab_words = [word for word, count in counter.most_common()]
        return vocab_words

    def download_model(
            self,
            model_name='word2vec-google-news-300',
            load_from_local=False,
            local_embedding_path='embedding_matrix.npy',
            local_vocab_path='vocab.pkl',
            save_embedding=False,
            filter_from_vocab=None
        ):
        if load_from_local:
            self.load_embedding_matrix(
                local_embedding_path=local_embedding_path,
                local_vocab_path=local_vocab_path
            )
        else:
            vectors = gensim.downloader.load(model_name)
            
            if filter_from_vocab:
                self.vocab_size += len(filter_from_vocab)
            else:
                self.vocab_size += len(vectors.key_to_index)

            self.embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim))

            self.embedding_matrix[1] = np.random.normal(scale=0.6, size=(self.embedding_dim,))

            if filter_from_vocab:
                for i, word in enumerate(filter_from_vocab):
                    if vectors.__contains__(word):
                        index = i + 2
                        self.embedding_matrix[index] = vectors[word]
                        self.word_to_index[word] = index
                        self.index_to_word[index] = word
            else:
                for i, word in enumerate(vectors.key_to_index):
                    index = i + 2
                    self.embedding_matrix[index] = vectors[word]
                    self.word_to_index[word] = index
                    self.index_to_word[index] = word

            if save_embedding:
                self.save_embedding_matrix(
                    local_embedding_path=local_embedding_path,
                    local_vocab_path=local_vocab_path
                )
            
            del vectors
            import gc
            gc.collect()
    
    def save_embedding_matrix(self, local_embedding_path, local_vocab_path):
        import pickle
        np.save(local_embedding_path, self.embedding_matrix)
        with open(local_vocab_path, 'wb') as f:
            pickle.dump((self.word_to_index, self.index_to_word), f)

    def load_embedding_matrix(self, local_embedding_path, local_vocab_path):
        import pickle
        self.embedding_matrix = np.load(local_embedding_path)
        with open(local_vocab_path, 'rb') as f:
            self.word_to_index, self.index_to_word = pickle.load(f)
        self.vocab_size = len(self.index_to_word)

    def stopwords_intervention(self):
        pass
        # Customize stopword handling here if needed

    def clean_text(self, text, use_stopwords=True, use_lemmatization=False, use_stemming=False):
        """Basic text cleaning with optional lemmatization and stemming"""
        text = text.lower()
        text = re.sub(r"(?<!\w)[`“”‘’]+|[`“”‘’]+(?!\w)", "", text)
        text = re.sub(r'^[\'"]+(.+?)[\'"]+$', r'\1', text)
        text = re.sub(r"\s?'\s?", " ", text)
        text = re.sub(r"[?,]", "", text)
        text = re.sub(r"\s?''\s?", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        tokens = text.split()
        cleaned_tokens = []

        for token in tokens:
            if use_stopwords and self.stopwords and token in self.stopwords:
                continue
            if use_lemmatization:
                token = self.lemmatizer.lemmatize(token)
            if use_stemming:
                token = self.stemmer.stem(token)
            cleaned_tokens.append(token)

        return " ".join(cleaned_tokens)

    def fit(self, texts, use_stopwords=True, use_lemmatization=False, use_stemming=False):
        for text in texts:
            clean_text = self.clean_text(
                text,
                use_stopwords=use_stopwords,
                use_lemmatization=use_lemmatization,
                use_stemming=use_stemming
            )
            tokens = clean_text.split()
            self.word_counts.update(tokens)

        vocab_words = [word for word, count in self.word_counts.most_common()]

        for word in vocab_words:
            self.word_to_index[word] = self.vocab_size
            self.index_to_word[self.vocab_size] = word
            self.vocab_size += 1

        print(f"Vocabulary size: {self.vocab_size}")

    def transform(self, texts, use_stopwords=True, use_lemmatization=False, use_stemming=False):
        sequences = []
        for text in texts:
            clean_text = self.clean_text(
                text,
                use_stopwords=use_stopwords,
                use_lemmatization=use_lemmatization,
                use_stemming=use_stemming
            )
            tokens = clean_text.split()
            if len(tokens) > self.max_seq_length:
                tokens = tokens[:self.max_seq_length]
            seq = [self.word_to_index.get(word, self.word_to_index["<UNK>"]) for word in tokens]
            sequences.append(seq)
        return sequences
    
    def retransform(self, tokens):
        sequences = []
        for ids in tokens:
            seq = [self.index_to_word.get(id, self.index_to_word[1]) for id in ids]
            sequences.append(seq)
        return sequences

if __name__ == "__main__":
    pass
