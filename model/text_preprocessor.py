from collections import Counter
import gensim
import gensim.downloader
import re
import numpy as np

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
    
    def get_vocab_from_texts(self, texts, use_stopwords=True):
        # Clean and tokenize all texts
        counter = Counter()
        for text in texts:
            clean_text = self.clean_text(text, use_stopwords)
            tokens = clean_text.split()
            
            counter.update(tokens)

        # Keep only the most common words (minus PAD and UNK which we already have)
        # vocab_words = [word for word, count in self.word_counts.most_common(self.max_vocab_size - 2)]
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

            # set embedding for <UNK>
            self.embedding_matrix[1] = np.random.normal(scale=0.6, size=(self.embedding_dim,))

            if filter_from_vocab:
                for i, word in enumerate(filter_from_vocab):
                    if vectors.__contains__(word):
                        index = i+2
                        self.embedding_matrix[index] = vectors[word]
                        self.word_to_index[word] = index
                        self.index_to_word[index] = word
            else:
                for i, word in enumerate(vectors.key_to_index):
                    index = i+2
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
    
    def save_embedding_matrix(self,
        local_embedding_path,
        local_vocab_path
        ):
        import pickle
        np.save(local_embedding_path, self.embedding_matrix)
        with open(local_vocab_path, 'wb') as f:
            pickle.dump((self.word_to_index, self.index_to_word), f)


    def load_embedding_matrix(self,
        local_embedding_path,
        local_vocab_path
        ):
        import pickle
        self.embedding_matrix = np.load(local_embedding_path)
        with open(local_vocab_path, 'wb') as f:
            self.word_to_index, self.index_to_word = pickle.load(f)

    def stopwords_intervention(self):
        self.stopwords.add("'s")
        self.stopwords.add("'t")
        self.stopwords.remove('who')
        self.stopwords.remove('where')

    def clean_text(self, text, use_stopwords):
        """Basic text cleaning"""
        # Convert to lowercase
        text = text.lower()
        # Remove unnecessary quotes (`` and '') but keep apostrophes in contractions
        text = re.sub(r"(?<!\w)[`“”‘’]+|[`“”‘’]+(?!\w)", "", text)

        # Remove starting and ending single or double quotes (like ' qigong ')
        text = re.sub(r'^[\'"]+(.+?)[\'"]+$', r'\1', text)

        # Remove standalone single quotes that are not part of contractions
        text = re.sub(r"\s?'\s?", " ", text)

        # Remove specific punctuation: question marks and commas
        text = re.sub(r"[?,]", "", text)

        # Remove remaining '' inside the text
        text = re.sub(r"\s?''\s?", " ", text)

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        # use stopword
        if use_stopwords and self.stopwords:
            new_tokens = []
            tokens = text.split()

            for token in tokens:
                if token not in self.stopwords:
                    new_tokens.append(token)

            return " ".join(new_tokens)
        return text

    def fit(self, texts):
        """Build vocabulary from texts"""
        # Clean and tokenize all texts
        for text in texts:
            clean_text = self.clean_text(text)
            tokens = clean_text.split()
            self.word_counts.update(tokens)

        # Keep only the most common words (minus PAD and UNK which we already have)
        # vocab_words = [word for word, count in self.word_counts.most_common(self.max_vocab_size - 2)]
        vocab_words = [word for word, count in self.word_counts.most_common()]

        # Create word to index mapping
        for word in vocab_words:
            self.word_to_index[word] = self.vocab_size
            self.index_to_word[self.vocab_size] = word
            self.vocab_size += 1

        print(f"Vocabulary size: {self.vocab_size}")

    def transform(self, texts, use_stopwords=True):
        """Convert texts to sequences of indices"""
        sequences = []
        for text in texts:
            clean_text = self.clean_text(text, use_stopwords)
            tokens = clean_text.split()
            # Truncate if longer than max_seq_length
            if len(tokens) > self.max_seq_length:
                tokens = tokens[:self.max_seq_length]

            # Convert tokens to indices
            seq = [self.word_to_index.get(word, self.word_to_index["<UNK>"]) for word in tokens]
            sequences.append(seq)

        return sequences

if __name__ == "__main__":
    pass