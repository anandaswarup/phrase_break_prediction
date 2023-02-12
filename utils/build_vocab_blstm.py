"""Build token vocabularies from the processed dataset for training word embeddings from scratch"""

import argparse
import os
from collections import Counter
from utils import save_dict_to_json


class VocabBuilder:
    """Class for building the vocabulary from the dataset"""

    def __init__(self, dataset_dir, token):
        """Instantiate the builder
        Args:
            dataset_dir (str): Path to the directory containing the dataset
            token (str): The token type for which to build the vocabulary [either words or puncs]
        """
        # Vocab builder attributes
        self.dataset_dir = dataset_dir
        self.token = token
        self.vocab = None

        # Padding and unknown tokens
        if token == "words":
            self._pad = "_PAD_"
            self._unk = "_UNK_"

    def _update_from_file(self, filename, c):
        """Update a counter from file
        Args:
            filename (str): Path to file containing text (one sentence per line)
            c (Counter): Counter object to update
        """
        with open(filename, "r") as file_reader:
            for _, line in enumerate(file_reader):
                c.update(line.strip().split())

    def _write_to_file(self, filename, c):
        """Write to file by writing one token per line
        Args:
            filename (str): Path to the file to write the vocabulary
            c (iterable): Iterable object that yields tokens to be written to file
        """
        with open(filename, "w") as file_writer:
            for token in c:
                file_writer.write(token + "\n")

    def build_vocabulary(self):
        """Build vocabulary from the processed dataset and export them to disk to use later"""
        print(f"Building {self.token} vocabulary from sequences")

        # Build the vocabulary from the train, dev and test sequences
        c = Counter()
        filename = "sentences.txt" if self.token == "words" else "punctuations.txt"
        self._update_from_file(os.path.join(self.dataset_dir, f"train/{filename}"), c)
        self._update_from_file(os.path.join(self.dataset_dir, f"dev/{filename}"), c)
        self._update_from_file(os.path.join(self.dataset_dir, f"test/{filename}"), c)

        # Only keep those tokens which occur atleast 10 times in the vocabulary. All tokens occurring less than 10 times
        # will be replaced by "_UNK_". Also the LibriTTS Label dataset has a <unk> token which is treated as "_UNK_".
        if self.token == "words":
            self.vocab = [token for token, count in c.items() if token != "<unk>" and count >= 10]
        elif self.token == "puncs":
            self.vocab = [token for token, _ in c.items()]

        # Add padding and unknown tokens to the vocabulary
        if self.token == "words":
            self.vocab.insert(0, self._unk)
            self.vocab.insert(0, self._pad)

    def save_vocabulary(self):
        """Save the vocabulary to file"""
        print(f"Exporting {self.token} vocabulary to file")
        if not os.path.exists(os.path.join(self.dataset_dir, "vocab/blstm")):
            os.makedirs(os.path.join(self.dataset_dir, "vocab/blstm"))

        self._write_to_file(os.path.join(self.dataset_dir, f"vocab/blstm/{self.token}.txt"), self.vocab)


if __name__ == "__main__":
    # Setup parser to accept command line arguments
    parser = argparse.ArgumentParser(
        description="Build token vocabularies from the processed dataset for training word embeddings from scratch"
    )

    # Command line arguments
    parser.add_argument("--dataset_dir", help="Directory containing the processed dataset")

    # Parse and get the command line arguments
    args = parser.parse_args()
    dataset_dir = args.dataset_dir

    # Create directory for saving vocabularies
    if not os.path.exists(os.path.join(dataset_dir, "vocab")):
        os.makedirs(os.path.join(dataset_dir, "vocab"))

    # Instantiate VocabBuilder for words and puncs
    words_vocab = VocabBuilder(dataset_dir=dataset_dir, token="words")
    puncs_vocab = VocabBuilder(dataset_dir=dataset_dir, token="puncs")

    words_vocab.build_vocabulary()
    words_vocab.save_vocabulary()

    puncs_vocab.build_vocabulary()
    puncs_vocab.save_vocabulary()

    # Save dataset/vocab properties to disk
    params = {
        "num_words": len(words_vocab.vocab),
        "num_puncs": len(puncs_vocab.vocab),
        "words_pad_token": words_vocab._pad,
        "words_unk_token": words_vocab._unk,
    }
    save_dict_to_json(params, os.path.join(dataset_dir, "vocab/blstm/params.json"))
