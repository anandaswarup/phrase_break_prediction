"""Build vocabularies of words and punctuations from the processed dataset for training word embeddings from scratch"""

import argparse
import json
import os
from collections import Counter

# Padding and unknown tokens in the vocabulary
_pad_word = "_PAD_"
_pad_punc = "_X_"
_unk = "_UNK_"


def save_vocab_to_txt_file(vocab, filename):
    """Saves the vocab to file by writing one token per line

        Args:
            vocab (iterable object): yields token
            filename (string): path to the file to write the vocab
    """
    with open(filename, "w") as file_writer:
        for token in vocab:
            file_writer.write(token + "\n")


def save_dict_to_json(d, filename):
    """Save python dictionary to json file

        Args:
            d (dictionary): python dictionary to write to file
            filename (string): path to the json file to write the dictionary
    """
    with open(filename, "w") as file_writer:
        d = {k: v for k, v in d.items()}
        json.dump(d, file_writer, indent=4)


def update_vocab(filename, vocab):
    """Update the vocabulary from file

        Args:
            filename (string): path to the file which contains one sentence per line
            vocab (dictionary or Counter): python object with update method
    """
    with open(filename, "r") as file_reader:
        for idx, line in enumerate(file_reader):
            vocab.update(line.strip().split(" "))

    return idx + 1


def build_vocabulary(data_dir):
    """Build the vocabularies of words and tags from the processed dataset and export them to disk to be used later
    """
    # Build word vocab with train, dev and test datasets
    print("Building word vocabulary")
    words = Counter()
    train_set_sentences_size = update_vocab(os.path.join(data_dir, "train/sentences.txt"), words)
    dev_set_sentences_size = update_vocab(os.path.join(data_dir, "dev/sentences.txt"), words)
    test_set_sentences_size = update_vocab(os.path.join(data_dir, "test/sentences.txt"), words)

    # Build tag vocab with train, dev and test datasets
    print("Building tag vocabulary")
    puncs = Counter()
    train_set_puncs_size = update_vocab(os.path.join(data_dir, "train/puncs.txt"), puncs)
    dev_set_puncs_size = update_vocab(os.path.join(data_dir, "dev/puncs.txt"), puncs)
    test_set_puncs_size = update_vocab(os.path.join(data_dir, "test/puncs.txt"), puncs)

    # Sanity checks
    assert train_set_sentences_size == train_set_puncs_size
    assert dev_set_sentences_size == dev_set_puncs_size
    assert test_set_sentences_size == test_set_puncs_size

    # Only keep words which occur atleast 10 times in the vocabulary. All words occurring less 10 times will be
    # replaced with "_UNK_". Also the LibriTTS dataset a <unk> token. I am assuming that token also represents "_UNK_"
    # This will ensure some training data for the "_UNK_" token
    words = [token for token, count in words.items() if token != "<unk>" and count >= 10]
    # Keep all tags
    puncs = [token for token, _ in puncs.items()]

    # Add pad tokens and unknown tokens to the vocabulary
    words.insert(0, _unk)
    words.insert(0, _pad_word)
    puncs.insert(0, _pad_punc)

    # Save the vocabularies to file
    print("Saving vocabularies to file")
    if not os.path.exists(os.path.join(data_dir, "vocab")):
        os.makedirs(os.path.join(data_dir, "vocab"))

    save_vocab_to_txt_file(words, os.path.join(data_dir, "vocab/words.txt"))
    save_vocab_to_txt_file(puncs, os.path.join(data_dir, "vocab/tags.txt"))

    # Save dataset properties to disk as json
    dataset_params = {
        "train_size": train_set_sentences_size,
        "dev_size": dev_set_sentences_size,
        "test_size": test_set_sentences_size,
        "vocab_size": len(words),
        "num_puncs": len(puncs),
        "pad_word": _pad_word,
        "pad_punc": _pad_punc,
        "unk": _unk,
    }
    save_dict_to_json(dataset_params, os.path.join(data_dir, "vocab/dataset_params.json"))


if __name__ == "__main__":
    # Setup parser to accept command line arguments
    parser = argparse.ArgumentParser(description="Build vocabularies of words and tags from the processed dataset")

    # Command line arguments
    parser.add_argument("--data_dir", help="Directory containing the processed dataset")

    # Parse and get the command line arguments
    args = parser.parse_args()

    # Build the vocabularies of words and tags from the processed dataset and export them to disk to be used later
    build_vocabulary(args.data_dir)
