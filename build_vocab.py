"""Build vocabularies of words and tags from datasets"""

import argparse
import json
import os
from collections import Counter

# Padding and unknown tokens in the vocabulary
_pad_word = "_PAD_"
_pad_tag = "O"
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


if __name__ == "__main__":
    # Setup parser to accept command line arguments
    parser = argparse.ArgumentParser(description="Build vocabularies of words and tags from datasets")

    # Command line arguments
    parser.add_argument("--data_dir", help="Directory containing the dataset")

    # Parse and get the command line arguments
    args = parser.parse_args()

    # Build word vocab with train, dev and test datasets
    print("Building word vocabulary")
    words = Counter()
    train_set_sentences_size = update_vocab(os.path.join(args.data_dir, "train/sentences.txt"), words)
    dev_set_sentences_size = update_vocab(os.path.join(args.data_dir, "dev/sentences.txt"), words)
    test_set_sentences_size = update_vocab(os.path.join(args.data_dir, "test/sentences.txt"), words)

    # Build tag vocab with train, dev and test datasets
    print("Building tag vocabulary")
    tags = Counter()
    train_set_tags_size = update_vocab(os.path.join(args.data_dir, "train/labels.txt"), tags)
    dev_set_tags_size = update_vocab(os.path.join(args.data_dir, "dev/labels.txt"), tags)
    test_set_tags_size = update_vocab(os.path.join(args.data_dir, "test/labels.txt"), tags)

    # Sanity checks
    assert train_set_sentences_size == train_set_tags_size
    assert dev_set_sentences_size == dev_set_tags_size
    assert test_set_sentences_size == test_set_tags_size

    # Only keep words which occur atleast 10 times in the vocabulary. All words occurring less that 10 times will be 
    # replaced with "_UNK_". This will ensure some training data for the "_UNK_" token
    words = [token for token, count in words.items() if count >= 10]
    # Keep all tags
    tags = [token for token, _ in tags.items()]

    # Add pad tokens to the vocabulary
    words.append(_pad_word)
    tags.append(_pad_tag)

    # Add _UNK_ token for unknown words
    words.append(_unk)

    # Save the vocabularies to file
    print("Saving vocabularies to file")
    save_vocab_to_txt_file(words, os.path.join(args.data_dir, "words.txt"))
    save_vocab_to_txt_file(tags, os.path.join(args.data_dir, "tags.txt"))

    # Save dataset properties to disk as json
    dataset_params = {
        "train_size": train_set_sentences_size,
        "dev_size": dev_set_sentences_size,
        "test_size": test_set_sentences_size,
        "vocab_size": len(words),
        "num_tags": len(tags),
        "pad_word": _pad_word,
        "pad_tag": _pad_tag,
        "unk": _unk,
    }
    save_dict_to_json(dataset_params, os.path.join(args.data_dir, "dataset_params.json"))
