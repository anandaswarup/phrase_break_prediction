"""Model data loader"""

import os
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils import Config


def load_sentences_tags(sentences_file_path, tags_file_path):
    """Loads sentences and tags from the corresponding files.
        Args:
            sentences_file_path (str): path to file with sentences
            tags_file_path (str): paath to file with phrase break tags for the sentences
    """
    sentences, tags = [], []

    with open(sentences_file_path, "r") as reader:
        for sentence in reader.read().splitlines():
            s = [word for word in sentence.split(" ")]
            sentences.append(s)

    with open(tags_file_path, "r") as reader:
        for sentence in reader.read().splitlines():
            s = [tag for tag in sentence.split(" ")]
            tags.append(s)

    # sanity checks (check if each token has an associated tag)
    assert len(tags) == len(sentences)
    for idx in range(len(tags)):
        assert len(tags[idx]) == len(sentences[idx])

    return sentences, tags


class TextTagDataset(Dataset):
    """Dataset class
        (1) Loads text, tag pairs
        (2) Returns sequences of ID's corresponding to texts and tags
    """

    def __init__(self, data_dir, split="train"):
        """Instantiate the dataset

            Args:
                data_dir (str): Directory containing the processed dataset
                split (str): The dataset split to process (one of "train", "dev" or "test)
        """
        # Loading dataset params
        dataset_params_path = os.path.join(data_dir, "vocab/dataset_params.json")
        assert os.path.isfile(dataset_params_path), f"No json file found at {dataset_params_path}, run build_vocab.py"
        self.dataset_params = Config(dataset_params_path)

        # Loading vocabulary (to map words to indices)
        vocab_path = os.path.join(data_dir, "vocab/words.txt")
        self.vocab_map = {}
        with open(vocab_path, "r") as reader:
            for idx, word in enumerate(reader.read().splitlines()):
                self.vocab_map[word] = idx

        # Loading tags (to map tags to indices)
        tags_path = os.path.join(data_dir, "vocab/tags.txt")
        self.tag_map = {}
        with open(tags_path, "r") as reader:
            for idx, tag in enumerate(reader.read().splitlines()):
                self.tag_map[tag] = idx

        # Setting indices for _UNK_ and _PAD_ (unknown words and padding)
        self.unk_idx = self.vocab_map[self.dataset_params.unk]
        self.pad_word_idx = self.vocab_map[self.dataset_params.pad_word]
        self.pad_tag_idx = self.tag_map[self.dataset_params.pad_tag]

        # Load the sentences and tags for the specified split
        sentences_file_path = os.path.join(data_dir, f"{split}/sentences.txt")
        tags_file_path = os.path.join(data_dir, f"{split}/labels.txt")
        self.sentences, self.tags = load_sentences_tags(sentences_file_path, tags_file_path)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence, tags = self.sentences[idx], self.tags[idx]

        words_seq = [self.vocab_map[word] if word in self.vocab_map else self.unk_idx for word in sentence]
        tags_seq = [self.tag_map[tag] for tag in tags]

        return (words_seq, tags_seq)

    def pad_collate(self, batch):
        """Create padded batches
        """
        sentences, tags = zip(*batch)

        sentences, tags = list(sentences), list(tags)

        sentence_lengths = [len(word_seq) for word_seq in sentences]
        tag_lengths = [len(tag_seq) for tag_seq in tags]

        sentences = pad_sequence(sentences, batch_first=True, padding_value=self.pad_word_idx)
        tags = pad_sequence(tags, batch_first=True, padding_value=self.pad_tag_idx)

        return torch.LongTensor(sentences), torch.LongTensor(tags), sentence_lengths, tag_lengths


# class DataLoader:
#     """Data loader class. Stores the dataset_params, vocabulary and tags with mapping to indices
#     """

#     def __init__(self, data_dir, cfg):
#         """Instantiate the data loader

#             Args:
#                 data_dir (str): directory containing the processed dataset
#                 cfg (params): configuration parameters
#         """
#         # Loading dataset params
#         dataset_params_path = os.path.join(data_dir, "vocab/dataset_params.json")
#         assert os.path.isfile(dataset_params_path), f"No json file found at {dataset_params_path}, run build_vocab.py"
#         self.dataset_params = Config(dataset_params_path)

#         # Loading vocabulary (to map words to indices)
#         vocab_path = os.path.join(data_dir, "vocab/words.txt")
#         self.vocab = {}
#         with open(vocab_path, "r") as reader:
#             for idx, word in enumerate(reader.read().splitlines()):
#                 self.vocab[word] = idx

#         # Setting indices for _UNK_ and _PAD_ (unknown words and padding)
#         self.unk_idx = self.vocab[self.dataset_params.unk]
#         self.pad_idx = self.vocab[self.dataset_params.pad_word]

#         # Loading tags (to map tags to indices)
#         tags_path = os.path.join(data_dir, "vocab/tags.txt")
#         self.tag_map = {}
#         with open(tags_path, "r") as reader:
#             for idx, tag in enumerate(reader.read().splitlines()):
#                 self.tag_map[tag] = idx

#         # Updating Configuration parameters with dataset parameters
#         cfg.update(dataset_params_path)

#     def load_sentences_tags(self, sentences_file_path, tags_file_path, data_dict):
#         """Loads sentences and tags from the corresponding files. Maps tokens (words and tags) to respective indices
#         and stores them in the data_dict

#             Args:
#                 sentences_file_path (str): path to file with sentences
#                 tags_file_path (str): paath to file with phrase break tags for the sentences
#                 data_dict (dict): dictionary where the loaded data is stored
#         """
#         sentences, tags = [], []

#         with open(sentences_file_path, "r") as reader:
#             for sentence in reader.read().splitlines():
#                 s = [self.vocab[token] if token in self.vocab else self.unk_idx for token in sentence.split(" ")]
#                 sentences.append(s)

#         with open(tags_file_path, "r") as reader:
#             for sentence in reader.read().splitlines():
#                 l = [self.tag_map[label] for label in sentence.split(" ")]
#                 tags.append(l)

#         # sanity checks (check if each token has an associated tag)
#         assert len(tags) == len(sentences)
#         for idx in range(len(tags)):
#             assert len(tags[idx]) == len(sentences[idx])

#         # Store the data in data_dict
#         data_dict["sentences"] = sentences
#         data_dict["tags"] = tags
#         data_dict["size"] = len(sentences)

#     def load_data(self, data_dir, splits):
#         """Loads the data for each split in data_dir

#             Args:
#                 data_dir (str): path to dir containing the processed dataset
#                 splits (list): the split to process ["train", "val", "test"]

#             Returns:
#                 data_dict (dict): dictionary containing sentences and corresponding tags for each split in data_dir
#         """
#         data = {}

#         for s in splits:
#             sentences_file_path = os.path.join(data_dir, f"{s}/sentences.txt")
#             tags_file_path = os.path.join(data_dir, f"{s}/labels.txt")
#             data[s] = {}
#             self.load_sentences_tags(sentences_file_path, tags_file_path, data[s])

#         return data

#     def data_iterator(self, data, cfg, shuffle=False):
#         """Returns a generator that yields batches of sentences with tags. Expires after one pass over the data

#             Args:
#                 data (dict): dictionary containing data with keys "sentences", "tags" and "size"
#                 batch_size (int): batch_size
#                 shuffle (bool): whether the data should be shuffled

#             Returns:
#                 batch_sentences (torch.LongTensor): paddded batched sentence data [B, T_max]
#                 batch_tags (torch.LongTensor): padded batched tags data [B, T_max]
#         """
#         order = list(range(data["size"]))
#         if shuffle:
#             random.seed(1234)
#             random.shuffle(order)

#         # perform one pass over the data
#         for i in range((data["size"] + 1) // cfg.batch_size):
#             # fetch sentences and tags
#             batch_sentences = [data["sentences"][idx] for idx in order[i * cfg.batch_size : (i + 1) * cfg.batch_size]]
#             batch_tags = [data["tags"][idx] for idx in order[i * cfg.batch_size : (i + 1) * cfg.batch_size]]

#             # length of longest sentence in batch
#             max_len = max([len(s) for s in batch_sentences])

#             # Create a numpy array with the data, initialising the data with pad_ind and all labels with -1
#             # initialising labels to -1 differentiates tokens with tags from PADding tokens
#             x = self.pad_idx * np.ones((len(batch_sentences), max_len))
#             y = -1 * np.ones((len(batch_sentences), max_len))

#             # Copy data to numpy array
#             for j in range(len(batch_sentences)):
#                 current_length = len(batch_sentences[j])
#                 x[j][:current_length] = batch_sentences[j]
#                 y[j][:current_length] = batch_tags[j]

#             # Convert them to PyTorch tensors
#             x, y = torch.LongTensor(x), torch.LongTensor(y)

#             yield x, y
