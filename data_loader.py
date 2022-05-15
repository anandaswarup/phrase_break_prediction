"""Model data loader"""

import os

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
