"""Data loader for BLSTM token classification model using task specific word embeddings trained from scratch"""

import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from utils.utils import load_json_to_dict, load_vocab_to_dict, read_dataset_file


class PhraseBreakDataset(Dataset):
    """Dataset class
    (1) Loads text, punctuations pairs
    (2) Returns sequences of IDs corresponding to texts and punctuations
    """

    def __init__(self, data_dir, split="train"):
        """Instantiate the dataset
        Args:
            data_dir (str): Directory containing the processed dataset
            split (str): The dataset split to process ("train"/ "dev" / "test")
        """
        vocab_dir = os.path.join(data_dir, "vocab")
        assert os.path.isdir(vocab_dir), f"Vocab dir does not exist, run build_vocab_blstm.py"

        # Load vocabularies for both words and punctuations
        self.word_vocab = load_vocab_to_dict(os.path.join(data_dir, "vocab/words.txt"))
        self.punc_vocab = load_vocab_to_dict(os.path.join(data_dir, "vocab/puncs.txt"))
        # Load dataset / vocabulary params
        self.params = load_json_to_dict(os.path.join(data_dir, "vocab/params.json"))

        # Load sentence / punctuation sequences for the specified split
        self.sentences = read_dataset_file(os.path.join(data_dir, f"{split}/sentences.txt"))
        self.punctuations = read_dataset_file(os.path.join(data_dir, f"{split}/punctuations.txt"))
        assert len(self.sentences) == len(self.punctuations)

        # Setting indices for _UNK_, _PAD_, and _X_ tokens
        self.word_unk_idx = self.word_vocab[self.params["words_unk_token"]]
        self.word_pad_idx = self.word_vocab[self.params["words_pad_token"]]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence, punctuations = self.sentences[index], self.punctuations[index]

        text_seq = [self.word_vocab[word] if word in self.word_vocab else self.word_unk_idx for word in sentence]
        punc_seq = [self.punc_vocab[punc] for punc in punctuations]

        return (torch.LongTensor(text_seq), torch.LongTensor(punc_seq))

    def pad_collate(self, batch):
        """Create padded batches padded to the length of the longest sequence in the batch"""
        sentences, punctuations = zip(*batch)
        sentences, punctuations = list(sentences), list(punctuations)

        padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=self.word_pad_idx)
        padded_punctuations = pad_sequence(punctuations, batch_first=True, padding_value=-1)

        return padded_sentences, padded_punctuations
