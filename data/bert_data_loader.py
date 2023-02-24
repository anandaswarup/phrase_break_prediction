"""Data loader for fine tuning BERT model with a token classification head on phrase break prediction"""

import os
import torch
from torch.utils.data import Dataset
from utils.utils import load_vocab_to_dict, read_dataset_file
from transformers import BertTokenizerFast
import numpy as np


class BERTPhraseBreakDataset(Dataset):
    """Dataset class for fine tuning BERT model on phrase break prediction"""

    def __init__(self, model_name, data_dir, split="train"):
        """Instantiate the dataset
        Args:
            data_dir (str): Directory containing the processed dataset
            split (str): The dataset split to process ("train"/ "dev" / "test")
        """
        # Load vocabulary for punctuations
        vocab_dir = os.path.join(data_dir, "vocab")
        assert os.path.isdir(vocab_dir), f"Vocab dir does not exist, run build_vocab_blstm.py"
        self.punc_vocab = load_vocab_to_dict(os.path.join(data_dir, "vocab/puncs.txt"))

        # Load sentence / punctuation sequences for the specified split
        self.sentences = read_dataset_file(os.path.join(data_dir, f"{split}/sentences.txt"))
        self.punctuations = read_dataset_file(os.path.join(data_dir, f"{split}/punctuations.txt"))
        assert len(self.sentences) == len(self.punctuations)

        # Instantiate the tokenizer to tokenize the sentences
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence, puncs = self.sentences[idx], self.punctuations[idx]

        return sentence, puncs

    def _align_and_encode_punctuations(self, sentence_encodings, punctuations):
        """Aligns the punctuations with word-pieces and encodes them"""
        encoded_punctuations = [[self.punc_vocab[punc] for punc in sequence] for sequence in punctuations]

        aligned_punctuations = []
        for punc_encodings, offset in zip(encoded_punctuations, sentence_encodings.offset_mapping):
            # Create an empty array of -100
            aligned_puncs = np.ones(len(offset), dtype=int) * -100
            array_offset = np.array(offset)

            # Set punctuations for word-pieces whose first offset position is 0 and second is not 0
            aligned_puncs[(array_offset[:, 0] == 0) & (array_offset[:, 1] != 0)] = punc_encodings
            aligned_punctuations.append(aligned_puncs.tolist())

        return aligned_punctuations

    def pad_collate(self, batch):
        """Create batches padded to length of the longest sequence in the batch"""
        sentences, punctuations = zip(*batch)

        # Tokenize and encode the sentences
        sentence_encodings = self.tokenizer(
            sentences, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True
        )

        # Encode the punctuations and align them with word-pieces
        punctuation_encodings = self._align_and_encode_punctuations(sentence_encodings, punctuations)

        item = {key: torch.tensor(val) for key, val in sentence_encodings.items()}
        item["labels"] = torch.tensor(punctuation_encodings)

        return item
