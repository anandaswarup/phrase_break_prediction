"""Data loader for fine tuned BERT model with a token classification head"""

import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from utils.utils import load_vocab_to_dict, read_dataset_file


class BERTPhraseBreakDataset(Dataset):
    """Dataset class
    (1) Loads text, punctuations pairs
    (2) Returns sequences of IDs corresponding to texts, attention masks and punctuations
    """

    def __init__(self, data_dir, tokenizer, split="train"):
        """Instantiate the dataset
        Args:
            data_dir (str): Directory containing the processed dataset
            tokenizer (transformers.BertTokenizer): The tokenizer to use on the text
            split (str): The dataset split to process ("train"/ "dev" / "test")
        """
        # The tokenizer to tokenize the sentences
        self.tokenizer = tokenizer

        vocab_dir = os.path.join(data_dir, "vocab")
        assert os.path.isdir(vocab_dir), f"Vocab dir does not exist, run build_vocab_blstm.py"

        # Load vocabulary for punctuations
        self.punc_vocab = load_vocab_to_dict(os.path.join(data_dir, "vocab/puncs.txt"))

        # Load sentence / punctuation sequences for the specified split
        self.sentences = read_dataset_file(os.path.join(data_dir, f"{split}/sentences.txt"))
        self.punctuations = read_dataset_file(os.path.join(data_dir, f"{split}/punctuations.txt"))
        assert len(self.sentences) == len(self.punctuations)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence, punctuations = self._tokenize_and_preserve_puncs(self.sentences[index], self.punctuations[index])

        text_seq = self.tokenizer.convert_tokens_to_ids(sentence)
        punc_seq = [self.punc_vocab[punc] for punc in punctuations]

        # Add [CLS], [SEP] tokens to text_seq (Required by BERT); and correspondingly _NONE_ to punc_seq
        text_seq = [self.tokenizer.cls_token_id] + text_seq + [self.tokenizer.sep_token_id]
        punc_seq = [self.punc_vocab["_NONE_"]] + punc_seq + [self.punc_vocab["_NONE_"]]

        return (torch.LongTensor(text_seq), torch.LongTensor(punc_seq))

    def _tokenize_and_preserve_puncs(self, sentence, puncs):
        """Word piece tokenization makes it difficult to match punctuations back up with individual word pieces.
        This function tokenizes each word one at a time so that it is easier to preserve the correct punctuation for
        each subword
        """
        tokenized_sentence = []
        punctuations = []

        for word, punc in zip(sentence, puncs):
            # Tokenize the word and count number of subwords the word is broken into
            tokenized_word = self.tokenizer.tokenize(word)
            num_subwords = len(tokenized_word)

            # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized_word)

            # Add the same punctuation fo the new list of punctuations num_subwords times
            punctuations.extend([punc] * num_subwords)

        return tokenized_sentence, punctuations

    def pad_collate(self, batch):
        """Create padded batches padded to the length of the longest sequence in the batch"""
        sentences, punctuations = zip(*batch)
        sentences, punctuations = list(sentences), list(punctuations)

        padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_punctuations = pad_sequence(punctuations, batch_first=True, padding_value=-1)

        attention_masks = [
            [float(word_idx > 0) for word_idx in padded_sentence] for padded_sentence in padded_sentences
        ]
        attention_masks = torch.FloatTensor(attention_masks)

        return padded_sentences, attention_masks, padded_punctuations
