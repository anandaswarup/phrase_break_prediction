"""Evaluate the trained model"""

import argparse
import os

import numpy as np
import torch
from sklearn.metrics import f1_score

from model import PhraseBreakPredictor
from utils import Config


def load_trained_model(model_checkpoint, model, device):
    """Load checkpoint from specified path
    """
    checkpoint = torch.load(model_checkpoint, map_location=device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model


def load_sentences_tags(sentences_file_path, tags_file_path):
    """Loads sentences and tags from the corresponding files
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


def load_vocabs(vocab_dir):
    """Load vocab files
    """
    # Loading dataset params
    dataset_params_path = os.path.join(vocab_dir, "dataset_params.json")
    assert os.path.isfile(dataset_params_path), f"No json file found at {dataset_params_path}, run build_vocab.py"
    dataset_params = Config(dataset_params_path)

    # Loading vocabulary (to map words to indices)
    vocab_path = os.path.join(vocab_dir, "words.txt")
    vocab_map = {}
    with open(vocab_path, "r") as reader:
        for idx, word in enumerate(reader.read().splitlines()):
            vocab_map[word] = idx

    # Loading tags (to map tags to indices)
    tags_path = os.path.join(vocab_dir, "tags.txt")
    tag_map = {}
    with open(tags_path, "r") as reader:
        for idx, tag in enumerate(reader.read().splitlines()):
            tag_map[tag] = idx

    return dataset_params, vocab_map, tag_map


def evaluate_model(cfg, vocab_dir, data_dir, model_checkpoint):
    """Evaluate the trained model on the held-out test set
    """
    # Load vocabularies
    dataset_params, vocab_map, tag_map = load_vocabs(vocab_dir)

    # Specify the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    model = PhraseBreakPredictor(cfg, vocab_size=dataset_params.vocab_size, num_tags=dataset_params.num_tags)
    model = model.to(device)

    # Load the trained model
    model = load_trained_model(model_checkpoint, model, device)

    # Load the sentences and corresponding tags
    sentences_file_path = os.path.join(data_dir, "sentences.txt")
    tags_file_path = os.path.join(data_dir, "labels.txt")
    sentences, tags = load_sentences_tags(sentences_file_path, tags_file_path)

    for idx, (word_seq, tag_seq) in enumerate(zip(sentences, tags)):
        test_set_f1 = 0.0
        word_seq = torch.LongTensor(
            [vocab_map[word] if word in vocab_map else vocab_map[dataset_params.unk] for word in word_seq]
        ).unsqueeze(0)
        tag_seq = torch.LongTensor([tag_map[tag] for tag in tag_seq])

        word_seq, tag_seq = word_seq.to(device), tag_seq.to(device)

        with torch.no_grad():
            pred_tag_seq = model(word_seq)

            # Flatten labels
            tag_seq = tag_seq.data.cpu().numpy()
            tag_seq = tag_seq.ravel()

            # np.argmax gives the class predicted for each token by the model
            pred_tag_seq = pred_tag_seq.data.cpu().numpy()
            pred_tag_seq = np.argmax(pred_tag_seq, axis=1)

            test_set_f1 += f1_score(tag_seq, pred_tag_seq, average="micro")

    test_set_f1 += test_set_f1 / (idx + 1)
    test_set_f1 = round(test_set_f1 * 100, 2)
    print(f"Test set F1 score: {test_set_f1}")


if __name__ == "__main__":
    # Setup parser to parse and accept command line arguments
    parser = argparse.ArgumentParser(description="Evaluate the trained model on the held-out test set")

    parser.add_argument("--config_file", help="Configuration file (json file)", required=True)
    parser.add_argument("--vocab_dir", help="Directory containing the vocab files", required=True)
    parser.add_argument("--test_data_dir", help="Directory containing the test dataset", required=True)
    parser.add_argument("--model_checkpoint", help="Trained model checkpoint to use for eval", required=True)

    # Parse and get command line arguments
    args = parser.parse_args()

    # Load the configuration parameters from file
    assert os.path.isfile(args.config_file), f"No file found at {args.config_file}"
    cfg = Config(args.config_file)

    evaluate_model(cfg, args.vocab_dir, args.test_data_dir, args.model_checkpoint)
