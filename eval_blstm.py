"""Test the trained BLSTM token classification model using task specific word embeddings trained from scratch"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from data.blstm_data_loader import PhraseBreakDataset
from model.blstm import PhraseBreakPredictor
from utils.utils import load_checkpoint_to_evaluate_model, load_json_to_dict


def evaluate_model(cfg, dataset_dir, model_checkpoint):
    """Evaluate the trained model BLSTM token classification model on the test set"""
    # Instantiate the dataloader for the held-out test set
    dataset = PhraseBreakDataset(dataset_dir, split="test")
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=None, pin_memory=False, drop_last=False
    )

    # Specify the device to be used for the eval
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    model = PhraseBreakPredictor(
        num_words=len(dataset.word_vocab),
        word_embedding_dim=cfg["word_embedding_dim"],
        num_blstm_layers=cfg["num_blstm_layers"],
        blstm_layer_size=cfg["blstm_layer_size"],
        num_puncs=len(dataset.punc_vocab),
        padding_idx=dataset.word_pad_idx,
    )
    model = model.to(device)

    # Load the trained model from checkpoint
    model = load_checkpoint_to_evaluate_model(model_checkpoint, model, device)

    model.eval()
    with torch.no_grad():
        puncs_predictions, puncs_correct = [], []
        for texts, puncs in loader:
            # Place the tensors on the appropriate device
            texts, puncs = texts.to(device), puncs.to(device)

            # Forward pass (predictions)
            pred_probs = model(texts)
            pred_probs = pred_probs.view(-1, pred_probs.shape[2]).contiguous()

            # Reshape ground truth
            puncs = puncs.view(-1).contiguous()

            # Find the class predicted for each token by the model
            _, pred_puncs = torch.max(pred_probs, 1)

            pred_puncs = list(pred_puncs.cpu().numpy())
            puncs = list(puncs.cpu().numpy())

            puncs_predictions += pred_puncs
            puncs_correct += puncs

    test_set_F1_score = f1_score(puncs_correct, puncs_predictions, average="micro")

    print(f"F1 score on the Test Set: {test_set_F1_score:.2f}")


if __name__ == "__main__":
    # Setup command line parser to accept command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate the trained BLSTM token classification model on the held-out test set"
    )

    # Command line arguments
    parser.add_argument(
        "--config_file", help="Path to file containing the model/training configuration to be loaded", required=True
    )
    parser.add_argument("--dataset_dir", help="Directory containing the processed dataset", required=True)
    parser.add_argument(
        "--model_checkpoint",
        help="Path to the checkpoint containing the trained model to be used for eval",
        required=True,
    )

    # Parse and get command line arguments
    args = parser.parse_args()

    config_file = args.config_file
    dataset_dir = args.dataset_dir
    model_checkpoint = args.model_checkpoint

    # Load configuration from file
    assert os.path.isfile(args.config_file), f"No config file found at {args.config_file}"
    cfg = load_json_to_dict(config_file)

    # Evaluate the model on the test set
    evaluate_model(cfg, dataset_dir, model_checkpoint)
