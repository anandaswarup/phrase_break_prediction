"""Train and evaluate the BLSTM token classification model using task specific word embeddings trained from scratch"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from data.blstm_data_loader import PhraseBreakDataset
from model.blstm import PhraseBreakPredictor
from utils.utils import save_checkpoint, load_json_to_dict, save_dict_to_json


def _write_log_to_file(log, filename):
    """Write the training log to file by writing one log item per line"""
    with open(filename, "w") as file_writer:
        for line in log:
            file_writer.write(line + "\n")

    print(f"Written log file: {filename}")


def _remove_predictions_for_padded_tokens(pred_puncs, puncs):
    """Remove predictions corresponding to padded tokens"""
    pred_puncs_without_padded = []
    puncs_without_padded = []

    for pred_punc, punc in zip(pred_puncs, puncs):
        if punc > -1:
            pred_puncs_without_padded.append(pred_punc)
            puncs_without_padded.append(punc)

    return pred_puncs_without_padded, puncs_without_padded


def _eval(model, device, loader):
    """Evaluate the model"""
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

            # Remove predictions corresponding to padded tokens
            pred_puncs = list(pred_puncs.cpu().numpy())
            puncs = list(puncs.cpu().numpy())
            pred_puncs, puncs = _remove_predictions_for_padded_tokens(pred_puncs, puncs)

            puncs_predictions += pred_puncs
            puncs_correct += puncs

    model.train()

    return f1_score(puncs_correct, puncs_predictions, average="micro")


def train_and_evaluate_model(cfg, dataset_dir, experiment_dir):
    """Train the model and periodically evaluate it on the dev set"""
    # Set all random seeds (for reproducibility)
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1234)

    # Create all directories
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Instantiate the dataloaders for the train and dev splits
    train_dataset = PhraseBreakDataset(dataset_dir, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=8,
        collate_fn=train_dataset.pad_collate,
        pin_memory=True,
        drop_last=True,
    )

    dev_dataset = PhraseBreakDataset(dataset_dir, split="dev")
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=1,
        collate_fn=dev_dataset.pad_collate,
        pin_memory=False,
        drop_last=False,
    )

    # Specify the device to be used for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantite the model
    model = PhraseBreakPredictor(
        num_words=len(train_dataset.word_vocab),
        word_embedding_dim=cfg["word_embedding_dim"],
        num_blstm_layers=cfg["num_blstm_layers"],
        blstm_layer_size=cfg["blstm_layer_size"],
        num_puncs=len(train_dataset.punc_vocab),
        padding_idx=train_dataset.word_pad_idx,
    )
    model = model.to(device)

    # Instantiate the optimizer
    optimizer = optim.adam(model.parameters(), lr=cfg["lr"])

    # Specify the criterion to train the model
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    model.train()

    best_dev_F1_score = 0.0
    training_log = []
    for epoch in range(0, cfg["num_epochs"]):
        # Train step
        puncs_predictions, puncs_correct = [], []
        for texts, puncs in train_loader:
            # Place the tensors on the appropriate device
            texts, puncs = texts.to(device), puncs.to(device)

            # Clear all previous gradients
            optimizer.zero_grad()

            # Forward pass (predictions)
            pred_probs = model(texts)
            pred_probs = pred_probs.view(-1, pred_probs.shape[2]).contiguous()

            # Reshape ground truth
            puncs = puncs.view(-1).contiguous()

            # Loss Computation
            loss = criterion(pred_probs, puncs)

            # Backward pass (Gradient computation and weight update)
            loss.backward()
            optimizer.step()

            # Find the class predicted for each token by the model
            _, pred_puncs = torch.max(pred_probs, 1)

            # Remove predictions corresponding to padded tokens
            pred_puncs = list(pred_puncs.cpu().numpy())
            puncs = list(puncs.cpu().numpy())
            pred_puncs, puncs = _remove_predictions_for_padded_tokens(pred_puncs, puncs)

            puncs_predictions += pred_puncs
            puncs_correct += puncs

        train_F1_score = f1_score(puncs_correct, puncs_predictions, average="micro")
        train_F1_score = train_F1_score * 100

        # Eval step
        dev_F1_score = _eval(model, device, dev_loader)
        dev_F1_score = dev_F1_score * 100

        # Check if the dev_F1_score is better than the best_dev_F1_score
        is_best = dev_F1_score >= best_dev_F1_score
        if is_best:
            best_dev_F1_score = dev_F1_score
            best_epoch = epoch + 1

        # Save checkpoint after training for one epoch
        save_checkpoint(checkpoint_dir, model, optimizer, epoch + 1)

        # Log training for one epoch
        epoch_log_string = f"Epoch: {epoch + 1}, Train Set F1: {train_F1_score:.2f}, Dev Set F1: {dev_F1_score:.2f}"
        print(epoch_log_string)
        training_log.append(epoch_log_string)

    # Training summary
    summary_log_string = f"Best performing model - Epoch: {best_epoch}, F1 Score on Dev Set: {best_dev_F1_score:.2f}"
    print(summary_log_string)
    training_log.append(summary_log_string)

    return training_log


if __name__ == "__main__":
    # Setup parser to accept command line arguments
    parser = argparse.ArgumentParser(
        description="Train BLSTM token classification model using task specific word embeddings trained from scratch"
    )

    # Command line arguments
    parser.add_argument(
        "--config_file", help="Path to file containing the model/training configuration to be loaded", required=True
    )
    parser.add_argument("--dataset_dir", help="Directory containing the processed dataset", required=True)
    parser.add_argument("--experiment_dir", help="Directory where the training artifacts will be saved", required=True)

    # Parse and get the command line arguments
    args = parser.parse_args()

    config_file = args.config_file
    dataset_dir = args.dataset_dir
    experiment_dir = args.experiment_dir

    # Load configuration from file
    assert os.path.isfile(args.config_file), f"No config file found at {args.config_file}"
    cfg = load_json_to_dict(config_file)

    # Train the model and evaluate it periodically on the dev set
    training_log = train_and_evaluate_model(cfg, dataset_dir, experiment_dir)

    # Write training log and training configs to file
    _write_log_to_file(training_log, os.path.join(experiment_dir, "training_log.txt"))
    save_dict_to_json(cfg, os.path.join(experiment_dir, "config.json"))
