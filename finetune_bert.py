"""Finetune BERT model with a token classification head for phrase break prediction"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from data.bert_data_loader import BERTPhraseBreakDataset
from model.bert import BERTPhraseBreakPredictor
from utils.utils import save_checkpoint, load_json_to_dict, save_dict_to_json


def _write_log_to_file(log, filename):
    """Write the training log to file by writing one log item per line"""
    with open(filename, "w") as file_writer:
        for line in log:
            file_writer.write(line + "\n")

    print(f"Written log file: {filename}")


def _eval(model, device, loader, num_puncs):
    """Evaluate the model"""
    model.eval()
    with torch.no_grad():
        puncs_predictions, puncs_correct = [], []
        for batch in loader:
            texts = batch["input_ids"].to(device)
            attention_masks = batch["attention_mask"].to(device)
            punctuations = batch["labels"].to(device)

            # Forward pass
            _, logits = model(texts=texts, attention_masks=attention_masks, punctuations=punctuations)

            # Flatten punctuations and logits
            punctuations = punctuations.view(-1)
            logits = logits.view(-1, num_puncs)
            predicted_punctuations = torch.argmax(logits, axis=1)

            # Ignore predictions made for labels with value -100 (padding tokens)
            mask = punctuations != 100
            punctuations = torch.masked_select(punctuations, mask)
            predicted_punctuations = torch.masked_select(predicted_punctuations, mask)

            puncs_predictions.extend(list(predicted_punctuations.cpu().numpy()))
            puncs_correct.extend(list(punctuations.cpu().numpy()))

    model.train()

    return f1_score(puncs_correct, puncs_predictions, average="micro")


def finetune_and_evaluate_model(cfg, dataset_dir, experiment_dir):
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
    train_dataset = BERTPhraseBreakDataset(cfg["bert_model_name"], dataset_dir, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    dev_dataset = BERTPhraseBreakDataset(cfg["bert_model_name"], dataset_dir, split="dev")
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        drop_last=False,
    )

    # Specify the device to be used for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantite the model
    num_puncs = len(train_dataset.punc_vocab)
    model = BERTPhraseBreakPredictor(cfg["bert_model_name"], num_puncs=num_puncs)
    model = model.to(device)

    # Instantiate the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    model.train()

    best_dev_F1_score = 0.0
    training_log = []
    for epoch in range(0, cfg["num_epochs"]):
        # Train step
        puncs_predictions, puncs_correct = [], []
        for batch in train_loader:
            texts = batch["input_ids"].to(device)
            attention_masks = batch["attention_mask"].to(device)
            punctuations = batch["labels"].to(device)

            # Clear all previous gradients
            optimizer.zero_grad()

            # Forward pass
            loss, logits = model(texts=texts, attention_masks=attention_masks, punctuations=punctuations)

            # Backward pass (Gradient computation and weight update)
            loss.backward()
            optimizer.step()

            # Flatten punctuations and logits
            punctuations = punctuations.view(-1)
            logits = logits.view(-1, num_puncs)
            predicted_punctuations = torch.argmax(logits, axis=1)

            # Ignore predictions made for labels with value -100 (padding tokens)
            mask = punctuations != 100
            punctuations = torch.masked_select(punctuations, mask)
            predicted_punctuations = torch.masked_select(predicted_punctuations, mask)

            puncs_predictions.extend(list(predicted_punctuations.cpu().numpy()))
            puncs_correct.extend(list(punctuations.cpu().numpy()))

        train_F1_score = f1_score(puncs_correct, puncs_predictions, average="micro")
        train_F1_score = train_F1_score * 100

        # Eval step
        dev_F1_score = _eval(model, device, dev_loader, num_puncs)
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
        description="Finetune BERT model with a token classification head for phrase break prediction"
    )

    # Command line arguments
    parser.add_argument(
        "--config_file", help="Path to file containing the finetuning configuration to be loaded", required=True
    )
    parser.add_argument("--dataset_dir", help="Directory containing the processed dataset", required=True)
    parser.add_argument(
        "--experiment_dir", help="Directory where the finetuning artifacts will be saved", required=True
    )

    # Parse and get the command line arguments
    args = parser.parse_args()

    config_file = args.config_file
    dataset_dir = args.dataset_dir
    experiment_dir = args.experiment_dir

    # Load configuration from file
    assert os.path.isfile(args.config_file), f"No config file found at {args.config_file}"
    cfg = load_json_to_dict(config_file)

    # Train the model and evaluate it periodically on the dev set
    training_log = finetune_and_evaluate_model(cfg, dataset_dir, experiment_dir)

    # Write training log and training configs to file
    _write_log_to_file(training_log, os.path.join(experiment_dir, "finetune_log.txt"))
    save_dict_to_json(cfg, os.path.join(experiment_dir, "config.json"))
