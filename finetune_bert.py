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


def _remove_predictions_for_padded_tokens(pred_puncs, puncs):
    """Remove predictions corresponding to padded tokens"""
    pred_puncs_without_padded = []
    puncs_without_padded = []

    for pred_punc, punc in zip(pred_puncs, puncs):
        if punc > -100:
            pred_puncs_without_padded.append(pred_punc)
            puncs_without_padded.append(punc)

    return pred_puncs_without_padded, puncs_without_padded


def _eval(model, device, loader):
    """Evaluate the model"""
    model.eval()

    with torch.no_grad():
        predictions, ground_truth = [], []
        for batch in loader:
            text_ids, attention_masks, punc_ids = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["labels"].to(device),
            )

            # Forward pass (predictions)
            logits = model(text_ids=text_ids, attention_masks=attention_masks)

            # Reshape logits and ground truth
            logits = logits.view(-1, logits.shape[2]).contiguous()
            punc_ids = punc_ids.view(-1).contiguous()

            # Convert logits to prediction ids
            _, pred_punc_ids = torch.max(logits, 1)

            # Remove predictions corresponding to paddings
            pred_punc_ids = list(pred_punc_ids.cpu().numpy())
            punc_ids = list(punc_ids.cpu().numpy())
            pred_punc_ids, punc_ids = _remove_predictions_for_padded_tokens(pred_punc_ids, punc_ids)

            predictions += pred_punc_ids
            ground_truth += punc_ids

    model.train()

    return f1_score(ground_truth, predictions, average="micro")


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

    # Instantiate the dataloaders for train/dev splits
    train_dataset = BERTPhraseBreakDataset(model_name=cfg["bert_model_name"], data_dir=dataset_dir, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=train_dataset.pad_collate,
        pin_memory=True,
        drop_last=True,
    )

    dev_dataset = BERTPhraseBreakDataset(model_name=cfg["bert_model_name"], data_dir=dataset_dir, split="dev")
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=dev_dataset.pad_collate,
        pin_memory=True,
        drop_last=True,
    )

    # Specify the device to be used for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    num_puncs = len(train_dataset.punc_vocab)
    model = BERTPhraseBreakPredictor(model_name=cfg["bert_model_name"], num_puncs=num_puncs)
    model = model.to(device)

    # Instantiate the optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg["lr"])

    # Specify the criterion to fine-tune the model
    criterion = torch.nn.CrossEntropyLoss()

    model.train()

    best_dev_F1_score = 0.0
    training_log = []
    for epoch in range(0, cfg["num_epochs"]):
        # Train step
        predictions, ground_truth = [], []
        for batch in train_loader:
            text_ids, attention_masks, punc_ids = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["labels"].to(device),
            )

            # Clear all previous gradients
            optimizer.zero_grad()

            # Forward pass (predictions)
            logits = model(text_ids=text_ids, attention_masks=attention_masks)

            # Reshape logits and ground truth
            logits = logits.view(-1, logits.shape[2]).contiguous()
            punc_ids = punc_ids.view(-1).contiguous()

            # Loss computation
            loss = criterion(logits, punc_ids)

            # Backward pass (Gradient computation, gradient clipping and weight update)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            optimizer.step()

            # Convert logits to prediction ids
            _, pred_punc_ids = torch.max(logits, 1)

            # Remove predictions corresponding to paddings
            pred_punc_ids = list(pred_punc_ids.cpu().numpy())
            punc_ids = list(punc_ids.cpu().numpy())
            pred_punc_ids, punc_ids = _remove_predictions_for_padded_tokens(pred_punc_ids, punc_ids)

            predictions += pred_punc_ids
            ground_truth += punc_ids

        train_F1_score = f1_score(ground_truth, predictions, average="micro")
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
        description="Finetune BERT model with a token classification head for phrase break prediction"
    )

    # Command line arguments
    parser.add_argument(
        "--config_file", help="Path to file containing the fine tuning configuration to be loaded", required=True
    )
    parser.add_argument("--dataset_dir", help="Directory containing the processed dataset", required=True)
    parser.add_argument("--experiment_dir", help="Directory where artifacts will be saved", required=True)

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
