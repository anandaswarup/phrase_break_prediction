"""Train BLSTM token classification model using task specific word embeddings trained from scratch"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from data.blstm_data_loader import PhraseBreakDataset
from model.blstm import PhraseBreakPredictor
from utils.utils import save_checkpoint, load_checkpoint_to_evaluate_model, load_json_to_dict, save_dict_to_json


class Trainer:
    """Trainer class (Handles the model training process)"""

    def __init__(self, experiment_dir, model, optimizer, device=None):
        """Instantiate the trainer
        Args:
            experiment_dir (str): Folder where all training artifacts will be saved to disk
            model (torch.nn.Module): The model to be trained
            optimizer (torch.optim): Optimizer to be used to train the model
            device (str, optional): Device on which to train the model
        """
        self.model = model
        self.optimizer = optimizer
        self.device = self._get_device(device)

        # Send model to device
        self.model.to(self.device)

        # Create directories to save all training artifacts
        self.experiment_dir = experiment_dir
        self.checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Log
        self._log = []

    def _get_device(self, device=None):
        """Get the device on which to train the model"""
        if device is None:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = device

        return dev

    def _compute_loss(self, pred_labels, labels):
        """Compute cross-entropy loss between predicted and ground truth labels
        Args:
            pred_labels (torch.Tensor): Model predicted labels
            labels (torch.Tensor): Actual ground truth labels
        """
        # [B, T_max] -> [B * T_max]
        labels = labels.view(-1)

        # Mask out the "_PAD_" tokens
        mask = (labels >= 0).float()

        # Number of tokens is the sum of elements in the mask
        num_tokens = int(torch.sum(mask).data[0])

        # Apply mask to predictions
        pred_labels = pred_labels[range(pred_labels.shape[0]), labels] * mask

        # Compute cross-entropy loss for all non "_PAD_" tokens
        loss = -torch.sum(pred_labels) / num_tokens

        return loss

    def _compute_F1_score(self, pred_labels, labels):
        """Compute F1 score between predicted and ground truth labels
        Args:
            pred_labels (torch.Tensor): Model predicted labels
            labels (torch.Tensor): Actual ground truth labels
        """
        # Flatten ground truth labels
        labels = labels.cpu().numpy()
        labels = labels.ravel()

        # np.argmax gives the class predicted for each token by the model
        pred_labels = pred_labels.cpu().numpy()
        pred_labels = np.argmax(pred_labels, axis=1)

        return f1_score(labels, pred_labels, average="micro")

    def _write_log_to_file(self, filename):
        """Write log to file by writing one log item per line
        Args:
            filename (str): Path to the log file
        """
        with open(filename, "w") as file_writer:
            for line in self._training_log:
                file_writer.write(line + "\n")

        print(f"Written log file: {filename}")

    def _train(self, loader):
        """Train the model for one epoch"""
        self.model.train()

        f1_score = 0.0
        for idx, (sentences, punctuations) in enumerate(loader):
            # Place tensors on device
            sentences, punctuations = sentences.to(self.device), punctuations.to(self.device)

            # Clear all previous gradients
            self.optimizer.zero_grad()

            # Forward pass and loss computation
            predicted_punctuations = self.model(sentences)
            loss = self._compute_loss(predicted_punctuations, punctuations)

            # Backward pass (gradient computation and weight updates)
            loss.backward()
            self.optimizer.step()

            # Compute F1 score
            f1_score += self._compute_F1_score(predicted_punctuations, punctuations)

        f1_score = f1_score / (idx + 1)
        f1_score = round(f1_score * 100, 2)

        return f1_score

    def _eval(self, loader):
        """Evaluate the model"""
        self.model.eval()

        with torch.no_grad():
            f1_score = 0.0
            for idx, (sentences, punctuations) in enumerate(loader):
                # Place tensors on device
                sentences, punctuations = sentences.to(self.device), punctuations.to(self.device)

                # Model predictions
                predicted_punctuations = self.model(sentences)

                # Compute F1 score
                f1_score += self._compute_F1_score(predicted_punctuations, punctuations)

            f1_score = f1_score / (idx + 1)
            f1_score = round(f1_score * 100, 2)

        return f1_score

    def fit(self, train_loader, dev_loader, num_epochs):
        """Train the model
        Args:
            train_loader (torch.utils.data.DataLoader): Dataloader for training set
            dev_loader (torch.utils.data.DataLoader): Dataloader for dev set
            num_epochs (int): Number of epochs to train the model
        """
        # Set random seeds (for reproducibility)
        torch.manual_seed(1234)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(1234)

        best_epoch = 0
        best_dev_f1 = 0.0
        for epoch in range(0, num_epochs):
            # Train the model for one epoch
            train_f1_score = self._train(train_loader)

            # Evaluate the model after training for one epoch
            dev_f1_score = self._eval(dev_loader)

            # Check for the best dev F1 score
            if dev_f1_score > best_dev_f1:
                best_dev_f1 = dev_f1_score
                best_epoch = epoch

            # Save checkpoint after training for one epoch
            save_checkpoint(self.checkpoint_dir, self.model, self.optimizer, epoch + 1)

            # Log the training for one epoch
            epoch_log_string = f"epoch: {epoch + 1}, train F1 score: {train_f1_score}, dev F1 score: {dev_f1_score}"
            print(epoch_log_string)
            self._log.append(epoch_log_string)

        return best_dev_f1, best_epoch


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

    # Load configuration from file
    assert os.path.isfile(args.config_file), f"No config file found at {args.config_file}"
    cfg = load_json_to_dict(args.config_file)

    # Instantiate the training and dev dataloaders
    train_set = PhraseBreakDataset(args.dataset_dir, split="train")
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=8,
        collate_fn=train_set.pad_collate,
        pin_memory=True,
        drop_last=True,
    )

    dev_set = PhraseBreakDataset(args.dataset_dir, split="dev")
    dev_loader = DataLoader(
        dev_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=1,
        collate_fn=dev_set.pad_collate,
        pin_memory=False,
        drop_last=False,
    )

    # Instantiate the model
    model = PhraseBreakPredictor(
        num_words=len(train_set.word_vocab),
        word_embedding_dim=cfg["word_embedding_dim"],
        num_blstm_layers=cfg["num_blstm_layers"],
        blstm_layer_size=cfg["blstm_layer_size"],
        num_puncs=len(train_set.punc_vocab),
        padding_idx=train_set.pad_idx,
    )

    # Instantiate the optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    # Instantiate the trainer
    trainer = Trainer(args.experiment_dir, model, optimizer)

    # Train the model
    best_dev_f1, best_epoch = trainer.fit(train_loader, dev_loader, cfg["num_epochs"])

    # Log training summary
    training_summary_log_string = f"Training summary: best dev f1 score: {best_dev_f1} at epoch: {best_epoch + 1}"
    print(training_summary_log_string)
    trainer._log.append(training_summary_log_string)

    # Write logs and configs to file
    trainer._write_log_to_file(os.path.join(trainer.experiment_dir, "training_log.txt"))
    save_dict_to_json(cfg, os.path.join(trainer.experiment_dir, "config.json"))