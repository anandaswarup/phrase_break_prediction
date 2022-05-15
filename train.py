"""Train the model"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from data_loader import TextTagDataset
from model import PhraseBreakPredictor
from utils import Config, load_checkpoint, save_checkpoint, save_dict_to_json


def evaluate(model, device, dev_dataloader):
    """Evaluate the model on the dev set
    """
    model.eval()
    with torch.no_grad():
        dev_set_f1 = 0.0
        for idx, (sentences, tags) in enumerate(dev_dataloader):
            sentences, tags = sentences.to(device), tags.to(device)

            # Model predictions
            pred_tags = model(sentences)

            # Flatten labels
            tags = tags.data.cpu().numpy()
            tags = tags.ravel()

            # np.argmax gives the class predicted for each token by the model
            pred_tags = pred_tags.data.cpu().numpy()
            pred_tags = np.argmax(pred_tags, axis=1)

            dev_set_f1 += f1_score(tags, pred_tags, average="micro")
        dev_set_f1 = dev_set_f1 / (idx + 1)
    model.train()

    return dev_set_f1


def train_and_evaluate_model(cfg, data_dir, experiment_dir, resume_checkpoint_path):
    """Train the model
    """
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1234)

    # Create directories
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Prepare the train dataset and dataloader
    train_dataset = TextTagDataset(data_dir, split="train")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=train_dataset.pad_collate,
        pin_memory=True,
        drop_last=True,
    )

    # Prepare the dev dataset and dataloader
    dev_dataset = TextTagDataset(data_dir, split="dev")
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=dev_dataset.pad_collate,
        pin_memory=False,
        drop_last=False,
    )

    # Specify the device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    model = PhraseBreakPredictor(
        cfg, vocab_size=train_dataset.dataset_params.vocab_size, num_tags=train_dataset.dataset_params.num_tags
    )
    model = model.to(device)

    # Instantiate the optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # Load checkpoint and resume training from that point (if specified)
    if resume_checkpoint_path is not None:
        start_epoch = load_checkpoint(resume_checkpoint_path, model, optimizer)
    else:
        start_epoch = 0

    model.train()

    # Main training loop
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        train_set_f1 = 0.0
        for idx, (sentences, tags) in enumerate(train_dataloader):
            sentences, tags = sentences.to(device), tags.to(device)

            # Clear all previous gradients
            optimizer.zero_grad()

            # Forward pass and loss computation
            pred_tags = model(sentences)
            loss = F.nll_loss(pred_tags, tags)

            # Backward pass (gradient computation and weight update)
            loss.backward()
            optimizer.step()

            # Flatten labels
            tags = tags.data.cpu().numpy()
            tags = tags.ravel()

            # np.argmax gives the class predicted for each token by the model
            pred_tags = pred_tags.data.cpu().numpy()
            pred_tags = np.argmax(pred_tags, axis=1)

            train_set_f1 += f1_score(tags, pred_tags, average="micro")

        train_set_f1 = train_set_f1 / (idx + 1)

        # Evaluate the model
        dev_set_f1 = evaluate(model, device, dev_dataloader)

        # Log progress
        print(f"epoch: {epoch}, train set F1 score: {train_set_f1}, dev set F1 score: {dev_set_f1}")

        # Save checkpoint
        save_checkpoint(checkpoint_dir, model, optimizer, epoch)

    # Write dataset config and model/training config to file
    save_dict_to_json(cfg, os.path.join(experiment_dir, "config.json"))
    save_dict_to_json(train_dataset.dataset_params, os.path.join(experiment_dir, "dataset_params.json"))


if __name__ == "__main__":
    # Setup parser to accept command line arguments
    parser = argparse.ArgumentParser(description="Train the phrase break prediction model")

    # Command line arguments
    parser.add_argument("--config_file", help="Configuration file (json file)", required=True)
    parser.add_argument("--data_dir", help="Directory containing the processed dataset", required=True)
    parser.add_argument("--experiment_dir", help="Directory where training artifacts will be saved", required=True)
    parser.add_argument("--resume_checkpoint_path", help="If specified, load checkpoint and resume training")

    # Parse and get command line arguments
    args = parser.parse_args()

    # Load the configuration parameters from file
    assert os.path.isfile(args.config_file), f"No file found at {args.config_file}"
    cfg = Config(args.config_file)

    train_and_evaluate_model(cfg, args.data_dir, args.experiment_dir, args.resume_checkpoint_path)
