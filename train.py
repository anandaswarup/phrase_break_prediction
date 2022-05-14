"""Train the model"""

import argparse
import os

import numpy as np
import torch
import torch.optim as optim

from data_loader import DataLoader
from model import PhraseBreakPredictor, f1_measure, loss_fn
from utils import Config, load_checkpoint, save_checkpoint, save_dict_to_json


def evaluate(model, device, val_data_iterator, num_val_steps):
    """Evaluate the model on the dev set
    """
    # Set model to eval mode
    model.eval()
    with torch.no_grad():
        val_f1_score = 0.0

        for idx in range(num_val_steps):
            # Fetch the next val batch
            x_val, y_val = next(val_data_iterator)
            x_val, y_val = x_val.to(device), y_val.to(device)

            y_val_pred = model(x_val)
            f1_score = f1_measure(y_val_pred.data.cpu().numpy(), y_val.data.cpu().numpy())

            val_f1_score += f1_score

        val_f1_score = val_f1_score / (idx + 1)
    model.train()

    return val_f1_score


def train_epoch(model, optimizer, device, train_data_iterator, num_train_steps):
    """Train the model for one epoch
    """
    # set model to training mode
    model.train()

    train_loss = 0.0
    train_f1_score = 0.0
    for idx in range(num_train_steps):
        # Fetch the next training batch
        x, y = next(train_data_iterator)
        x, y = x.to(device), y.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Compute model output
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        f1_score = f1_measure(y_pred.data.cpu().numpy(), y.data.cpu().numpy())

        # Gradient computation and weight update
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_f1_score += f1_score

    train_loss = train_loss / (idx + 1)
    train_f1_score = train_f1_score / (idx + 1)

    return train_loss, train_f1_score


def train_and_evaluate_model(cfg, data_dir, experiment_dir):
    """Train the model
    """
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1234)

    # Create the directories
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Specify the training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the dataloader
    data_loader = DataLoader(data_dir, cfg)
    data = data_loader.load_data(data_dir, ["train", "dev"])
    train_data = data["train"]
    dev_data = data["dev"]

    # Get the sizes of train and dev datatsets and write it to cfg
    cfg.train_size = train_data["size"]
    cfg.dev_size = dev_data["size"]

    # Instantiate the model
    model = PhraseBreakPredictor(cfg)
    model = model.to(device)

    # Instantiate the optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    val_f1_scores = {}

    for epoch in range(cfg.num_epochs):
        # Train for one epoch (one full pass over the training set)
        num_train_steps = (cfg.train_size + 1) // cfg.batch_size
        train_data_iterator = data_loader.data_iterator(train_data, cfg, shuffle=True)
        train_loss, train_f1_score = train_epoch(model, optimizer, device, train_data_iterator, num_train_steps)

        # Evaluate the model after each epoch on the dev  set
        num_val_steps = (cfg.dev_size + 1) // cfg.batch_size
        val_data_iterator = data_loader.data_iterator(dev_data, cfg, shuffle=False)
        val_f1_score = evaluate(model, device, val_data_iterator, num_val_steps)
        val_f1_scores[epoch] = val_f1_score

        # Log training params
        print(f"Epoch: {epoch}, Loss: {train_loss}, Train F1 score: {train_f1_score},  Val F1 score: {val_f1_score}")

        # Save checkpoint
        save_checkpoint(checkpoint_dir, model, optimizer, epoch)

    # Write F1 scores to json file
    json_file_path = os.path.join(experiment_dir, "val_f1_scores.json")
    save_dict_to_json(val_f1_scores, json_file_path)


if __name__ == "__main__":
    # Setup parser to accept command line arguments
    parser = argparse.ArgumentParser(description="Train the phrase break prediction model")

    # Command line arguments
    parser.add_argument("--config_file", help="Configuration file (json file)", required=True)
    parser.add_argument("--data_dir", help="Directory containing the processed dataset", required=True)
    parser.add_argument("--experiment_dir", help="Directory where training artifacts will be saved", required=True)

    # Parse and get command line arguments
    args = parser.parse_args()

    # Load the configuration parameters from file
    assert os.path.isfile(args.config_file), f"No file found at {args.config_file}"
    cfg = Config(args.config_file)

    train_and_evaluate_model(cfg, args.data_dir, args.experiment_dir)
