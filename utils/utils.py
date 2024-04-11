"""General utilities"""

import json
import os

import torch


def load_vocab_to_dict(filename):
    """Load vocabulary file to python dictionary
    Args:
        filename (str): Path to the vocabulary file to load
    """
    d = {}
    with open(filename, "r") as file_reader:
        for idx, token in enumerate(file_reader.read().splitlines()):
            d[token] = idx

    return d


def read_dataset_file(filename):
    """Read text file and return contents as a list
    Args:
        filename (str): Path to the dataset file to read
    """
    d = []
    with open(filename, "r") as file_reader:
        for line in file_reader.read().splitlines():
            s = [word for word in line.split()]
            d.append(s)

    return d


def load_json_to_dict(filename):
    """Load json file to python dictionary
    Args:
        filename (str): Path to the json file to load
    """
    with open(filename, "r") as file_reader:
        d = json.load(file_reader)

    return d


def save_dict_to_json(d, filename):
    """Save python dictionary to json file
    Args:
        d (dict): Python dictionary to write to file
        filename (str): Path to the json file to write the dictionary
    """
    with open(filename, "w") as file_writer:
        d = {key: value for key, value in d.items()}
        json.dump(d, file_writer, indent=4)


def save_checkpoint(checkpoint_dir, model, optimizer, epoch):
    """Save model checkpoint to disk
    Args:
        checkpoint_dir (str): Location where checkpoints will be written to disk
        model (torch model object): Model
        optimizer (torch optimizer object): Optimizer
        epoch (int): Current model training epoch
    """
    checkpoint_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch{epoch:04d}.pth")
    torch.save(checkpoint_state, checkpoint_path)


def load_checkpoint_to_evaluate_model(checkpoint_path, model, device):
    """Load trained model from specified path (to test on held-out set)
    Args:
        checkpoint_path (str): Path to the model checkpoint to load
        model (torch.nn.Module): Model
        device (torch.device): Device on which to load the trained model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model"], strict=False)

    return model
