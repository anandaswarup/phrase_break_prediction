"""General utilities"""

import json
import os
import torch


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
    """Save model checkpoint
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


def load_checkpoint(checkpoint_path, model, optimizer):
    """Load model checkpoint from disk
    Args:
        checkpoint_path (str): Path to the checkpoint to be loaded
        model (torch model object): Model
        optimizer (torch optimizer object): Optimizer
    """
    print(f"Loading checkpoint: {checkpoint_path} from disk")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["epoch"]
