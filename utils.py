"""Utilities"""

import json
import os

import torch


class Config:
    """Class that loads configuration parameters from json file
    """

    def __init__(self, filename):
        """Instantiate the class
        """
        with open(filename, "r") as file_reader:
            cfg = json.load(file_reader)
            self.__dict__.update(cfg)

    def save(self, filename):
        """Write configuration dictionary to json file
        """
        with open(filename, "w") as file_writer:
            json.dump(self.__dict__, file_writer, indent=4)

    def update(self, filename):
        """Loads configuration from json file
        """
        with open(filename, "r") as file_reader:
            cfg = json.load(file_reader)
            self.__dict__.update(cfg)

    @property
    def dict(self):
        """Gives a dictionary like acceess to Config instance
        """
        return self.__dict__


def save_dict_to_json(d, filename):
    """Save python dictionary to json file
    """
    with open(filename, "w") as file_writer:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, file_writer, indent=4)


def save_checkpoint(checkpoint_dir, model, optimizer, epoch):
    """Saves model and training parameters (state dict) as a checkpoint
    """
    checkpoint_state = {
        "model": model.state_dict(),
        "optmizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch{epoch:04d}.pth")

    torch.save(checkpoint_state, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer):
    """Loads model parameters (state dict) from checkpoint_path
    """
    print(f"Loading checkpoint: {checkpoint_path} from disk")

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["epoch"]
