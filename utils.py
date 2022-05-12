"""Utilities"""

import json
import os
import shutil

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


def save_checkpoint(checkpoint_state, is_best, checkpoint_dir):
    """Saves model and training parameters (state dict) as a checkpoint
    """
    filepath = os.path.join(checkpoint_dir, "last.pth")
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint dir does not exist. Making directory {checkpoint_dir}")
        os.makedirs(checkpoint_dir)
    else:
        print("Using existing checkpoint dir")

    torch.save(checkpoint_state, filepath)
    if is_best:
        shutil.copy(filepath, os.path.join(checkpoint_dir, "best.pth"))


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Loads model parameters (state dict) from checkpoint_path
    """
    if not os.path.exists(checkpoint_path):
        raise (f"Checkpoint {checkpoint_path} does not exist")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optim_dict"])
    return checkpoint
