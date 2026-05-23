from typing import Literal, Union, Dict
from pathlib import Path
import yaml
import random
import os

import numpy as np

import torch
import torch.nn.functional as F


__all__ = [
    "TASK",
    "exists",
    "default",
    "PathOrStr",
    "cycle",
    "zero_out_randomly",
    "get_base_model",
    "get_config",
    "save_config",
    "select_equally_distributed_numbers",
    "get_random_labels",
    "seed_everything"
]


TASK = Literal["classification", "regression"]

PathOrStr = Union[str, Path]


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def zero_out_randomly(
    tensor: torch.Tensor, probability: float, dim: int = 0
) -> torch.Tensor:
    """Zero out randomly selected elements of a tensor with a given probability at a given dimension

    Args:
        tensor: tensor to zero out
        probability: probability of zeroing out an element
        dim: dimension along which to zero out elements

    Returns:
        torch.Tensor: tensor with randomly zeroed out elements
    """

    mask = torch.rand(tensor.shape[dim]) < probability
    tensor[mask] = 0
    return tensor


def get_base_model(model):
    if hasattr(model, "module"):
        return model.module
    return model


def get_config(config):
    with open(config, "r") as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def save_config(config: Dict, yaml_file_path: PathOrStr) -> None:
    """save config to yaml file

    Args:
        config: config to save
        yaml_file_path: path to yaml file
    """

    try:
        with open(yaml_file_path, "w") as file:
            yaml.dump(config, file, sort_keys=False, default_flow_style=False)
    except Exception as e:
        print(f"Error saving dictionary to YAML file: {e}")


def select_equally_distributed_numbers(N: int, K: int) -> np.ndarray:
    if K >= N:
        return np.arange(0, N)     # can't have more steps than trained timesteps
    if K == 1:
        return np.array([N - 1])
    if N % K == 0:
        return np.arange(0, N, N // K)

    step = (N - 1) // (K - 1)
    return np.arange(0, N, step)[:K]


def get_random_labels(
    *,
    conditional: bool,
    task: TASK,
    n_classes: int,
    classifier_free_guidance: bool,
    n_labels: int,
    device,
) -> torch.Tensor | None:
    """Get random labels for a given task

    Args:
        conditional: if conditional generate labels, if not return None
        task: task to generate labels for
        n_classes:  number of classes for classification task
        classifier_free_guidance: if True, classification labels will be one-hot encoded
        n_labels: number of labels to generate
        device: device to use

    Returns:
        torch.Tensor | None: labels to generate or None
    """

    if not conditional:
        return None

    if task == "classification":
        labels = torch.randint(0, n_classes, size=(n_labels,), device=device)

        if classifier_free_guidance:
            labels = F.one_hot(labels, num_classes=n_classes).to(device=device).float()
    else:
        labels = torch.rand((n_labels, 1), device=device)

    return labels


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Setting all seeds to be {seed} to reproduce...")
