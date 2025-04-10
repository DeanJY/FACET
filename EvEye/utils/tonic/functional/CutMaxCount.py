import numpy as np
import torch


def cut_max_count(events: np.ndarray, maxcount: int = 10, normalize: bool = False):
    """
    Limit the number of events to maxcount and optionally normalize the array.

    Args:
        events (np.ndarray): The input array of events.
        maxcount (int): The maximum number of events allowed. Default is 1000.
        normalize (bool): Whether to normalize the array after cutting. Default is False.

    Return:
        None
    """
    np.clip(events, 0, maxcount, out=events)

    if normalize:
        events /= maxcount


def tensor_cut_max_count(
    events: torch.Tensor, maxcount: int = 10, normalize: bool = False
):
    # Clamp the values to be between 0 and maxcount
    events.clamp_(0, maxcount)

    # Normalize if required
    if normalize:
        events /= maxcount
