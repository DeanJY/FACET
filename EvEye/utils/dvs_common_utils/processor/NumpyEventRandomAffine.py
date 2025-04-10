import math
import numpy as np
import torch
from torch.nn import functional as F


def rand_range(amin, amax):
    """Return a random number in the range [amin, amax]"""
    return amin + (amax - amin) * np.random.rand()


def temporal_shift(start_time, time_window, shift_ratio=0.1):
    """
    Return a shifted time window and the offset

    Args:
        start_time (float): start time of the time window
        time_window (float): time window
        shift_ratio (float): ratio of the time window to shift
    Returns:
        start_time (float): shifted start time
        end_time (float): shifted end time
        offset (float): offset
    """
    end_time = start_time + time_window
    max_offset = round(time_window * shift_ratio)
    if start_time - max_offset >= 0:
        offset = np.random.rand() * max_offset
    else:
        offset = 0
    start_time -= offset
    end_time -= offset
    return start_time, end_time, offset


def temporal_scale(event, num_frames, end_time, time_window, scale_ratio=0.8):
    if end_time - num_frames * time_window * scale_ratio < 0:
        scale_factor = float(rand_range(0.8, 1.2))
    else:
        scale_factor = 1
    event[-1] *= scale_factor  # event['t'] *= scale_factor
    return event, scale_factor


class NumpyEventRandomAffine:
    """
    Perform random affine transformations on the events and labels

    Args:
        height (int): height of the event
        width (int): width of the event
        degrees (float): rotation angle in degrees
        translate (tuple): maximum translation in x and y directions
        scale (tuple): range of scaling factor
        spatial_jitter (float): spatial jitter
        augment_flag (bool): whether to apply the augmentation
    Returns:
        events (np.ndarray): transformed events
    """

    def __init__(
        self,
        size=(260, 346),
        degrees=15,
        translate=(0.2, 0.2),
        scale=(0.8, 1.2),
        spatial_jitter=None,
        augment_flag=True,
    ):
        self.height, self.width = size
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.spatial_jitter = spatial_jitter
        self.augment_flag = augment_flag

    def normalize(self, coords, backward=False):
        if not backward:
            coords[0] = coords[0] / self.width - 0.5
            coords[1] = coords[1] / self.height - 0.5
        else:
            coords[0] = (coords[0] + 0.5) * self.width
            coords[1] = (coords[1] + 0.5) * self.height

        return coords

    def __call__(self, events, labels=None):
        """
        Apply random affine transformations on the events and labels

        Args:
            events (np.ndarray): events with shape (4, N)
            labels (np.ndarray): labels with shape (N, 3)
        Returns:
            events (np.ndarray): transformed events
            centers (np.ndarray): transformed centers x,y
            closes (np.ndarray): transformed closes
        """
        if self.augment_flag:
            degrees = rand_range(-self.degrees, self.degrees) / 180 * math.pi
            translate = [rand_range(-t, t) for t in self.translate]
            scale = [rand_range(*self.scale) for _ in range(2)]

            cos, sin = math.cos(degrees), math.sin(degrees)
            R = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=float)
            S = np.array([[scale[0], 0, 0], [0, scale[1], 0], [0, 0, 1]], dtype=float)
            T = np.array(
                [[1, 0, translate[0]], [0, 1, translate[1]], [0, 0, 1]], dtype=float
            )

            trans_matrix = np.dot(np.dot(T, R), S)

        else:
            trans_matrix = np.eye(3, dtype=float)

        coords = np.pad(events[1:3], ((0, 1), (0, 0)), constant_values=1)
        coords = self.normalize(np.dot(trans_matrix, self.normalize(coords)), True)
        if self.spatial_jitter is not None:
            coords[:2] += np.random.randn(2, coords.shape[1]) * self.spatial_jitter

        events[1:3] = coords[:2]
        val_inds = (
            (coords[0] >= 0)
            & (coords[0] < self.width)
            & (coords[1] >= 0)
            & (coords[1] < self.height)
        )
        events = events[:, val_inds]
        if labels is None:
            return events
        else:
            labels = labels.T
            centers = np.pad(labels[0:2], ((0, 1), (0, 0)), constant_values=1)
            centers = np.dot(trans_matrix, self.normalize(centers))
            centers = centers[:2] + 0.5

            closes = labels[-1]

            return events, centers.T, closes


def main():
    augment = EventRandomAffine(height=480, width=640)
    pxyt = torch.tensor(
        [
            [1, 0, 1, 0, 1],
            [320, 319, 318, 317, 316],
            [240, 239, 238, 237, 236],
            [0, 1, 2, 3, 4],
        ],
        dtype=torch.float,
    )
    label = torch.tensor(
        [[320, 240, 0], [319, 239, 0], [318, 238, 0], [317, 237, 0], [316, 236, 0]],
        dtype=torch.float,
    )
    events, centers, closes = augment(pxyt, label)
    print(f"events: {events}")
    print(f"centers: {centers}")
    print(f"closes: {closes}")


if __name__ == "__main__":
    main()
