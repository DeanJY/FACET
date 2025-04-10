import numpy as np


def normalize(data: np.array, min, max, n_bins: int = 1):
    """
    Normalize the array into n_bins.

    Args:
        data (np.array): The input array.
        n_bins (int): The number of bins to normalize the array. Default is 1.
    Return:
        data (np.array): The normalized array.
    Examples:
        >>> data = np.array([0, 1, 2, 3, 4, 5])
        >>> normalize(data, 2)
        array([0.  0.4 0.8 1.2 1.6 2. ])
    """
    assert data.ndim == 1, "Data must be 1D."
    # min, max = data.min(), data.max()
    assert min < max, "Data is invalid."
    data = n_bins * (data - min) / (max - min)
    data.astype(np.float64)
    return data


def bilinear_interpolation(input: np.ndarray, scale: int, input_max: int):
    """
    Bilinear interpolation for the input array. Return the input, input_next, weight_input_now, weight_input_next.

    Args:
        input (np.ndarray): The input array.
        scale (int): The scale for interpolation.
        input_max (int): The maximum value for input.
    Return:
        input (np.ndarray): The input array.
        input_next (np.ndarray): The next input array.
        weight_input_now (np.ndarray): The weight for the current input.
        weight_input_next (np.ndarray): The weight for the next input.
    Examples:
        >>> input = np.array([0, 1, 2, 3, 4, 5])
        >>> bilinear_interpolation(input, 2, 5)
        input: [0 0 1 1 2 2]
        input_next: [1 1 2 2 3 3]
        weight_input_now: [1.  0.  1.  0.5 1.  0.5]
        weight_input_next: [0.  1.  0.  0.5 0.  0.5]
    """
    weight_input_next = input % scale / scale
    weight_input_now = 1 - weight_input_next
    input = (input // scale).clip(0, input_max).astype(int)
    input_next = (input + 1).clip(0, input_max).astype(int)

    return input, input_next, weight_input_now, weight_input_next


def to_frame_stack_numpy(
    events: np.ndarray,
    sensor_size: tuple,  # (W, H, 2) representing the x, y, p
    n_time_bins: int = 1,
    mode: str = "causal_linear",  # "bilinear", "causal_linear", "nearest"
    start_time: int = None,
    end_time: int = None,
    weight: int = 1,
):
    assert "x" and "y" and "t" and "p" in events.dtype.names
    assert sensor_size[2] == 2

    frame_stack = np.zeros((n_time_bins, 2, sensor_size[1], sensor_size[0]), float)
    ts = events["t"].astype(int)
    if start_time is not None and end_time is not None:
        ts = normalize(ts, start_time, end_time, n_time_bins)
    xs = events["x"].astype(int)
    ys = events["y"].astype(int)
    ps = events["p"].astype(int)

    t_now, t_next, weight_t_now, weight_t_next = bilinear_interpolation(
        ts, 1, n_time_bins
    )

    if mode == "nearest":
        t = np.where(weight_t_now > weight_t_next, t_now, t_next)
        mask = t >= 0
        np.add.at(frame_stack, (t[mask], ps[mask], ys[mask], xs[mask]), 1 * weight)

    elif mode == "bilinear":
        mask = (t_now >= 0) & (t_now < n_time_bins)
        np.add.at(
            frame_stack,
            (t_now[mask], ps[mask], ys[mask], xs[mask]),
            weight_t_next[mask] * weight,
        )
        t_prev = t_now - 1
        np.add.at(
            frame_stack,
            (t_prev[mask], ps[mask], ys[mask], xs[mask]),
            weight_t_now[mask] * weight,
        )

    elif mode == "causal_linear":
        mask = (t_now >= 0) & (t_now < n_time_bins)
        np.add.at(
            frame_stack,
            (t_now[mask], ps[mask], ys[mask], xs[mask]),
            weight_t_next[mask] * weight,
        )

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return frame_stack


def main():
    input = np.array([0, 1, 2, 3, 4, 5])
    a, b, c, d = bilinear_interpolation(input, 2, 5)
    print(a, b, c, d)


if __name__ == "__main__":
    main()
