import numpy as np


def NumpyFrameStack(
    events, size, num_frames, spatial_downsample, temporal_downsample, mode="bilinear"
):
    """
    Perform bilinear interpolation directly on the events,
    while converting them to frames and do spatial and temporal downsamplings
    all at the same time.
    """
    height, width = size
    p, x, y, t = events
    events_frames = np.zeros([num_frames, 2, height, width], dtype=np.float32)

    def bilinear_interp(x, scale, x_max):
        if scale == 1:
            return x, x, np.ones_like(x), np.zeros_like(x)
        xd1 = x % scale / scale
        xd = 1 - xd1
        x = (x / scale).astype(int).clip(0, x_max)
        x1 = (x + 1).clip(0, x_max)
        return x, x1, xd, xd1

    if mode == "nearest":
        p = np.round(p).astype(int)
        x = (x / spatial_downsample[0]).round().astype(int).clip(0, width - 1)
        y = (y / spatial_downsample[1]).round().astype(int).clip(0, height - 1)
        t = (t / temporal_downsample - 0.5).round().astype(int).clip(0, num_frames - 1)
        np.add.at(events_frames, (t, p, y, x), 1.0)
        return events_frames

    x, x1, xd, xd1 = bilinear_interp(x, spatial_downsample[0], width - 1)
    y, y1, yd, yd1 = bilinear_interp(y, spatial_downsample[1], height - 1)
    t, t1, td, td1 = bilinear_interp(t, temporal_downsample, num_frames - 1)

    # similar to bilinear, but temporally causal
    if mode == "causal_linear":
        p = np.repeat(p.astype(int), 4)

        x = np.concatenate([np.repeat(x, 2), np.repeat(x1, 2)])
        y = np.concatenate([y, y1]).repeat(2)
        t = np.repeat(t, 4)

        xd = np.concatenate([np.repeat(xd, 2), np.repeat(xd1, 2)])
        yd = np.concatenate([yd, yd1]).repeat(2)
        td = np.repeat(td1, 4)  # causal

        indices = (t, p, y, x)
        values = xd * yd * td
        np.add.at(events_frames, indices, values)
        return events_frames

    if mode == "bilinear":
        p = np.repeat(p.astype(int), 8)

        x = np.concatenate([np.repeat(x, 4), np.repeat(x1, 4)])
        y = np.concatenate([np.repeat(y, 2), np.repeat(y1, 2)]).repeat(2)
        t = np.concatenate([t, t1]).repeat(4)

        xd = np.concatenate([np.repeat(xd, 4), np.repeat(xd1, 4)])
        yd = np.concatenate([np.repeat(yd, 2), np.repeat(yd1, 2)]).repeat(2)
        td = np.concatenate([td, td1]).repeat(4)

        indices = (t, p, y, x)
        values = xd * yd * td
        np.add.at(events_frames, indices, values)
        return events_frames
