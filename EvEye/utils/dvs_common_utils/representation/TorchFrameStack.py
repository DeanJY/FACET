from typing import Any
import torch


def TorchFrameStack(
    events,
    size,
    num_frames,
    spatial_downsample,
    temporal_downsample,
    mode="bilinear",
    max_count=10,
    normalize=True,
):
    """
    Perform bilinear interpolation directly on the events,
    while converting them to frames and do spatial and temporal downsamplings
    all at the same time.
    """
    height, width = size
    p, x, y, t = events
    events_frames = torch.zeros(
        [num_frames, 2, height, width], device="cuda:0"
    ).type_as(events)

    def bilinear_interp(x, scale, x_max):
        if scale == 1:
            return x, x, torch.ones_like(x), torch.zeros_like(x)
        xd1 = x % scale / scale
        xd = 1 - xd1
        x = (x / scale).long().clamp(0, x_max)
        x1 = (x + 1).clamp(0, x_max)
        return x, x1, xd, xd1

    if mode == "nearest":
        p = p.round().long()
        x = (x / spatial_downsample[0]).round().long().clamp(0, width - 1)
        y = (y / spatial_downsample[1]).round().long().clamp(0, height - 1)
        t = (t / temporal_downsample - 0.5).round().long().clamp(0, num_frames - 1)
        events_frames.index_put_(
            (t, p, y, x), torch.ones_like(p, dtype=torch.float32), accumulate=True
        )
        return events_frames

    x, x1, xd, xd1 = bilinear_interp(x, spatial_downsample[0], width - 1)
    y, y1, yd, yd1 = bilinear_interp(y, spatial_downsample[1], height - 1)
    t, t1, td, td1 = bilinear_interp(t, temporal_downsample, num_frames - 1)

    # similar to bilinear, but temporally causal
    if mode == "causal_linear":
        p = p.long().repeat(4)

        x = torch.cat([x.repeat(2), x1.repeat(2)])
        y = torch.cat([y, y1]).repeat(2)
        t = t.repeat(4)

        xd = torch.cat([xd.repeat(2), xd1.repeat(2)])
        yd = torch.cat([yd, yd1]).repeat(2)
        td = td1.repeat(4)  # causal

        indices = (t.int(), p.int(), y.int(), x.int())
        values = xd * yd * td
        events_frames.index_put_(indices, values, accumulate=True)

        return events_frames

    if mode == "bilinear":
        p = p.long().repeat(8)

        x = torch.cat([x.repeat(4), x1.repeat(4)])
        y = torch.cat([y.repeat(2), y1.repeat(2)]).repeat(2)
        t = torch.cat([t, t1]).repeat(4)

        xd = torch.cat([xd.repeat(4), xd1.repeat(4)])
        yd = torch.cat([yd.repeat(2), yd1.repeat(2)]).repeat(2)
        td = torch.cat([td, td1]).repeat(4)

        indices = (t.int(), p.int(), y.int(), x.int())
        values = xd * yd * td
        events_frames.index_put_(indices, values, accumulate=True)
        return events_frames
