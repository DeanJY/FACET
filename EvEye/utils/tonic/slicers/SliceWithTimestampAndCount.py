import numpy as np


def slice_events_by_timestamp_and_count(
    events: np.ndarray,
    timestamp: int,
    count: int,
    forward: bool = False,
):
    t = events["t"]
    timestamp_index = np.searchsorted(t, timestamp)
    if forward:
        start_index = timestamp_index
        end_index = timestamp_index + count
    else:
        start_index = timestamp_index - count
        end_index = timestamp_index

    start_index = max(0, start_index)
    end_index = min(len(events), end_index)

    return events[start_index:end_index]
