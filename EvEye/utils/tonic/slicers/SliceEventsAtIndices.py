import numpy as np


def slice_events_at_timepoints(
    events: np.ndarray, start_time: int, end_time: int
) -> np.ndarray:
    """
    Slice events at timepoints.
    Args:
        events: Structured events.
        start_tw: Start time stamp.
        end_tw: End time stamp.
    Returns:
        Sliced events.
    """
    start_index = np.searchsorted(events["t"], start_time, side="left")
    end_index = np.searchsorted(events["t"], end_time, side="right")
    return events[start_index:end_index]
