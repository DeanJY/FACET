import numpy as np


class EventsIterator:
    def __init__(
        self,
        event_data: np.ndarray,  # xypt events structured array with fields 'x', 'y', 'p', 't'
        frame_gap_us: int,  # the time gap between two frames
        start_time: int,  # the start time of the first frame
    ):
        """
        Initializes the EventsIterator with event data and time parameters.

        Args:
            event_data (np.ndarray): Array of events, each with 'x', 'y', 'p', 't' fields.
            frame_gap_us (int): Time gap in microseconds between two consecutive frames.
            start_time (int): Start time in microseconds for the first frame.
        """
        self.event_data = event_data
        self.current_event_data = None
        self.frame_gap_us = frame_gap_us
        self.start_time = start_time
        self.current_time = start_time
        self.end_time = None
        self.index = 0
        self.dtype = self.event_data.dtype
        if not np.all(self.event_data[:-1]["t"] <= self.event_data[1:]["t"]):
            self.event_data = np.sort(self.event_data, order="t")

    def __iter__(self):
        return self

    def __next__(self):
        """
        Returns the next set of events within the current time slice.
        If no events are found, returns an empty array of the same structure as event_data.

        Returns:
            np.ndarray: A structured array of events happening in the current time slice.

        Raises:
            StopIteration: If all events have been processed and the iterator is exhausted.
        """
        if self.current_time >= self.event_data["t"].max():
            raise StopIteration

        self.end_time = self.current_time + self.frame_gap_us
        mask = (self.event_data["t"] >= self.current_time) & (
            self.event_data["t"] < self.end_time
        )
        self.current_event_data = self.event_data[mask]
        self.current_time = self.end_time

        return self.current_event_data

    def __len__(self):
        total_frames = (
            self.event_data["t"].max() - self.start_time + self.frame_gap_us - 1
        ) // self.frame_gap_us
        return int(total_frames)

    @property
    def total_time(self):
        return self.event_data["t"].max() - self.start_time


def main():
    from EvEye.utils.processor.TxtProcessor import TxtProcessor

    txt_processor = TxtProcessor(
        "/mnt/data2T/junyuan/eye-tracking/EV_Eye_dataset/raw_data/Data_davis/user1/left/session_1_0_1/events/events.txt"
    )
    event_data = txt_processor.load_events_from_txt()
    start_time = event_data["t"].min()
    event_iterator = EventsIterator(
        event_data, frame_gap_us=3000, start_time=start_time
    )
    print("Total number of frames:", len(event_iterator))
    for event in event_iterator:
        print(
            "Events from time",
            event_iterator.current_time - event_iterator.frame_gap_us,
            "to",
            event_iterator.current_time,
        )
        print(event)


if __name__ == "__main__":
    main()
