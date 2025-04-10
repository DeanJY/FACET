from typing import Iterator, Optional, Tuple, Iterator
from pathlib import Path

import os
import os.path as osp

import numpy as np
import cv2

from dv_processing.io import MonoCameraRecording
from dv_processing import Accumulator
from dv_processing import EventStore
from dv_processing import Frame

from dvs_common_utils.utils.dv_accumulator import DvAccumulator

EVENTS_KEY = "events"
FRAMES_KEY = "frames"
TIMESTAMP_KEY = "timestamp"
X_KEY = "x"
Y_KEY = "y"
POLARITY_KEY = "polarity"


class EventIterator(object):
    def __init__(
        self, 
        aedat_reader: MonoCameraRecording, 
        interval_us: int = 10000,   # event length
        frame_interval_us: int | None = None,  # time between starts of two frames
        start_offset_us: int = 0, 
    ) -> None:
        super().__init__()
        self._aedat_reader = aedat_reader
        self._interval_us = interval_us
        self._frame_interval_us = frame_interval_us \
            if frame_interval_us is not None else interval_us
        self._start_time_us, self._end_time_us = self._aedat_reader.getTimeRange()
        
        self._iter_start_time_us = self._start_time_us + start_offset_us
        self._iter_end_time_us = self._start_time_us + self._interval_us
        if self._iter_start_time_us is not None:
            assert self._start_time_us <= self._iter_start_time_us <=self._end_time_us
            
    def __iter__(self) -> Iterator:
        return self
        
    def __next__(self) -> EventStore:
        if self._iter_start_time_us > self._end_time_us:
            raise StopIteration
        ret = self._aedat_reader.getEventsTimeRange(
            self._iter_start_time_us,
            self._iter_end_time_us
        )
        self._iter_start_time_us += self._frame_interval_us
        self._iter_end_time_us = self._iter_start_time_us + self._interval_us
        
        return ret
    
    @property
    def start_timestamp(self) -> int:
        return self._iter_start_time_us - self._frame_interval_us
    
    @property
    def end_timestamp(self) -> int:
        return self._iter_end_time_us - self._frame_interval_us


class FrameIterator(Iterator):
    def __init__(
        self, 
        aedat_reader: MonoCameraRecording
    ) -> None:
        super().__init__()
        self._aedat_reader = aedat_reader
        
    def __next__(self) -> Frame:
        return self._aedat_reader.getNextFrame()


class RgbEventIterator(Iterator):
    def __init__(
        self, 
        aedat_reader: MonoCameraRecording,
        event_interval_us: int = 10000    
    ) -> None:
        super().__init__()
        self._aedat_reader = aedat_reader
        self._event_interval_us = event_interval_us
        self._aedat_reader.getNextFrame()   # skip the first RGB frame. Events start after the first RGB frame.
        
    def __next__(self) -> tuple[Frame, EventStore]:
        rgb_image = self._aedat_reader.getNextFrame()
        if rgb_image is None:
            raise StopIteration
        self._rgb_timestamp: int = rgb_image.timestamp
        event_start_time: int = self._rgb_timestamp - self._event_interval_us
        event_chunk = self._aedat_reader.getEventsTimeRange(
            event_start_time, self._rgb_timestamp
        )
        return rgb_image, event_chunk
    
    @property
    def start_timestamp(self) -> int:
        return self._rgb_timestamp - self._event_interval_us
    
    @property
    def end_timestamp(self) -> int:
        return self._rgb_timestamp
    
class RgbEventAsyncIterator(Iterator):
    f"""Iterator for asynchronous RGB frames and event chunks
    
    The iterator works as a event iterator without any gap in between \n
    It also returns the last rgb frame (last means rgb frame start time before event chunk end)
    """
    def __init__(
        self, 
        aedat_reader: MonoCameraRecording,
        event_interval_us: int = 10000, 
        frame_interval_us: int | None = None
    ) -> None:
        """Iterator for asynchronous RGB frames and event chunks

        Args:
            aedat_reader (MonoCameraRecording): Aedat Reader 
            event_interval_us (int, optional): Event chunk length in microsecond. Defaults to 10000.
            
        The iterator works as a event iterator without any gap in between \n
        It also returns the last rgb frame (last means rgb frame start time before event chunk end)
        """
        super().__init__()
        self._aedat_reader = aedat_reader
        self._event_interval_us = event_interval_us
        self._frame_interval_us = frame_interval_us \
            if frame_interval_us is not None else event_interval_us
        
    def __iter__(self):
        self._start_time_us, self._end_time_us = self._aedat_reader.getTimeRange()
        self._iter_start_time_us = self._start_time_us
        self._iter_end_time_us = self._start_time_us + self._event_interval_us
        self._last_frame = self._aedat_reader.getNextFrame()
        self._next_frame = self._aedat_reader.getNextFrame()
        return self
        
    def __next__(self) -> tuple[Frame, EventStore]:
        
        if self._iter_start_time_us > self._end_time_us or self._next_frame is None:
            raise StopIteration
        
        event_chunk = self._aedat_reader.getEventsTimeRange(
            self._iter_start_time_us,
            self._iter_end_time_us
        )
        if self._last_frame.timestamp < self._iter_end_time_us:
            # update frame
            self._last_frame = self._next_frame
            self._next_frame = self._aedat_reader.getNextFrame()
        
        self._iter_start_time_us += self._frame_interval_us
        self._iter_end_time_us += self._frame_interval_us
        
        return self._last_frame, event_chunk
    
    @property
    def start_timestamp(self) -> int:
        return self._iter_start_time_us - self._frame_interval_us
    
    @property
    def end_timestamp(self) -> int:
        return self._iter_end_time_us - self._frame_interval_us
    
class Aedat4Processor(object):
    def __init__(
        self, 
        path: str,
    ) -> None:
        self._path = path
        assert osp.exists(self._path), "File {} does not exist".format(self._path)
        self._aedat_reader: MonoCameraRecording = MonoCameraRecording(path)
        self._start_timestamp, self._end_timestamp = self._aedat_reader.getTimeRange()
        
    @property
    def frame_resolution(self) -> Tuple[int, int]:
        return self._aedat_reader.getFrameResolution()
        
    @property
    def event_resolution(self) -> Tuple[int, int]:
        return self._aedat_reader.getEventResolution()
    
    @property
    def start_timestamp(self) -> int:
        return self._start_timestamp
    
    @property
    def end_timestamp(self) -> int:
        return self._end_timestamp
    
    @property
    def has_rgb_frames(self) -> bool:
        return self._aedat_reader.isFrameStreamAvailable()
    
    def get_events_iterator(
        self, 
        interval_us: int, 
        frame_interval_us: int | None = None, 
        start_offset_us: int = 0
    ) -> EventIterator:
    
        return EventIterator(
            self._aedat_reader, interval_us, frame_interval_us, start_offset_us
        )
    
    def get_frame_iterator(self) -> FrameIterator:
        return FrameIterator(self._aedat_reader)
    
    def get_sync_rgb_events_iterator(
        self, 
        event_interval_us: int = 10000
    ):
        return RgbEventIterator(self._aedat_reader, event_interval_us)
    
    def get_async_rgb_event_iterator(
        self, 
        event_interval_us: int = 5000, 
        frame_interval_us: int | None = None
    ) -> RgbEventAsyncIterator:
        """Iterator for asynchronous RGB frames and event chunks

        Args:
            aedat_reader (MonoCameraRecording): Aedat Reader 
            event_interval_us (int, optional): Event chunk length in microsecond. Defaults to 10000.
            
        The iterator works as a event iterator without any gap in between \n
        It also returns the last rgb frame (last means rgb frame start time before event chunk end)
        """
        return RgbEventAsyncIterator(
            self._aedat_reader,
            event_interval_us,
            frame_interval_us
        )
    
    def get_events_by_time_range(
        self, start_time_us: int, end_time_us: int
    ) -> EventStore:
        return self._aedat_reader.getEventsTimeRange(
            start_time_us, end_time_us
        )

def aedat_to_framestacks(
    aedat_path: str | Path,
    output_dir: str | Path,
    event_frame_len_us: int = 10000,
    frame_interval_us: int | None = None,
    frame_stack_channel_num: int = 5,
    start_offset_us: int = 0
):
    """Turn Aedat4 file into frame stacks

    Args:
        aedat_path (str | Path): The path of aedat4 file to be converted
        output_dir (str | Path): Directory to store the output frame_stacks
        event_sequence_len_us (int, optional): The temporal length of a single frame. Defaults to 10000.
        frame_interval_us (int | None, optional): Interval between two frames, from start to start. \
            If set to None, the start of current frame stack will be equal to the end of the last one. Defaults to None.
        frame_stack_channel_num (int, optional): Channel number of frame stacks. Defaults to 5.
        start_offset_us (int, optional): Offset of the starting time. Defaults to 0.
        
    Return:
        starting index: From which index is the frame stack generated. (if starting time before the sequence start, the frame stack will be ignored)
    """
    aedat_path = Path(aedat_path)
    output_dir = Path(output_dir)
    assert aedat_path.exists(), f"Aedat4 file {aedat_path} does not exist."
    assert output_dir.exists(), f"output directory {output_dir} does not exist."
        
    aedat4_processor = Aedat4Processor(str(aedat_path))
    event_iterator = aedat4_processor.get_events_iterator(
        interval_us=event_frame_len_us * frame_stack_channel_num,
        frame_interval_us=frame_interval_us,
        start_offset_us=start_offset_us
    )
    
    dv_accumulator = DvAccumulator(resolution=aedat4_processor.event_resolution)

    for index, event_chunk in enumerate(event_iterator):
        frame_stack = list()
        frame_start_timestamp = event_iterator.start_timestamp
        frame_end_timestamp = frame_start_timestamp + event_frame_len_us
        for i in range(frame_stack_channel_num):
            frame_stack.insert(
                0, 
                dv_accumulator.generate_np_frame_e2e(event_chunk.sliceTime(
                    frame_start_timestamp, frame_end_timestamp
                ))
            )
            frame_start_timestamp = frame_start_timestamp + event_frame_len_us
            frame_end_timestamp = frame_start_timestamp + event_frame_len_us
        frame_stack = np.stack(frame_stack, axis=-1)
        
        frame_stack_path = output_dir / '{:08d}'.format(index)
        # cv2.imwrite(str(frame_stack_path), frame_stack)
        frame_stack_img = frame_stack[..., -3:]
        np.save(frame_stack_path, frame_stack)

    
def aedat_to_rgb_framestacks(
    aedat_path: str, 
    save_dir: str, 
    frame_stack_channel_num: int = 3, 
    event_frame_len_us: int = 1000
):
    """Turn DAVIS data (Aedat4) to RGB and event pairs

    Args:
        aedat_path (str): The path of aedat4 file to be converted
        save_dir (str): Directory to store the output frame_stacks
        frame_stack_channel_num (int, optional): Channel number of frame stacks. Defaults to 3.
        event_frame_interval_us (int, optional): The temporal length of a single frame. Defaults to 1000.
    """
    aedat4_processor = Aedat4Processor(aedat_path)
    assert osp.exists(save_dir)
    rgb_event_iterator = aedat4_processor.get_sync_rgb_events_iterator(
        event_interval_us=frame_stack_channel_num * event_frame_len_us
    )
    dv_accumulator = DvAccumulator(resolution=aedat4_processor.event_resolution)
    for idx, (rgb, event_chunk) in enumerate(rgb_event_iterator):
        event_end_timestamp = rgb.timestamp
        frame_stack = list()
        for i in range(frame_stack_channel_num):
            event_start_timestamp = event_end_timestamp - event_frame_len_us
            frame_stack.insert(
                0, 
                dv_accumulator.generate_np_frame_e2e(event_chunk.sliceTime(
                    event_start_timestamp, event_end_timestamp
                ))
            )
            event_end_timestamp = event_end_timestamp - event_frame_len_us
            
        frame_stack = np.stack(frame_stack, axis=-1)
        np.save(osp.join(save_dir, '{:08d}'.format(idx)), frame_stack)
        cv2.imwrite(osp.join(save_dir, '{:08d}.jpg'.format(idx)), rgb.image)
        
        
def aedat_to_rgb_framestacks_synced_with_video(
    video_path: str | Path,
    aedat_path: str | Path, 
    output_dir: str | Path, 
    frame_stack_channel_num: int = 5, 
    event_frame_len_us: int = 10000
) -> int:
    """generate frame_stacks according to a aedat4 file and the corresponding video snippet

    Args:
        video_path (str | Path): The path of the reference video
        aedat_path (str | Path): The path of aedat4 file to be converted
        output_dir (str | Path): Directory to store the output frame_stacks
        frame_stack_channel_num (int, optional): Channel number of frame stacks. Defaults to 3.
        event_frame_interval_us (int, optional): The temporal length of a single frame. Defaults to 1000.

    Returns:
        int: Number of output files
    """
    video_path = Path(video_path)
    aedat_path = Path(aedat_path)
    output_dir = Path(output_dir)
    assert video_path.exists(), f"video {video_path} does not exist."
    assert aedat_path.exists(), f"Aedat4 file {aedat_path} does not exist."
    assert output_dir.exists(), f"output directory {output_dir} does not exist."
    
    aedat4_processor = Aedat4Processor(str(aedat_path))
    video = cv2.VideoCapture(str(video_path))
    dv_accumulator = DvAccumulator(resolution=aedat4_processor.event_resolution)
    
    aedat4_start_timestamp = aedat4_processor.start_timestamp
    aedat4_end_timestamp = aedat4_processor.end_timestamp
    
    event_chunk_end_timestamp_us = aedat4_start_timestamp
    
    frame_index = 0
    fps = video.get(cv2.CAP_PROP_FPS)
    interval_us = (1.0 / fps) * 1000000
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        event_chunk_end_timestamp_us = event_chunk_end_timestamp_us + int(interval_us)
        event_chunk_start_timestamp_us = event_chunk_end_timestamp_us - \
            event_frame_len_us * frame_stack_channel_num
            
        if event_chunk_start_timestamp_us < aedat4_start_timestamp:
            # frame_index += 1
            continue
        
        if event_chunk_start_timestamp_us > aedat4_end_timestamp:
            break
            
        event_chunk = aedat4_processor.get_events_by_time_range(
            event_chunk_start_timestamp_us, 
            event_chunk_end_timestamp_us
        )
        
        frame_stack_buffer: list[np.ndarray] = list()
        frame_start_timestamp = event_chunk_start_timestamp_us
        frame_end_timestamp = frame_start_timestamp + event_frame_len_us
        for i in range(frame_stack_channel_num):
            frame_stack_buffer.insert(
                0, 
                dv_accumulator.generate_np_frame_e2e(event_chunk.sliceTime(
                    frame_start_timestamp, frame_end_timestamp
                ))
            )
            frame_start_timestamp = frame_start_timestamp + event_frame_len_us
            frame_end_timestamp = frame_start_timestamp + event_frame_len_us
        frame_stack = np.stack(frame_stack_buffer, axis=-1)
        
        rgb_img_path = output_dir / '{:08d}.jpg'.format(frame_index)
        frame_stack_path = output_dir / '{:08d}'.format(frame_index)
        
        np.save(frame_stack_path, frame_stack)
        cv2.imwrite(str(rgb_img_path), frame)
        frame_index += 1
        
    return frame_index
    

if __name__ == "__main__":
    import cv2
    aedat_file_path = "data/DAVIS346/high_speed_motion/dvSave-2023_06_30_14_09_22.aedat4"
    processor = Aedat4Processor(aedat_file_path)
    
    for rgb, event_chunk in RgbEventAsyncIterator(processor._aedat_reader):
        print(rgb)
    


