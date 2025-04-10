from typing import Any
import numpy as np


class FrameStackBuilder:
    def __init__(
        self,
        height: int,
        width: int,
        num_frames: int,
        normalize: bool = True,
    ) -> None:
        self._height = height
        self._width = width
        self._num_frames = num_frames
        self._normalize = normalize

    def __call__(
        self,
        xypt: np.ndarray,
        start_time: int,
        spatial_downsample: tuple = (1, 1),  # height_downsample, width_downsample
        temporal_downsample: int = 1,
        mode: str = "bilinear",  # 'bilinear', 'nearest', 'causal_linear'
        max_count: int = 10,
    ):
        """
        Generate the frame stack

        Args:
            xypt (np.ndarray): xypt events
            start_time (int): the start time of the frame stack
            mode (str): the mode of the frame stack, 'bilinear', 'nearest', 'causal_linear'
            spatial_downsample (tuple): the downsample of the spatial, (height_downsample, width_downsample)
            temporal_downsample (int): the downsample of the temporal
            max_count (int): the max count of the frame stack
        Returns:
            np.ndarray: shape=(num_frames, 2, h, w), channel 0 for off events. channel 1 for on events
        """
        assert mode in ["bilinear", "nearest", "causal_linear"]
        if xypt.size == 0:
            return np.zeros(
                [self._num_frames, 2, self._height, self._width], dtype=np.float32
            )
        frame_stack = np.zeros(
            [self._num_frames, 2, self._height, self._width], dtype=np.float32
        )

        x, y, p, t = xypt["x"], xypt["y"], xypt["p"], xypt["t"] - start_time

        if mode == "nearest":
            x = (x / spatial_downsample[1]).round().astype(int).clip(0, self._width - 1)
            y = (
                (y / spatial_downsample[0])
                .round()
                .astype(int)
                .clip(0, self._height - 1)
            )
            p = p.round().astype(int)
            t = (
                (t / temporal_downsample - 0.5)
                .round()
                .astype(int)
                .clip(0, self._num_frames - 1)
            )
            np.add.at(frame_stack, (t, p, y, x), 1)
            if self._normalize:
                frame_stack = frame_stack / max_count
            return frame_stack

        x_now, x_next, weight_x_now, weight_x_next = self.bilinear_interpolation(
            x, spatial_downsample[1], self._width - 1
        )
        y_now, y_next, weight_y_now, weight_y_next = self.bilinear_interpolation(
            y, spatial_downsample[0], self._height - 1
        )
        temporal_downsample = temporal_downsample / self._num_frames
        t_now, t_next, weight_t_now, weight_t_next = self.bilinear_interpolation(
            t, temporal_downsample, self._num_frames - 1
        )

        if mode == "bilinear":
            np.add.at(
                frame_stack,
                (t_now, p, y_now, x_now),
                weight_x_now * weight_y_now * weight_t_now,
            )
            np.add.at(
                frame_stack,
                (t_now, p, y_now, x_next),
                weight_x_next * weight_y_now * weight_t_now,
            )
            np.add.at(
                frame_stack,
                (t_now, p, y_next, x_now),
                weight_x_now * weight_y_next * weight_t_now,
            )
            np.add.at(
                frame_stack,
                (t_now, p, y_next, x_next),
                weight_x_next * weight_y_next * weight_t_now,
            )
            np.add.at(
                frame_stack,
                (t_next, p, y_now, x_now),
                weight_x_now * weight_y_now * weight_t_next,
            )
            np.add.at(
                frame_stack,
                (t_next, p, y_now, x_next),
                weight_x_next * weight_y_now * weight_t_next,
            )
            np.add.at(
                frame_stack,
                (t_next, p, y_next, x_now),
                weight_x_now * weight_y_next * weight_t_next,
            )
            np.add.at(
                frame_stack,
                (t_next, p, y_next, x_next),
                weight_x_next * weight_y_next * weight_t_next,
            )
            return frame_stack

        if mode == "causal_linear":
            np.add.at(
                frame_stack,
                (t_now, p, y_now, x_now),
                weight_x_now * weight_y_now * weight_t_now,
            )
            np.add.at(
                frame_stack,
                (t_now, p, y_now, x_next),
                weight_x_next * weight_y_now * weight_t_now,
            )
            np.add.at(
                frame_stack,
                (t_now, p, y_next, x_now),
                weight_x_now * weight_y_next * weight_t_now,
            )
            np.add.at(
                frame_stack,
                (t_now, p, y_next, x_next),
                weight_x_next * weight_y_next * weight_t_now,
            )
            return frame_stack

    @staticmethod
    def bilinear_interpolation(input: np.ndarray, scale: int, input_max: int):
        """
        Calculate the bilinear interpolation of the input

        Args:
            input (np.ndarray): input
            scale (int): scale factor
            input_max (int): the max value of the input
        Returns:
            input (np.ndarray): the input after bilinear interpolation
            input_next (np.ndarray): the next input after bilinear interpolation
            weight_input (np.ndarray): the weight of the input
            weight_input_next (np.ndarray): the weight of the next input
        Example:
            input = 108
            scale = 5
            input_max = 346
            input, input_next, weight_input, weight_input_next = bilinear_interpolation(input, scale, input_max)
            input, input_next, weight_input, weight_input_next
            21 ,22, 0.4, 0.6
        """
        if scale == 1:
            return (
                input.astype(int),
                input.astype(int),
                np.ones_like(input).astype(int),
                np.zeros_like(input).astype(int),
            )
        weight_input_next = input % scale / scale
        weight_input = 1 - weight_input_next
        input = (input / scale).round().astype(int).clip(0, input_max)
        input_next = (input + 1).round().astype(int).clip(0, input_max)
        weight_input = weight_input.round().astype(int)
        weight_input_next = weight_input_next.round().astype(int)
        return (
            input,
            input_next,
            weight_input,
            weight_input_next,
        )

    @staticmethod
    def visualize(frame_stack: np.ndarray, threshold: float = 0):
        """Visualize the frame stack in cv2 BGR format"""
        assert frame_stack.ndim == 4 and frame_stack.shape[1] == 2
        n, _, h, w = frame_stack.shape
        canvases = np.zeros((n, h, w, 3), dtype=np.uint8)
        for frame_index in range(n):
            off_frame_stack = frame_stack[frame_index, 0]
            on_frame_stack = frame_stack[frame_index, 1]
            canvases[frame_index][off_frame_stack > threshold, 0] = 255
            canvases[frame_index][on_frame_stack > threshold, 2] = 255
        return canvases


def main():
    from EvEye.utils.processor.TxtProcessor import TxtProcessor
    from EvEye.utils.visualization.visualization import save_image

    txt_path = "/mnt/data2T/junyuan/eye-tracking/EV_Eye_dataset/raw_data/Data_davis/user1/left/session_1_0_1/events/events.txt"
    txt_processor = TxtProcessor(txt_path)
    # events = txt_processor.load_events_from_txt()
    # start_time = events["t"][0]
    start_time = 1657710786154759
    frame_stack_builder = FrameStackBuilder(
        height=260, width=346, num_channels=10, interval_perframe=1000
    )
    # frame_stack = frame_stack_builder(events_data, start_time)
    # frame_stack_vis = FrameStackBuilder.visualize(frame_stack)


if __name__ == "__main__":
    main()
