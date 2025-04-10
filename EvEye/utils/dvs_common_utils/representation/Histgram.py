from typing import Any
import numpy as np


class HistgramBuilder:
    def __init__(
        self,
        height: int,
        width: int,
        frame_interval_us: int,
        max_count: int,
        normalize: bool = True,
        single_channel: bool = False,
    ) -> None:
        """Histgram counts events on every pixel, thus forming a histgram.

        Args:
            height (int)
            width (int)
            max_count (int): max counting number on each pixel per polarity
        """
        self._height = height
        self._width = width
        self._frame_interval_us = frame_interval_us
        self._max_count = max_count
        self._normalize = normalize
        self._single_channel = single_channel

    def __call__(
        self,
        xypt: np.ndarray,
        start_time: int,
    ) -> np.ndarray:
        """Generate the histgram

        Args:
            xypt (np.ndarray): xypt events
            start_time_us (int): the start time of the histgram
        Returns:
            np.ndarray: shape=(2, h, w), channel 0 for off events. channel 1 for on events
        """
        if xypt.size == 0:
            return (
                np.zeros((1, self._height, self._width), dtype=np.uint32)
                if self._single_channel
                else np.zeros((2, self._height, self._width), dtype=np.uint32)
            )

        t = xypt["t"] - start_time
        mask = t < self._frame_interval_us
        x, y, p = xypt["x"][mask], xypt["y"][mask], xypt["p"][mask]

        if self._single_channel:
            return self._calc_single_channel_histgram(x, y, p)
        else:
            return self._calc_two_channel_histgram(x, y, p)

    def _calc_single_channel_histgram(self, x, y, p) -> np.ndarray:
        histgram = np.zeros((self._height, self._width), dtype=np.int32)
        p = p.astype(np.int32) * 2 - 1
        np.add.at(histgram, (y, x), p)
        histgram = np.clip(histgram, -self._max_count, self._max_count)
        if self._normalize:
            histgram = histgram / self._max_count
        return histgram

    def _calc_two_channel_histgram(self, x, y, p) -> np.ndarray:
        histgram = np.zeros((2, self._height, self._width), dtype=np.uint32)
        np.add.at(histgram[0], (y[p == 0], x[p == 0]), 1)
        np.add.at(histgram[1], (y[p == 1], x[p == 1]), 1)
        histgram = np.clip(histgram, 0, self._max_count)

        # histgram_2 = np.zeros((2, self._height, self._width), dtype=np.uint32)
        # for px, py, pp in zip(x, y, p):
        #     if histgram_2[pp, py, px] < self._max_count:
        #         histgram_2[pp, py, px] += 1
        if self._normalize:
            histgram = histgram / self._max_count
        return histgram

    @property
    def max_count(self) -> int:
        return self._max_count

    @staticmethod
    def visualize(histgram: np.ndarray):
        assert histgram.shape[0] == 2
        _, h, w = histgram.shape
        canvas = np.full((h, w, 3), fill_value=0, dtype=np.uint8)
        off_hist, on_hist = histgram
        # process on events
        canvas[on_hist > 0.3, 2] = 255
        # canvas[..., 2] = (canvas[..., 2] * on_hist).astype(np.uint8)
        # process off events
        canvas[off_hist > 0.3, 0] = 255
        # canvas[..., 0] = (canvas[..., 0] * off_hist).astype(np.uint8)
        return canvas

    @staticmethod
    def visualize_gray(histgram: np.ndarray):
        assert histgram.shape[0] == 2
        _, h, w = histgram.shape
        canvas = np.full((h, w), fill_value=0, dtype=np.uint8)
        off_hist, on_hist = histgram
        canvas[on_hist > 0.3] = 255
        canvas[off_hist > 0.3] = 255
        return canvas


def main():
    from EvEye.utils.processor.TxtProcessor import TxtProcessor
    from EvEye.utils.visualization.visualization import save_image

    txt_path = "/mnt/data2T/junyuan/eye-tracking/EV_Eye_dataset/raw_data/Data_davis/user1/left/session_1_0_1/events/events.txt"
    txt_processor = TxtProcessor(txt_path)
    events = txt_processor.load_events_from_txt()
    histgram_builder = HistgramBuilder(height=260, width=346, max_count=10)
    histgram = histgram_builder(events, start_time_us=events["t"][0])
    canvas = HistgramBuilder.visualize(histgram)
    save_image(canvas, "histgram.png")


if __name__ == "__main__":
    main()
