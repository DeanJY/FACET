import numpy as np


class TimeSurfaceBuilder:
    def __init__(
        self,
        height: int,
        width: int,
        frame_interval_us: int,
        single_channel: bool = False,
    ) -> None:
        self._height = height
        self._width = width
        self._frame_interval_us = frame_interval_us
        self._single_channel = single_channel

    def __call__(
        self,
        xypt: np.ndarray,
        start_time: int,
    ) -> np.ndarray:
        """
        Generate the time surface

        Args:
            xypt (np.ndarray): xypt events
            start_time_us (int): the start time of the time surface
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
            return self._calc_single_channel_time_surface(x, y, p, t)
        else:
            return self._calc_two_channel_time_surface(x, y, p, t)

    def _calc_single_channel_time_surface(self, x, y, p, t) -> np.ndarray:
        time_surface = np.zeros((self._height, self._width), dtype=np.float32)
        np.add.at(time_surface, (y, x), p * 2 - 1)
        time_surface = np.clip(time_surface, -1, 1)
        return time_surface

    def _calc_two_channel_time_surface(self, x, y, p, t) -> np.ndarray:
        time_surface = np.zeros((2, self._height, self._width), dtype=np.float32)
        np.add.at(time_surface[0], (y[p == 0], x[p == 0]), t[p == 0])
        np.add.at(time_surface[1], (y[p == 1], x[p == 1]), t[p == 1])
        time_surface = time_surface / self._frame_interval_us
        return time_surface

    @staticmethod
    def visualize(time_surface: np.ndarray) -> np.ndarray:
        """ 
        visualize time surfaces

        Args:
            time_surface (np.ndarray): time surface of shape (2, h, w), normalized to (0, 1)
        Returns:
            np.ndarray: visualized time surface. shape=(h, w, 3), dtype=np.uint8. blue for off events and red for on events. \
                The intensity of the color represents the normalized t.
        """
        _, h, w = time_surface.shape
        canvas = np.full((h, w, 3), fill_value=0, dtype=np.uint8)
        off_time_surface, on_time_surface = time_surface
        # process on events
        canvas[on_time_surface > 0, 2] = 255
        canvas[..., 2] = (canvas[..., 2] * on_time_surface).astype(
            np.uint8
        )  #  这一步是为了让红色的强度和时间成正比，其原理是将红色通道的值乘以时间
        # process off events
        canvas[off_time_surface > 0, 0] = 255
        canvas[..., 0] = (canvas[..., 0] * off_time_surface).astype(np.uint8)
        return canvas


def main():
    from EvEye.utils.processor.TxtProcessor import TxtProcessor
    from EvEye.utils.visualization.visualization import save_image

    txt_path = "/mnt/data2T/junyuan/eye-tracking/EV_Eye_dataset/raw_data/Data_davis/user1/left/session_1_0_1/events/events.txt"
    txt_processor = TxtProcessor(txt_path)
    events = txt_processor.load_events_from_txt()
    start_time = 1657710786154759
    time_surface_builder = TimeSurfaceBuilder(260, 346, 1000)
    time_surface = time_surface_builder(events, start_time)
    canvas = TimeSurfaceBuilder.visualize(time_surface)
    save_image(canvas, "time_surface.png")


if __name__ == "__main__":
    main()
