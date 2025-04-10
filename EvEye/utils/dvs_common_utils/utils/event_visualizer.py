import numpy as np

from dv_processing import EventStore
from dv_processing.visualization import EventVisualizer


class DvVisualizer(object):
    def __init__(self, resolution: tuple[int, int]) -> None:
        self._resolution = resolution
        self._visualizer = EventVisualizer(resolution)

    def visualize(self, event_chunk: EventStore) -> np.ndarray:
        return self._visualizer.generateImage(event_chunk)

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution


def main():
    visualizer = DvVisualizer((346, 260))
    event_chunk = EventStore()
    image = visualizer.visualize(event_chunk)
    print(image.shape)


if __name__ == "__main__":
    main()
