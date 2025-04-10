import numpy as np


class MotionCompensation2D:
    def __init__(
        self, 
        height: int, 
        width: int,
        reserve_time :bool = True
    ) -> None:
        self._height = height
        self._width = width
        self._reserve_time = reserve_time

    def __call__(
        self, 
        xypt: np.ndarray, 
        start_time_us: int, 
        vx: int|float = 0,
        vy: int|float = 0
    ) -> np.ndarray:
        """Compensate for 2D motion

        Args:
            xypt (np.ndarray): xypt events
            vx (int | float, optional): x velocity in pixel/s, positive to the right. Defaults to 0.
            vy (int | float, optional): y velocity in pixel/s, positive to the bottom. Defaults to 0.

        Returns:
            np.ndarray: Compensated xypt events
        """
        x, y, p, t = xypt['x'], xypt['y'], xypt['p'], xypt['t']
        output = xypt.copy()
        output['x'] = x - (t-start_time_us)*vx*1e-6
        output['y'] = y - (t-start_time_us)*vy*1e-6
        if not self._reserve_time:
            output['t'] = start_time_us
        return output[self._indices_in_fov(output)]
    
    def _indices_in_fov(self, xypt: np.ndarray) -> tuple[np.ndarray]:
        is_in_x = np.bitwise_and(xypt['x']>=0, xypt['x']<self._width)
        is_in_y = np.bitwise_and(xypt['y']>=0, xypt['y']<self._height)
        return np.bitwise_and(is_in_x, is_in_y)
