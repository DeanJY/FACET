import math
import numpy as np
from skimage.transform import AffineTransform, warp


def rand_range(amin, amax):
    """Return a random number in the range [amin, amax]"""
    return amin + (amax - amin) * np.random.rand()


class NumpyEventFrameRandomAffine:
    def __init__(
        self,
        size=(260, 346),
        translate=(0.2, 0.2),
        degrees=15,
        scale=(0.8, 1.2),
    ):
        self.height, self.width = size
        self.translate = translate
        self.degrees = degrees
        self.scale = scale
        self.affine_matrix = self.get_affine_matrix()

    def normalize(self, coords, backward=False):
        """
        Normalize or denormalize the coordinates.

        Args:
            coords (np.ndarray): The coordinates to be normalized or denormalized.
                coords[0] -> x
                coords[1] -> y
            backward (bool): If True, denormalize the coordinates. Otherwise, normalize them.

        Returns:
            np.ndarray: The transformed coordinates.
        """
        if not backward:
            coords[0] = coords[0] / (self.width - 1) - 0.5
            coords[1] = coords[1] / (self.height - 1) - 0.5
        else:
            coords[0] = (coords[0] + 0.5) * (self.width - 1)
            coords[1] = (coords[1] + 0.5) * (self.height - 1)

        return coords

    def get_translation_matrix(self):
        translate = [rand_range(-t, t) for t in self.translate]
        T = np.array(
            [[1, 0, translate[0]], [0, 1, translate[1]], [0, 0, 1]], dtype=float
        )
        return T

    def get_rotation_matrix(self):
        degrees = rand_range(-self.degrees, self.degrees) / 180 * math.pi
        cos, sin = np.cos(degrees), np.sin(degrees)
        R = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=float)
        return R

    def get_scale_matrix(self):
        scale = [rand_range(*self.scale) for _ in range(2)]
        S = np.array([[scale[0], 0, 0], [0, scale[1], 0], [0, 0, 1]], dtype=float)
        return S

    def get_affine_matrix(self):
        """
        Return the affine transformation matrix: A = TRS
        """
        T = self.get_translation_matrix()
        R = self.get_rotation_matrix()
        S = self.get_scale_matrix()
        A = np.dot(np.dot(T, R), S)
        return A

    def transform_event_frame(self, event_frame: np.array):
        """event_frame.shape -> (2, 50, 130, 173)"""
        assert event_frame.ndim == 3 or event_frame.ndim == 4
        if event_frame.ndim == 4:
            event_frame = np.moveaxis(event_frame, 0, 1)
        elif event_frame.ndim == 3:
            event_frame = np.expand_dims(event_frame, axis=0)
        t, c, h, w = event_frame.shape
        transform = AffineTransform(matrix=self.affine_matrix)
        for i in range(t):
            for j in range(c):
                event_frame[i, j] = warp(
                    event_frame[i, j],
                    transform,
                    preserve_range=True,
                    output_shape=(h, w),
                )
        if event_frame.shape[0] == 1:
            event_frame = np.squeeze(event_frame, axis=0)
        else:
            event_frame = np.moveaxis(event_frame, 1, 0)
        return event_frame

    def transform_label(self, label: np.array):
        """label.shape -> (2, 50)"""
        assert label.shape[0] == 2 and label.ndim == 2
        label_ones = np.ones((1, label.shape[1]))
        label_stacked = np.vstack((label, label_ones))
        label_affined = np.dot(self.affine_matrix, label_stacked)
        label = label_affined[:2].round()
        return label

    def temporal_flip(
        self, event_frame: np.array, label: np.array, closeness: np.array
    ):
        assert event_frame.ndim == 4 and event_frame.shape[0] == 2
        assert label.ndim == 2 and label.shape[0] == 2
        assert closeness.ndim == 1 and closeness.shape[0] == label.shape[1]

        # 使用切片操作进行翻转
        event_frame = event_frame[::-1, :, :, :]  # 翻转第0维
        event_frame = event_frame[:, ::-1, :, :]  # 翻转第1维
        label = label[:, ::-1]  # 翻转第1维
        closeness = closeness[::-1]  # 翻转第0维

        event_frame = np.ascontiguousarray(event_frame)
        label = np.ascontiguousarray(label)
        closeness = np.ascontiguousarray(closeness)

        return event_frame, label, closeness

    # def temporal_flip(
    #     self, event_frame: np.array, label: np.array, closeness: np.array
    # ):
    #     assert event_frame.ndim == 4 and event_frame.shape[0] == 2
    #     assert label.ndim == 2 and label.shape[0] == 2
    #     assert closeness.ndim == 1 and closeness.shape[0] == label.shape[1]

    #     event_frame = np.flip(event_frame, axis=0).copy()
    #     event_frame = np.flip(event_frame, axis=1).copy()
    #     label = np.flip(label, axis=1).copy()
    #     closeness = np.flip(closeness).copy()

    #     return event_frame, label, closeness
