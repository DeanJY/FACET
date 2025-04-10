import math
import torch
import torch.nn.functional as F


def rand_range(amin, amax):
    """Return a random number in the range [amin, amax]"""
    return amin + (amax - amin) * torch.rand(1).item()


class TorchEventFrameRandomAffine:
    def __init__(
        self,
        size=(260, 346),
        translate=(0.2, 0.2),
        degrees=15,
        scale=(0.8, 1.2),
        spatial_jitter=None,
    ):
        self.height, self.width = size
        self.translate = translate
        self.degrees = degrees
        self.scale = scale
        self.spatial_jitter = spatial_jitter
        self.affine_matrix = self.get_affine_matrix()

    def normalize(self, coords, backward=False):
        """
        Normalize or denormalize the coordinates.

        Args:
            coords (torch.Tensor): The coordinates to be normalized or denormalized.
                coords[0] -> x
                coords[1] -> y
            backward (bool): If True, denormalize the coordinates. Otherwise, normalize them.

        Returns:
            torch.Tensor: The transformed coordinates.
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
        T = torch.tensor(
            [[1, 0, translate[0]], [0, 1, translate[1]], [0, 0, 1]], dtype=torch.float32
        )
        return T

    def get_rotation_matrix(self):
        degrees = rand_range(-self.degrees, self.degrees) / 180 * math.pi
        cos, sin = torch.cos(torch.tensor(degrees)), torch.sin(torch.tensor(degrees))
        R = torch.tensor(
            [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=torch.float32
        )
        return R

    def get_scale_matrix(self):
        scale = [rand_range(*self.scale) for _ in range(2)]
        S = torch.tensor(
            [[scale[0], 0, 0], [0, scale[1], 0], [0, 0, 1]], dtype=torch.float32
        )
        return S

    def get_affine_matrix(self):
        """
        Return the affine transformation matrix: A = TRS
        """
        T = self.get_translation_matrix()
        R = self.get_rotation_matrix()
        S = self.get_scale_matrix()
        A = T @ R @ S
        return A

    def transform_event_frame(self, event_frame: torch.Tensor):
        """event_frame.shape -> (2, 50, 130, 176)"""
        assert event_frame.ndim == 3 or event_frame.ndim == 4
        if event_frame.ndim == 4:
            event_frame = event_frame.permute(1, 0, 2, 3)
        elif event_frame.ndim == 3:
            event_frame = event_frame.unsqueeze(0)

        t, c, h, w = event_frame.shape

        # 创建仿射变换矩阵
        affine_matrix = self.affine_matrix.to(event_frame.device)
        affine_matrix = affine_matrix.unsqueeze(0).repeat(t * c, 1, 1)

        # 创建网格
        grid = F.affine_grid(affine_matrix, [t * c, 1, h, w], align_corners=False)

        # 应用变换
        event_frame = event_frame.view(t * c, 1, h, w)
        event_frame = F.grid_sample(event_frame, grid, align_corners=False)
        event_frame = event_frame.view(t, c, h, w)

        if event_frame.shape[0] == 1:
            event_frame = event_frame.squeeze(0)
        else:
            event_frame = event_frame.permute(1, 0, 2, 3)

        return event_frame

    def transform_label(self, label: torch.Tensor):
        """label.shape -> (2, 50)"""
        assert label.shape[0] == 2 and label.ndim == 2
        label_ones = torch.ones((1, label.shape[1]))
        label_stacked = torch.cat((label, label_ones), dim=0)
        label_affined = torch.mm(self.affine_matrix, label_stacked)
        label = label_affined[:2].round()
        return label

    def temporal_flip(
        self, event_frame: torch.Tensor, label: torch.Tensor, closeness: torch.Tensor
    ):
        assert event_frame.ndim == 4 and event_frame.shape[0] == 2
        assert label.ndim == 2 and label.shape[0] == 2
        assert closeness.ndim == 1 and closeness.shape[0] == label.shape[1]

        event_frame = torch.flip(event_frame, dims=[0, 1])
        label = torch.flip(label, dims=[1])
        closeness = torch.flip(closeness, dims=[0])

        return event_frame, label, closeness
