import numpy as np
from EvEye.utils.dvs_common_utils.base.RawDtype import (
    raw_event_type,
    raw_label_type,
    raw_ellipse_type,
)


class TxtProcessor:
    def __init__(self, txt_path, encoding="utf-8"):
        self.filename = txt_path
        self.encoding = encoding

    def read(self):
        """Read the content of the txt file."""
        with open(self.filename, "r", encoding=self.encoding) as file:
            return file.read()

    def read_lines(self, n=None):
        """
        Read the content of the txt file.

        Args:
            n: Number of lines to read. If n is None, read all lines.
        Returns:
            List of lines read from the txt file.
        """
        with open(self.filename, "r", encoding=self.encoding) as file:
            if n is None:
                return file.readlines()
            else:
                return [file.readline() for _ in range(n)]

    def write(self, lines, mode="w"):
        """Write lines to the txt file."""
        with open(self.filename, mode, encoding=self.encoding) as file:
            file.writelines(lines)

    def preview(self, n=5):
        """Preview the content of the first n lines of the txt file."""
        return self.read_lines(n)

    # def load_events_from_txt_v1(self):
    #     """Load events from txt file."""
    #     events = []
    #     with open(self.filename, "r", encoding=self.encoding) as file:
    #         for line in file:
    #             if line.strip() == "":
    #                 continue
    #             t, x, y, p = line.split()
    #             events.append((t, x, y, p))
    #     return np.array(events, dtype=raw_event_type)

    def load_events_from_txt(self):
        """Load events from txt file using numpy."""

        events = np.loadtxt(self.filename, dtype=raw_event_type, encoding=self.encoding)
        return events  # t, x, y, p

    def load_labels_from_txt(self):
        """Load labels from txt file using numpy."""

        labels = np.loadtxt(
            self.filename, dtype=raw_label_type, delimiter=",", encoding=self.encoding
        )
        return labels  # t, x, y, close

    def load_ellipses_from_txt(self):
        """Load ellipse from txt file using numpy."""

        ellipses = np.loadtxt(
            self.filename, dtype=raw_ellipse_type, delimiter=" ", encoding=self.encoding
        )
        return ellipses


def main():
    txt_processor = TxtProcessor(
        "/mnt/data2T/junyuan/eye-tracking/datasets/DavisEyeCenterDataset/train/label/user1_left_session_1_0_1_centers.txt"
    )
    print(txt_processor.preview(5))
    print(txt_processor.read_lines(100))
    # events = txt_processor.load_events_from_txt()
    # print(events.shape)
    # # 现在 events 是一个结构化数组，可以按照类型字段访问
    # print(events[0])
    # print(events["x"], events["x"].shape)  # 打印所有的 x 坐标
    # print(events["y"], events["y"].shape)  # 打印所有的 y 坐标
    # print(events["p"], events["p"].shape)  # 打印所有的极性
    # print(events["t"], events["t"].shape)  # 打印所有的时间戳


if __name__ == "__main__":
    main()
