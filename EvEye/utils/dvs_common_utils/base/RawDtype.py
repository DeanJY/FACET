import numpy as np

raw_event_type = np.dtype(
    [("t", np.int64), ("x", np.int64), ("y", np.int64), ("p", np.int64)]
)
raw_label_type = np.dtype(
    [("t", np.int64), ("x", np.int64), ("y", np.int64), ("close", np.int64)]
)
raw_ellipse_type = np.dtype(
    [
        ("t", np.int64),
        ("x", np.float64),
        ("y", np.float64),
        ("a", np.float64),
        ("b", np.float64),
        ("ang", np.float64),
    ]
)

if __name__ == "__main__":
    x = np.array([1, 3, 5], dtype=np.uint8)
    y = np.array([])
