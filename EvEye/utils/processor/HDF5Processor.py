import numpy as np
import h5py


class HDF5Processor:
    def __init__(self, filepath: str, mode: str = "r") -> None:
        """
        Initialize the HDF5Processor with file path and mode.

        Args:
            filepath: Path to the HDF5 file.
            mode: Mode to open the file in. Including 'r', 'r+', 'w', 'w-', 'x', 'a'.
        Returns:
            None
        """
        self.filepath = filepath
        self.mode = mode
        self.file = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        """Open the HDF5 file."""
        if not self.file:
            self.file = h5py.File(self.filepath, self.mode)

    def close(self):
        """Close the HDF5 file if it's open."""
        if self.file:
            self.file.close()
            self.file = None

    def read_data(self, dataset_name: str) -> np.ndarray:
        """
        Read data from a dataset.

        Args:
            dataset_name: Name of the dataset to read.
        Returns:
            Data contained in the dataset.
        """
        with self:
            return self.file[dataset_name][:]

    def write_data(self, dataset_name: str, data: np.ndarray):
        """
        Write data to a dataset.

        Args:
            dataset_name: Name of the dataset to write to.
            data: Data to write to the dataset.
        Returns:
            None
        """
        with self:
            self.file.create_dataset(
                dataset_name, data=data, compression="gzip", compression_opts=9
            )

    def list_datasets(self):
        """List all datasets in the file."""
        with self:
            return list(self.file.keys())

    def get_attribute(self, dataset_name: str, attr_name: str):
        """
        Get an attribute from the dataset.

        Args:
            dataset_name: Name of the dataset.
            attr_name: Name of the attribute to get.
        Returns:
            Value of the attribute.
        """
        with self:
            return self.file[dataset_name].attrs[attr_name]

    def set_attribute(self, dataset_name: str, attr_name: str, attr_value):
        """
        Set an attribute for the dataset.

        Args:
            dataset_name: Name of the dataset.
            attr_name: Name of the attribute to set.
            attr_value: Value of the attribute.
        Returns:
            None
        """
        with self:
            self.file[dataset_name].attrs[attr_name] = attr_value


def main():
    h5_path = "/mnt/data2T/junyuan/eye-tracking/1_2.h5"
    processor = HDF5Processor(h5_path, "r")
    print(processor.list_datasets())
    events = processor.read_data("events")
    print(events)
    # data = processor.read_data("data")
    # label = processor.read_data("label")
    # print(type(data), data.shape)
    # print(type(label), label.shape)


if __name__ == "__main__":
    main()
