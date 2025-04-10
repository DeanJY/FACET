import numpy as np
from scipy.io import loadmat, savemat


class MatProcessor:
    def __init__(self, filepath: str, mode: str = "r") -> None:
        """
        Initialize the MatProcessor with file path and mode.
        Args:
            filepath: Path to the .mat file.
            mode: Mode to open the file in, including 'r' (read), 'u' (update), 'w' (write).
        Returns:
            None
        """
        self.filepath = filepath
        self.mode = mode
        self.mat_data = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        """Open or load the .mat file data based on the mode."""
        if self.mode in ["r", "u"]:
            self.mat_data = loadmat(self.filepath)
        elif self.mode == "w":
            self.mat_data = {}

    def close(self):
        """Save the .mat file data if in write or update mode and then clear the mat_data."""
        if self.mode in ["w", "u"] and self.mat_data is not None:
            savemat(self.filepath, self.mat_data)
        self.mat_data = None

    def read_all(self) -> dict:
        """
        Read all variables from the .mat file at once.
        Returns:
            A dictionary containing all variables loaded from the .mat file.
        """
        mat_data = loadmat(self.filepath)
        # Strip out MATLAB metadata __globals__, __version__, __header__ if present
        mat_data = {k: v for k, v in mat_data.items() if not k.startswith("__")}

    def read_data(self, variable_name: str) -> np.ndarray:
        """
        Read data from a variable in the .mat file. Automatically handles file opening and closing.
        Args:
            variable_name: Name of the variable to read.
        Returns:
            Data contained in the variable.
        """
        if self.mode in ["r", "u"]:
            with self:
                if variable_name in self.mat_data:
                    return self.mat_data[variable_name]
                else:
                    raise KeyError(f"{variable_name} not found in the .mat file.")
        else:
            raise Exception("File not opened in a readable mode.")

    def write_data(self, variable_name: str, data: np.ndarray):
        """
        Write data to a variable in the .mat file. Automatically handles file opening and closing.
        Args:
            variable_name: Name of the variable to write to.
            data: Data to write to the variable.
        Returns:
            None
        """
        if self.mode in ["w", "u"]:
            with self:
                self.mat_data[variable_name] = data
        else:
            raise Exception("File not opened in a writable or updatable mode.")

    def list_variables(self) -> list:
        """
        List all variable names in the .mat file. Automatically handles file opening and closing.
        Returns:
            List of variable names in the .mat file.
        """
        with self:
            return list(self.mat_data.keys())


def main():
    mat_path = "/mnt/data2T/junyuan/eye-tracking/EV_Eye_dataset/processed_data/Frame_event_pupil_track_result/left/update_20_point_user1_session_1_0_1.mat"
    processor = MatProcessor(mat_path, "u")

    with processor:
        print("Variables in the .mat file:", processor.list_variables())
        # Assuming 'data' is a variable in the .mat file
        data = processor.read_data("matcell")
        print("Data:", data)
        print(type(data), data.shape)

        # # Write new data to the .mat file
        # new_data = np.random.rand(5, 5)
        # processor.write_data("new_data", new_data)
        # print("New data written to the .mat file.")


if __name__ == "__main__":
    main()
