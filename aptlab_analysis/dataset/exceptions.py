from typing import Union
from pathlib import Path


class CSVLoadError(Exception):
    def __init__(
        self,
        csv_path: Union[Path, str],
        message: str = "No such CSV file or data is empty",
    ):
        self.csv_path = csv_path
        self.message = message
        super().__init__(self.csv_path, self.message)

    def __str__(self):
        return f"{self.message}: '{self.csv_path}'"
