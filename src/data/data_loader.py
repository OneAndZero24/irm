from typing import Union
from pathlib import Path
import pandas as pd
import numpy as np
import numpy.typing as npt

class DataLoader:
    """
    A class to load and process time series data from a CSV file.

    Attributes:
        data (pd.DataFrame): The internal DataFrame storing the raw and processed data.
        date_index(npt.NDArray[np.datetime64]): Dates (row keys).
        instrument_index(npt.NDArray[np.int_]): Instruments (column keys).

    Methods:
        get_date(date: np.datetime64) -> npt.NDArray[np.float64]:
            Returns the row of instrument data for the given date.

        get_instrument(idx: int) -> npt.NDArray[np.float64]:
            Returns the column of instrument data for the given index.
    """

    def __init__(self, file_path: Path) -> None:
        """
        Initializes the DataLoader instance and loads data from the given CSV file.

        Args:
            file_path (Path): The path to the CSV file containing the data.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file contains invalid data.
        """

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.data = pd.read_csv(file_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.date_index = self.data['Date'].to_numpy()
        self.instrument_index = np.arange(1, len(self.data.columns))

    def get_date(self, date: np.datetime64) -> npt.NDArray[np.float64]:
        """
        Returns the row of instrument data for the given date.

        Args:
            date (np.datetime64): The date for which to fetch the data.

        Returns:
            npt.NDArray[np.float64]: Array of instrument data for the specified date.

        Raises:
            KeyError: If the date is not found in the data.
        """

        row = self.data.loc[self.data['Date'] == date]
        if row.empty:
            raise KeyError(f"Date {date} not found in the data.")
        return row.iloc[0, 1:].to_numpy()

    def get_instrument(self, idx: int) -> npt.NDArray[np.float64]:
        """
        Returns the column of instrument data for the given index.

        Args:
            idx (int): The index of the instrument.

        Returns:
            npt.NDArray[np.float64]: Array of instrument data for the specified index.

        Raises:
            IndexError: If the index is out of bounds.
        """

        if idx not in self.instrument_index:
            raise IndexError(f"Instrument index {idx} is out of bounds.")
        return self.data.iloc[:, idx].to_numpy()
