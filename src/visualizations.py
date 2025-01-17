import os
import typing
import logging
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


log = logging.getLogger(__name__)

def plot_sim(Ys: list[np.ndarray], y: typing.Optional[np.ndarray], x: np.ndarray, save_plot: bool):
    """
    Plot simulation results.

    Args:
        Ys (List[np.ndarray]): List of 1D arrays to be plotted as individual lines.
        y (typing.Optional[np.ndarray]): 1D array of values to be marked with crosses on the plot.
        x (np.ndarray): 1D array of x-axis labels corresponding to Ys and y.
        save_plot (bool): Whether to save the plot to OUTPUT_DIR.

    Raises:
        ValueError: If the lengths of x, y, or any element in Ys do not match.
    """

    tmp = Ys
    if y is not None:
        tmp = Ys+[y]
    if any(len(arr) != len(x) for arr in tmp):
        raise ValueError("All input arrays must have the same length as x.")
    
    plt.figure(figsize=(10, 6))

    for Y in Ys:
        plt.plot(x, Y, alpha=0.4)

    upper_bound = np.quantile(Ys, q=0.95, axis=0)
    lower_bound = np.quantile(Ys, q=0.05, axis=0)
    average_line = np.mean(Ys, axis=0)
    plt.plot(x, upper_bound, 'k--', label="Upper Bound")
    plt.plot(x, lower_bound, 'k--', label="Lower Bound")
    plt.plot(x, average_line, 'k:', label="Average")
    if y is not None:
        plt.plot(x, y, color='red', label="Real")

    plt.xlabel("X-axis")
    plt.ylabel("Values")
    plt.title("Simulation Plot ")
    plt.legend()
    plt.tight_layout()

    if save_plot:
        output_dir = os.getenv("OUTPUT_DIR", ".")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(output_dir, f"simulation{timestamp}.png")
        plt.savefig(file_path, dpi=300)
        log.info(f"Plot saved to {file_path}")
    
    plt.show()