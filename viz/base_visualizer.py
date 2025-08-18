"""
Base Visualizer Class

Common functionality and utilities for all visualization components.
"""

import warnings
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.logging import get_logger

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configure matplotlib for better plots
plt.style.use("default")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


class BaseVisualizer:
    """
    Base class for all visualization components.

    Provides common functionality:
    - Color palette management
    - Output directory handling
    - Plot saving utilities
    - Data loading helpers
    """

    def __init__(self, output_dir: str = "results/analysis"):
        """
        Initialize base visualizer.

        Args:
            output_dir: Directory to save generated plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color palette for consistent styling
        self.colors = {
            "primary": "#2E86AB",
            "secondary": "#A23B72",
            "success": "#F18F01",
            "warning": "#C73E1D",
            "info": "#6B5B95",
            "light": "#E8E8E8",
            "dark": "#2C3E50",
        }

        self.logger = get_logger(self.__class__.__name__)
        self.logger.info(
            f"{self.__class__.__name__} initialized. Output directory: {self.output_dir}"
        )

    def save_plot(self, fig: plt.Figure, filename: str, dpi: int = 300) -> str:
        """
        Save a plot to the output directory.

        Args:
            fig: Matplotlib figure
            filename: Output filename
            dpi: Resolution for saving

        Returns:
            Path to saved file
        """
        plot_path = self.output_dir / filename
        fig.savefig(
            plot_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        self.logger.info(f"Saved plot: {plot_path}")
        return str(plot_path)

    def create_subplot_grid(
        self, rows: int, cols: int, figsize: tuple = (16, 12)
    ) -> tuple:
        """
        Create a subplot grid with consistent styling.

        Args:
            rows: Number of rows
            cols: Number of columns
            figsize: Figure size

        Returns:
            Tuple of (figure, axes)
        """
        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        # Handle single subplot case
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        elif rows == 1 or cols == 1:
            axes = axes.reshape(-1)

        return fig, axes

    def style_axis(
        self, ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = ""
    ):
        """
        Apply consistent styling to an axis.

        Args:
            ax: Matplotlib axis
            title: Axis title
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        if title:
            ax.set_title(title, fontweight="bold", color=self.colors["dark"])
        if xlabel:
            ax.set_xlabel(xlabel, color=self.colors["dark"])
        if ylabel:
            ax.set_ylabel(ylabel, color=self.colors["dark"])

        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def load_json_data(self, file_path: str) -> Optional[Dict]:
        """
        Load JSON data from file.

        Args:
            file_path: Path to JSON file

        Returns:
            Loaded data or None if file doesn't exist
        """
        try:
            path = Path(file_path)
            if path.exists():
                import json

                with open(path) as f:
                    data = json.load(f)
                self.logger.info(f"Loaded JSON data from: {file_path}")
                return data
            else:
                self.logger.warning(f"JSON file not found: {file_path}")
                return None
        except Exception as e:
            self.logger.error(f"Error loading JSON data from {file_path}: {e}")
            return None

    def load_csv_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load CSV data from file.

        Args:
            file_path: Path to CSV file

        Returns:
            Loaded DataFrame or None if file doesn't exist
        """
        try:
            path = Path(file_path)
            if path.exists():
                df = pd.read_csv(path)
                self.logger.info(f"Loaded CSV data from: {file_path} ({len(df)} rows)")
                return df
            else:
                self.logger.warning(f"CSV file not found: {file_path}")
                return None
        except Exception as e:
            self.logger.error(f"Error loading CSV data from {file_path}: {e}")
            return None

    def generate_simulation_data(self, n_points: int = 100) -> Dict[str, np.ndarray]:
        """
        Generate realistic simulation data for testing.

        Args:
            n_points: Number of data points

        Returns:
            Dictionary of simulation data
        """
        dates = pd.date_range("2023-01-01", periods=n_points, freq="D")

        # Generate realistic trading data
        returns = np.random.normal(0.001, 0.02, n_points)
        cumulative_returns = np.cumprod(1 + returns) - 1

        # Generate feature importance data
        features = ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]
        importance_data = []

        for i in range(n_points // 10):  # Every 10 days
            for feature in features:
                importance_data.append(
                    {
                        "run_id": f"run_{i:03d}",
                        "feature_name": feature,
                        "importance": np.random.exponential(0.1),
                        "rank": np.random.randint(1, 6),
                    }
                )

        return {
            "dates": dates,
            "returns": returns,
            "cumulative_returns": cumulative_returns,
            "importance_data": pd.DataFrame(importance_data),
        }

    def format_percentage(self, value: float) -> str:
        """
        Format a value as a percentage string.

        Args:
            value: Value to format

        Returns:
            Formatted percentage string
        """
        return f"{value:.1%}"

    def format_number(self, value: float, decimals: int = 2) -> str:
        """
        Format a number with specified decimal places.

        Args:
            value: Value to format
            decimals: Number of decimal places

        Returns:
            Formatted number string
        """
        return f"{value:.{decimals}f}"
