"""
ML Visualizer Module

Specialized visualization for ML learning progress, prediction analysis,
and model performance metrics.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .base_visualizer import BaseVisualizer


class MLVisualizer(BaseVisualizer):
    """
    ML-specific visualization component.

    Features:
    - Learning progress tracking
    - Prediction accuracy analysis
    - Model performance metrics
    - Feature importance visualization
    """

    def __init__(self, output_dir: str = "results/analysis"):
        """Initialize ML visualizer."""
        super().__init__(output_dir)

    def create_learning_plots(self, save_plots: bool = True) -> dict[str, str]:
        """Create ML learning progress plots."""
        plots = {}

        # Create comprehensive learning analysis
        fig, axes = self.create_subplot_grid(2, 2, figsize=(16, 12))
        fig.suptitle(
            "ML Learning Progress Analysis",
            fontsize=16,
            fontweight="bold",
            color=self.colors["dark"],
        )

        # 1. Simulated Trade Count Over Time
        self._plot_trade_count_simulation(axes[0])

        # 2. Model Performance Metrics
        self._plot_model_performance_simulation(axes[1])

        # 3. Strategy Performance Comparison
        self._plot_strategy_comparison_simulation(axes[2])

        # 4. Learning Summary
        self._plot_learning_summary_simulation(axes[3])

        plt.tight_layout()

        if save_plots:
            plot_path = self.save_plot(fig, "ml_learning_progress.png")
            plots["ml_learning_progress"] = plot_path

        plt.show()
        return plots

    def create_prediction_plots(self, save_plots: bool = True) -> dict[str, str]:
        """Create ML prediction analysis plots."""
        plots = {}

        fig, axes = self.create_subplot_grid(2, 2, figsize=(16, 12))
        fig.suptitle(
            "ML Prediction Analysis",
            fontsize=16,
            fontweight="bold",
            color=self.colors["dark"],
        )

        # 1. Predicted vs Actual Profits
        self._plot_predicted_vs_actual_simulation(axes[0])

        # 2. Prediction Confidence Distribution
        self._plot_confidence_distribution_simulation(axes[1])

        # 3. Prediction Error Over Time
        self._plot_prediction_error_simulation(axes[2])

        # 4. Feature Importance
        self._plot_feature_importance_simulation(axes[3])

        plt.tight_layout()

        if save_plots:
            plot_path = self.save_plot(fig, "ml_prediction_analysis.png")
            plots["ml_prediction_analysis"] = plot_path

        plt.show()
        return plots

    def _plot_trade_count_simulation(self, ax: plt.Axes):
        """Plot simulated trade count over time."""
        sim_data = self.generate_simulation_data(252)
        dates = sim_data["dates"]

        # Simulate cumulative trade count
        trade_counts = np.cumsum(np.random.poisson(5, len(dates)))

        ax.plot(dates, trade_counts, color=self.colors["primary"], linewidth=2)
        self.style_axis(ax, "Cumulative Trade Count", "Date", "Number of Trades")

        # Add trend line
        z = np.polyfit(range(len(dates)), trade_counts, 1)
        p = np.poly1d(z)
        ax.plot(
            dates,
            p(range(len(dates))),
            "--",
            color=self.colors["secondary"],
            alpha=0.7,
            label=f"Trend: {z[0]:.1f} trades/day",
        )
        ax.legend()

    def _plot_model_performance_simulation(self, ax: plt.Axes):
        """Plot simulated model performance metrics."""
        sim_data = self.generate_simulation_data(100)
        dates = sim_data["dates"]

        # Simulate performance metrics
        accuracy = 0.6 + 0.3 * np.cumsum(np.random.normal(0, 0.01, len(dates)))
        precision = 0.5 + 0.4 * np.cumsum(np.random.normal(0, 0.01, len(dates)))
        recall = 0.4 + 0.5 * np.cumsum(np.random.normal(0, 0.01, len(dates)))

        ax.plot(dates, accuracy, label="Accuracy", color=self.colors["primary"])
        ax.plot(dates, precision, label="Precision", color=self.colors["secondary"])
        ax.plot(dates, recall, label="Recall", color=self.colors["success"])

        self.style_axis(ax, "Model Performance Metrics", "Date", "Score")
        ax.legend()
        ax.set_ylim(0, 1)

    def _plot_strategy_comparison_simulation(self, ax: plt.Axes):
        """Plot simulated strategy performance comparison."""
        sim_data = self.generate_simulation_data(252)
        dates = sim_data["dates"]

        # Simulate different strategy returns
        baseline_returns = np.cumprod(1 + np.random.normal(0.0005, 0.015, len(dates))) - 1
        ml_returns = np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates))) - 1
        enhanced_returns = np.cumprod(1 + np.random.normal(0.0015, 0.018, len(dates))) - 1

        ax.plot(dates, baseline_returns, label="Baseline", color=self.colors["light"])
        ax.plot(dates, ml_returns, label="ML Strategy", color=self.colors["primary"])
        ax.plot(dates, enhanced_returns, label="Enhanced ML", color=self.colors["success"])

        self.style_axis(ax, "Strategy Performance Comparison", "Date", "Cumulative Return")
        ax.legend()

    def _plot_learning_summary_simulation(self, ax: plt.Axes):
        """Plot simulated learning summary statistics."""
        # Simulate learning metrics
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
        final_values = [0.85, 0.82, 0.88, 0.85, 0.91]
        initial_values = [0.65, 0.62, 0.68, 0.65, 0.71]

        x = np.arange(len(metrics))
        width = 0.35

        ax.bar(
            x - width / 2,
            initial_values,
            width,
            label="Initial",
            color=self.colors["light"],
            alpha=0.7,
        )
        ax.bar(
            x + width / 2,
            final_values,
            width,
            label="Final",
            color=self.colors["primary"],
        )

        self.style_axis(ax, "Learning Progress Summary", "Metrics", "Score")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45)
        ax.legend()
        ax.set_ylim(0, 1)

    def _plot_predicted_vs_actual_simulation(self, ax: plt.Axes):
        """Plot simulated predicted vs actual profits."""
        # Generate simulated data
        actual_profits = np.random.normal(0.02, 0.05, 100)
        predicted_profits = actual_profits + np.random.normal(0, 0.01, 100)

        ax.scatter(actual_profits, predicted_profits, alpha=0.6, color=self.colors["primary"])

        # Add perfect prediction line
        min_val = min(actual_profits.min(), predicted_profits.min())
        max_val = max(actual_profits.max(), predicted_profits.max())
        ax.plot([min_val, max_val], [min_val, max_val], "--", color=self.colors["dark"])

        self.style_axis(ax, "Predicted vs Actual Profits", "Actual Profit", "Predicted Profit")

        # Add correlation coefficient
        corr = np.corrcoef(actual_profits, predicted_profits)[0, 1]
        ax.text(
            0.05,
            0.95,
            f"Correlation: {corr:.3f}",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    def _plot_confidence_distribution_simulation(self, ax: plt.Axes):
        """Plot simulated prediction confidence distribution."""
        # Generate simulated confidence scores
        confidence_scores = np.random.beta(2, 2, 1000)

        ax.hist(confidence_scores, bins=30, alpha=0.7, color=self.colors["primary"])
        self.style_axis(ax, "Prediction Confidence Distribution", "Confidence Score", "Frequency")

        # Add mean line
        mean_conf = confidence_scores.mean()
        ax.axvline(
            mean_conf,
            color=self.colors["warning"],
            linestyle="--",
            label=f"Mean: {mean_conf:.3f}",
        )
        ax.legend()

    def _plot_prediction_error_simulation(self, ax: plt.Axes):
        """Plot simulated prediction error over time."""
        sim_data = self.generate_simulation_data(100)
        dates = sim_data["dates"]

        # Simulate prediction errors
        errors = np.random.normal(0, 0.01, len(dates))
        error_ma = pd.Series(errors).rolling(window=10).mean()

        ax.plot(dates, errors, alpha=0.5, color=self.colors["light"])
        ax.plot(dates, error_ma, color=self.colors["primary"], linewidth=2)
        ax.axhline(0, color=self.colors["dark"], linestyle="-", alpha=0.5)

        self.style_axis(ax, "Prediction Error Over Time", "Date", "Error")

    def _plot_feature_importance_simulation(self, ax: plt.Axes):
        """Plot simulated feature importance."""
        # Generate simulated feature importance
        features = ["Price_Momentum", "Volatility", "Volume", "RSI", "MACD"]
        importance = np.random.exponential(0.1, len(features))

        # Sort by importance
        sorted_idx = np.argsort(importance)[::-1]
        features_sorted = [features[i] for i in sorted_idx]
        importance_sorted = importance[sorted_idx]

        bars = ax.bar(features_sorted, importance_sorted, color=self.colors["primary"])

        # Add value labels on bars
        for bar, value in zip(bars, importance_sorted, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        self.style_axis(ax, "Feature Importance", "Features", "Importance")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    def load_ml_data(self) -> dict | None:
        """Load ML analysis data from existing files."""
        try:
            # Try to load from various possible locations
            ml_files = [
                "results/ml_analysis/ml_analysis_report.md",
                "state/ml_profit_learner/performance_history.pkl",
                "runs/feature_importance.csv",
            ]

            for file_path in ml_files:
                if Path(file_path).exists():
                    self.logger.info(f"Found ML data at: {file_path}")
                    return {"source": file_path, "exists": True}

            self.logger.warning("No ML data files found")
            return None

        except Exception as e:
            self.logger.error(f"Error loading ML data: {e}")
            return None
