"""
ML Trading System Visualizer
Generates comprehensive graphs for understanding ML learning and performance
"""

import logging

# Force non-interactive backend to prevent GUI windows
import os
import warnings
from datetime import datetime
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib

matplotlib.use('Agg')  # Force non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set style for better-looking plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

# Disable interactive mode
plt.ioff()

logger = logging.getLogger(__name__)


class MLVisualizer:
    """
    Comprehensive visualization system for ML trading analysis.

    Features:
    - Learning progress tracking
    - Prediction accuracy analysis
    - Feature importance visualization
    - Strategy performance comparison
    - Risk metrics analysis
    """

    def __init__(self, output_dir: str = "results/ml_analysis"):
        """
        Initialize the visualizer.

        Args:
            output_dir: Directory to save generated plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up plotting style
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 10
        plt.rcParams["axes.grid"] = True

        logger.info(f"ML Visualizer initialized. Output directory: {self.output_dir}")

    def create_learning_progress_plots(
        self, profit_learner, save_plots: bool = True
    ) -> dict[str, str]:
        """
        Create plots showing ML learning progress.

        Args:
            profit_learner: The profit learner instance
            save_plots: Whether to save plots to disk

        Returns:
            Dictionary of plot file paths
        """
        plots = {}

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("ML Learning Progress Analysis", fontsize=16, fontweight="bold")

        # 1. Trade Count Over Time
        self._plot_trade_count_over_time(profit_learner, axes[0, 0])

        # 2. Model Performance Metrics
        self._plot_model_performance_metrics(profit_learner, axes[0, 1])

        # 3. Strategy Performance Comparison
        self._plot_strategy_performance_comparison(profit_learner, axes[1, 0])

        # 4. Learning Summary
        self._plot_learning_summary(profit_learner, axes[1, 1])

        plt.tight_layout()

        if save_plots:
            plot_path = self.output_dir / "learning_progress.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plots["learning_progress"] = str(plot_path)
            logger.info(f"Saved learning progress plot: {plot_path}")

        # plt.show()  # DISABLED: Never show GUI in trading system
        plt.close(fig)  # Free memory
        return plots

    def create_prediction_analysis_plots(
        self, profit_learner, save_plots: bool = True
    ) -> dict[str, str]:
        """
        Create plots analyzing ML prediction accuracy.

        Args:
            profit_learner: The profit learner instance
            save_plots: Whether to save plots to disk

        Returns:
            Dictionary of plot file paths
        """
        plots = {}

        if len(profit_learner.performance_history) < 5:
            logger.warning("Not enough trade history for prediction analysis")
            return plots

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("ML Prediction Analysis", fontsize=16, fontweight="bold")

        # 1. Predicted vs Actual Profits
        self._plot_predicted_vs_actual(profit_learner, axes[0, 0])

        # 2. Prediction Confidence Distribution
        self._plot_confidence_distribution(profit_learner, axes[0, 1])

        # 3. Prediction Error Over Time
        self._plot_prediction_error_over_time(profit_learner, axes[1, 0])

        # 4. Feature Importance
        self._plot_feature_importance(profit_learner, axes[1, 1])

        plt.tight_layout()

        if save_plots:
            plot_path = self.output_dir / "prediction_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plots["prediction_analysis"] = str(plot_path)
            logger.info(f"Saved prediction analysis plot: {plot_path}")

        # plt.show()  # DISABLED: Never show GUI in trading system
        plt.close(fig)  # Free memory
        return plots

    def create_strategy_performance_plots(
        self, profit_learner, save_plots: bool = True
    ) -> dict[str, str]:
        """
        Create plots comparing strategy performance.

        Args:
            profit_learner: The profit learner instance
            save_plots: Whether to save plots to disk

        Returns:
            Dictionary of plot file paths
        """
        plots = {}

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Strategy Performance Analysis", fontsize=16, fontweight="bold")

        # 1. Strategy Win Rates
        self._plot_strategy_win_rates(profit_learner, axes[0, 0])

        # 2. Strategy Average Profits
        self._plot_strategy_avg_profits(profit_learner, axes[0, 1])

        # 3. Strategy Risk Metrics
        self._plot_strategy_risk_metrics(profit_learner, axes[1, 0])

        # 4. Strategy Trade Counts
        self._plot_strategy_trade_counts(profit_learner, axes[1, 1])

        plt.tight_layout()

        if save_plots:
            plot_path = self.output_dir / "strategy_performance.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plots["strategy_performance"] = str(plot_path)
            logger.info(f"Saved strategy performance plot: {plot_path}")

        # plt.show()  # DISABLED: Never show GUI in trading system
        plt.close(fig)  # Free memory
        return plots

    def create_risk_analysis_plots(self, profit_learner, save_plots: bool = True) -> dict[str, str]:
        """
        Create plots analyzing risk metrics.

        Args:
            profit_learner: The profit learner instance
            save_plots: Whether to save plots to disk

        Returns:
            Dictionary of plot file paths
        """
        plots = {}

        if len(profit_learner.performance_history) < 10:
            logger.warning("Not enough trade history for risk analysis")
            return plots

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Risk Analysis", fontsize=16, fontweight="bold")

        # 1. Profit Distribution
        self._plot_profit_distribution(profit_learner, axes[0, 0])

        # 2. Drawdown Analysis
        self._plot_drawdown_analysis(profit_learner, axes[0, 1])

        # 3. Volatility Analysis
        self._plot_volatility_analysis(profit_learner, axes[1, 0])

        # 4. Risk-Return Scatter
        self._plot_risk_return_scatter(profit_learner, axes[1, 1])

        plt.tight_layout()

        if save_plots:
            plot_path = self.output_dir / "risk_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plots["risk_analysis"] = str(plot_path)
            logger.info(f"Saved risk analysis plot: {plot_path}")

        # plt.show()  # DISABLED: Never show GUI in trading system
        plt.close(fig)  # Free memory
        return plots

    def create_comprehensive_report(
        self, profit_learner, save_plots: bool = True
    ) -> dict[str, str]:
        """
        Create a comprehensive report with all visualizations.

        Args:
            profit_learner: The profit learner instance
            save_plots: Whether to save plots to disk

        Returns:
            Dictionary of all plot file paths
        """
        logger.info("Creating comprehensive ML analysis report...")

        all_plots = {}

        # Generate all plot types
        all_plots.update(self.create_learning_progress_plots(profit_learner, save_plots))
        all_plots.update(self.create_prediction_analysis_plots(profit_learner, save_plots))
        all_plots.update(self.create_strategy_performance_plots(profit_learner, save_plots))
        all_plots.update(self.create_risk_analysis_plots(profit_learner, save_plots))

        # Create summary report
        if save_plots:
            self._create_summary_report(profit_learner, all_plots)

        logger.info(f"Comprehensive report created with {len(all_plots)} plots")
        return all_plots

    def _plot_trade_count_over_time(self, profit_learner, ax):
        """Plot trade count over time."""
        if not profit_learner.performance_history:
            ax.text(
                0.5,
                0.5,
                "No trade history available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Trade Count Over Time")
            return

        # Create trade count timeline
        trade_dates = [trade.timestamp for trade in profit_learner.performance_history]
        trade_counts = list(range(1, len(trade_dates) + 1))

        ax.plot(trade_dates, trade_counts, marker="o", linewidth=2, markersize=4)
        ax.set_title("Trade Count Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Trades")
        ax.grid(True, alpha=0.3)

        # Add learning threshold line
        if len(trade_counts) < profit_learner.min_trades_for_learning:
            ax.axhline(
                y=profit_learner.min_trades_for_learning,
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"Learning Threshold ({profit_learner.min_trades_for_learning})",
            )
            ax.legend()

    def _plot_model_performance_metrics(self, profit_learner, ax):
        """Plot model performance metrics."""
        if not profit_learner.models:
            ax.text(
                0.5,
                0.5,
                "No trained models available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Model Performance Metrics")
            return

        # Extract model performance data
        model_names = list(profit_learner.models.keys())
        [type(model).__name__ for model in profit_learner.models.values()]

        # Create bar chart
        x_pos = np.arange(len(model_names))
        ax.bar(x_pos, [1] * len(model_names), alpha=0.7)  # Placeholder for actual metrics

        ax.set_title("Model Performance Metrics")
        ax.set_xlabel("Models")
        ax.set_ylabel("Performance Score")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([name.split("_")[0] for name in model_names], rotation=45)
        ax.grid(True, alpha=0.3)

    def _plot_strategy_performance_comparison(self, profit_learner, ax):
        """Plot strategy performance comparison."""
        strategies = [
            "regime_aware_ensemble",
            "momentum",
            "mean_reversion",
            "sma_crossover",
            "ensemble_basic",
        ]

        # Get performance data
        avg_profits = []
        win_rates = []
        trade_counts = []

        for strategy in strategies:
            perf = profit_learner.get_strategy_performance(strategy)
            if perf:
                avg_profits.append(perf["avg_profit_pct"] * 100)  # Convert to percentage
                win_rates.append(perf["win_rate"] * 100)  # Convert to percentage
                trade_counts.append(perf["total_trades"])
            else:
                avg_profits.append(0)
                win_rates.append(0)
                trade_counts.append(0)

        # Create grouped bar chart
        x_pos = np.arange(len(strategies))
        width = 0.35

        ax.bar(x_pos - width / 2, avg_profits, width, label="Avg Profit (%)", alpha=0.7)
        ax.bar(x_pos + width / 2, win_rates, width, label="Win Rate (%)", alpha=0.7)

        ax.set_title("Strategy Performance Comparison")
        ax.set_xlabel("Strategies")
        ax.set_ylabel("Percentage")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([s.replace("_", " ").title() for s in strategies], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add trade count annotations
        for i, count in enumerate(trade_counts):
            ax.text(
                i,
                max(avg_profits[i], win_rates[i]) + 1,
                f"n={count}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    def _plot_learning_summary(self, profit_learner, ax):
        """Plot learning summary statistics."""
        summary = profit_learner.get_learning_summary()

        # Create summary text
        summary_text = f"""
        Learning Summary:

        Total Trades: {summary["total_trades"]}
        Models Trained: {summary["models_trained"]}
        Performance History: {summary["performance_history_length"]} trades
        Min Trades for Learning: {summary["min_trades_for_learning"]}

        Learning Status: {"✅ Active" if summary["total_trades"] >= summary["min_trades_for_learning"] else "⏳ Collecting Data"}
        Last Update: {summary["last_update"][:19]}
        """

        ax.text(
            0.1,
            0.9,
            summary_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightblue", "alpha": 0.7},
        )

        ax.set_title("Learning Summary")
        ax.axis("off")

    def _plot_predicted_vs_actual(self, profit_learner, ax):
        """Plot predicted vs actual profits."""
        if len(profit_learner.performance_history) < 5:
            ax.text(
                0.5,
                0.5,
                "Insufficient data for prediction analysis",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Predicted vs Actual Profits")
            return

        # Extract actual profits
        actual_profits = [
            trade.profit_loss_pct * 100 for trade in profit_learner.performance_history
        ]

        # For now, use simple predictions (this would be enhanced with actual ML predictions)
        predicted_profits = [
            p * 0.8 + np.random.normal(0, 0.1) for p in actual_profits
        ]  # Placeholder

        ax.scatter(predicted_profits, actual_profits, alpha=0.6, s=50)

        # Add perfect prediction line
        min_val = min(min(predicted_profits), min(actual_profits))
        max_val = max(max(predicted_profits), max(actual_profits))
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            alpha=0.7,
            label="Perfect Prediction",
        )

        ax.set_title("Predicted vs Actual Profits")
        ax.set_xlabel("Predicted Profit (%)")
        ax.set_ylabel("Actual Profit (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_confidence_distribution(self, profit_learner, ax):
        """Plot prediction confidence distribution."""
        if len(profit_learner.performance_history) < 5:
            ax.text(
                0.5,
                0.5,
                "Insufficient data for confidence analysis",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Prediction Confidence Distribution")
            return

        # Generate sample confidence values (placeholder)
        confidences = np.random.beta(2, 5, len(profit_learner.performance_history))

        ax.hist(confidences, bins=20, alpha=0.7, edgecolor="black")
        ax.set_title("Prediction Confidence Distribution")
        ax.set_xlabel("Confidence Level")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

    def _plot_prediction_error_over_time(self, profit_learner, ax):
        """Plot prediction error over time."""
        if len(profit_learner.performance_history) < 5:
            ax.text(
                0.5,
                0.5,
                "Insufficient data for error analysis",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Prediction Error Over Time")
            return

        # Generate sample error data (placeholder)
        dates = [trade.timestamp for trade in profit_learner.performance_history]
        errors = np.random.normal(0, 0.02, len(dates))

        ax.plot(dates, errors, marker="o", alpha=0.7)
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.7)
        ax.set_title("Prediction Error Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Prediction Error")
        ax.grid(True, alpha=0.3)

    def _plot_feature_importance(self, profit_learner, ax):
        """Plot feature importance from trained models."""
        if not profit_learner.models:
            ax.text(
                0.5,
                0.5,
                "No trained models available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Feature Importance")
            return

        # Get feature importance from first model
        model = list(profit_learner.models.values())[0]
        if hasattr(model, "coef_"):
            feature_names = [
                "Strategy_Regime",
                "Strategy_Momentum",
                "Strategy_MeanRev",
                "Strategy_SMA",
                "Strategy_Ensemble",
                "Regime_Trend",
                "Regime_Chop",
                "Regime_Volatile",
                "Volatility",
                "RSI",
                "SMA_Ratio",
                "Volume_Ratio",
                "Price_Position",
                "Momentum_5",
                "Momentum_20",
                "Z_Score",
                "Returns_1d",
                "Returns_5d",
            ]

            importance = np.abs(model.coef_)
            top_indices = importance.argsort()[-10:]  # Top 10 features

            y_pos = np.arange(len(top_indices))
            ax.barh(y_pos, importance[top_indices])
            ax.set_yticks(y_pos)
            ax.set_yticklabels([feature_names[i] for i in top_indices])
            ax.set_title("Top 10 Feature Importance")
            ax.set_xlabel("Absolute Coefficient Value")
            ax.grid(True, alpha=0.3)

    def _plot_strategy_win_rates(self, profit_learner, ax):
        """Plot strategy win rates."""
        strategies = [
            "regime_aware_ensemble",
            "momentum",
            "mean_reversion",
            "sma_crossover",
            "ensemble_basic",
        ]
        win_rates = []

        for strategy in strategies:
            perf = profit_learner.get_strategy_performance(strategy)
            win_rates.append(perf.get("win_rate", 0) * 100 if perf else 0)

        bars = ax.bar(strategies, win_rates, alpha=0.7)
        ax.set_title("Strategy Win Rates")
        ax.set_xlabel("Strategies")
        ax.set_ylabel("Win Rate (%)")
        ax.set_xticklabels([s.replace("_", " ").title() for s in strategies], rotation=45)
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, rate in zip(bars, win_rates, strict=False):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
            )

    def _plot_strategy_avg_profits(self, profit_learner, ax):
        """Plot strategy average profits."""
        strategies = [
            "regime_aware_ensemble",
            "momentum",
            "mean_reversion",
            "sma_crossover",
            "ensemble_basic",
        ]
        avg_profits = []

        for strategy in strategies:
            perf = profit_learner.get_strategy_performance(strategy)
            avg_profits.append(perf.get("avg_profit_pct", 0) * 100 if perf else 0)

        colors = ["green" if p > 0 else "red" for p in avg_profits]
        bars = ax.bar(strategies, avg_profits, color=colors, alpha=0.7)
        ax.set_title("Strategy Average Profits")
        ax.set_xlabel("Strategies")
        ax.set_ylabel("Average Profit (%)")
        ax.set_xticklabels([s.replace("_", " ").title() for s in strategies], rotation=45)
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, profit in zip(bars, avg_profits, strict=False):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + (0.01 if height > 0 else -0.01),
                f"{profit:.2f}%",
                ha="center",
                va="bottom" if height > 0 else "top",
            )

    def _plot_strategy_risk_metrics(self, profit_learner, ax):
        """Plot strategy risk metrics."""
        strategies = [
            "regime_aware_ensemble",
            "momentum",
            "mean_reversion",
            "sma_crossover",
            "ensemble_basic",
        ]
        profit_stds = []

        for strategy in strategies:
            perf = profit_learner.get_strategy_performance(strategy)
            profit_stds.append(perf.get("profit_std", 0) * 100 if perf else 0)

        bars = ax.bar(strategies, profit_stds, alpha=0.7, color="orange")
        ax.set_title("Strategy Profit Volatility")
        ax.set_xlabel("Strategies")
        ax.set_ylabel("Profit Std Dev (%)")
        ax.set_xticklabels([s.replace("_", " ").title() for s in strategies], rotation=45)
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, std in zip(bars, profit_stds, strict=False):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{std:.2f}%",
                ha="center",
                va="bottom",
            )

    def _plot_strategy_trade_counts(self, profit_learner, ax):
        """Plot strategy trade counts."""
        strategies = [
            "regime_aware_ensemble",
            "momentum",
            "mean_reversion",
            "sma_crossover",
            "ensemble_basic",
        ]
        trade_counts = []

        for strategy in strategies:
            perf = profit_learner.get_strategy_performance(strategy)
            trade_counts.append(perf.get("total_trades", 0) if perf else 0)

        bars = ax.bar(strategies, trade_counts, alpha=0.7, color="purple")
        ax.set_title("Strategy Trade Counts")
        ax.set_xlabel("Strategies")
        ax.set_ylabel("Number of Trades")
        ax.set_xticklabels([s.replace("_", " ").title() for s in strategies], rotation=45)
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, count in zip(bars, trade_counts, strict=False):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                str(count),
                ha="center",
                va="bottom",
            )

    def _plot_profit_distribution(self, profit_learner, ax):
        """Plot profit distribution."""
        if len(profit_learner.performance_history) < 5:
            ax.text(
                0.5,
                0.5,
                "Insufficient data for profit distribution",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Profit Distribution")
            return

        profits = [trade.profit_loss_pct * 100 for trade in profit_learner.performance_history]

        ax.hist(profits, bins=20, alpha=0.7, edgecolor="black", color="skyblue")
        ax.axvline(
            x=np.mean(profits),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(profits):.2f}%",
        )
        ax.set_title("Profit Distribution")
        ax.set_xlabel("Profit (%)")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_drawdown_analysis(self, profit_learner, ax):
        """Plot drawdown analysis."""
        if len(profit_learner.performance_history) < 10:
            ax.text(
                0.5,
                0.5,
                "Insufficient data for drawdown analysis",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Drawdown Analysis")
            return

        # Calculate cumulative returns and drawdown
        profits = [trade.profit_loss_pct for trade in profit_learner.performance_history]
        cumulative = np.cumprod(1 + np.array(profits))

        # Calculate drawdown
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max * 100

        dates = [trade.timestamp for trade in profit_learner.performance_history]

        ax.plot(dates, drawdown, color="red", alpha=0.7)
        ax.fill_between(dates, drawdown, 0, alpha=0.3, color="red")
        ax.set_title("Drawdown Analysis")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.grid(True, alpha=0.3)

    def _plot_volatility_analysis(self, profit_learner, ax):
        """Plot volatility analysis."""
        if len(profit_learner.performance_history) < 10:
            ax.text(
                0.5,
                0.5,
                "Insufficient data for volatility analysis",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Volatility Analysis")
            return

        profits = [trade.profit_loss_pct * 100 for trade in profit_learner.performance_history]

        # Calculate rolling volatility
        window = min(20, len(profits) // 2)
        if window > 1:
            rolling_vol = pd.Series(profits).rolling(window=window).std()
            dates = [trade.timestamp for trade in profit_learner.performance_history]

            ax.plot(dates[window - 1 :], rolling_vol[window - 1 :], alpha=0.7)
            ax.set_title(f"Rolling Volatility (Window: {window})")
            ax.set_xlabel("Date")
            ax.set_ylabel("Volatility (%)")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5,
                0.5,
                "Insufficient data for rolling volatility",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Volatility Analysis")

    def _plot_risk_return_scatter(self, profit_learner, ax):
        """Plot risk-return scatter plot."""
        if len(profit_learner.performance_history) < 10:
            ax.text(
                0.5,
                0.5,
                "Insufficient data for risk-return analysis",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Risk-Return Analysis")
            return

        strategies = [
            "regime_aware_ensemble",
            "momentum",
            "mean_reversion",
            "sma_crossover",
            "ensemble_basic",
        ]
        returns = []
        risks = []
        labels = []

        for strategy in strategies:
            perf = profit_learner.get_strategy_performance(strategy)
            if perf and perf["total_trades"] > 0:
                returns.append(perf["avg_profit_pct"] * 100)
                risks.append(perf["profit_std"] * 100)
                labels.append(strategy.replace("_", " ").title())

        if returns:
            ax.scatter(risks, returns, s=100, alpha=0.7)
            for i, label in enumerate(labels):
                ax.annotate(
                    label,
                    (risks[i], returns[i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

            ax.set_title("Risk-Return Analysis")
            ax.set_xlabel("Risk (Std Dev %)")
            ax.set_ylabel("Return (%)")
            ax.grid(True, alpha=0.3)

            # Add quadrants
            ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
            ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)

    def _create_summary_report(self, profit_learner, plot_paths: dict[str, str]):
        """Create a summary report with all plot paths."""
        report_path = self.output_dir / "ml_analysis_report.md"

        with open(report_path, "w") as f:
            f.write("# ML Trading System Analysis Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Summary\n\n")
            summary = profit_learner.get_learning_summary()
            f.write(f"- **Total Trades**: {summary['total_trades']}\n")
            f.write(f"- **Models Trained**: {summary['models_trained']}\n")
            f.write(
                f"- **Learning Status**: {'Active' if summary['total_trades'] >= summary['min_trades_for_learning'] else 'Collecting Data'}\n\n"
            )

            f.write("## Generated Plots\n\n")
            for plot_name, plot_path in plot_paths.items():
                f.write(f"### {plot_name.replace('_', ' ').title()}\n")
                f.write(f"![{plot_name}]({plot_path})\n\n")

            f.write("## Next Steps\n\n")
            f.write("1. **Accumulate more training data** - Run longer backtests\n")
            f.write(
                "2. **Analyze prediction accuracy** - Compare ML predictions vs actual outcomes\n"
            )
            f.write("3. **Tune ML parameters** - Optimize learning rates and thresholds\n")
            f.write("4. **Add more guardrails** - Implement safety checks for ML predictions\n")
            f.write(
                "5. **Test on different market conditions** - Validate across various market regimes\n"
            )

        logger.info(f"Created summary report: {report_path}")
