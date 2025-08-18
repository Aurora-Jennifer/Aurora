#!/usr/bin/env python3
"""
Feature Persistence Dashboard

This script provides a comprehensive dashboard for analyzing feature persistence,
rank stability, and alpha generation potential across multiple ML runs.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

from core.utils import setup_logging
from experiments.persistence import FeaturePersistenceAnalyzer

logger = setup_logging("logs/persistence_dashboard.log", logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Feature Persistence Dashboard")

    parser.add_argument(
        "--runs-dir", type=str, default="runs", help="Directory containing run data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/dashboard",
        help="Output directory for dashboard",
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Show interactive plots"
    )
    parser.add_argument(
        "--save-plots", action="store_true", default=True, help="Save dashboard plots"
    )

    return parser.parse_args()


def create_persistence_dashboard(args):
    """Create comprehensive persistence dashboard."""
    try:
        # Initialize analyzer
        persistence_analyzer = FeaturePersistenceAnalyzer(args.runs_dir)

        # Analyze persistence
        persistence_data = persistence_analyzer.analyze_persistence()

        if "error" in persistence_data:
            logger.error(f"Persistence analysis error: {persistence_data['error']}")
            return False

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create dashboard
        create_comprehensive_dashboard(persistence_data, output_dir, args)

        return True

    except Exception as e:
        logger.error(f"Error creating dashboard: {e}")
        return False


def create_comprehensive_dashboard(persistence_data, output_dir, args):
    """Create comprehensive dashboard with multiple panels."""

    # Set up the figure
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Panel 1: Feature Importance Persistence Over Time
    ax1 = fig.add_subplot(gs[0, :2])
    create_importance_persistence_plot(persistence_data, ax1)

    # Panel 2: Rank Stability Heatmap
    ax2 = fig.add_subplot(gs[0, 2])
    create_rank_stability_heatmap(persistence_data, ax2)

    # Panel 3: Alpha Generation Potential
    ax3 = fig.add_subplot(gs[1, 0])
    create_alpha_generation_plot(persistence_data, ax3)

    # Panel 4: Performance vs Importance Correlation
    ax4 = fig.add_subplot(gs[1, 1])
    create_performance_correlation_plot(persistence_data, ax4)

    # Panel 5: Feature Stability Metrics
    ax5 = fig.add_subplot(gs[1, 2])
    create_stability_metrics_plot(persistence_data, ax5)

    # Panel 6: Run Performance Summary
    ax6 = fig.add_subplot(gs[2, :])
    create_run_performance_summary(persistence_data, ax6)

    # Add title
    fig.suptitle(
        "Feature Persistence & Continual Learning Dashboard",
        fontsize=16,
        fontweight="bold",
    )

    # Save dashboard
    if args.save_plots:
        dashboard_file = output_dir / "persistence_dashboard.png"
        plt.savefig(dashboard_file, dpi=300, bbox_inches="tight")
        logger.info(f"Dashboard saved: {dashboard_file}")

    # Show interactive plot
    if args.interactive:
        plt.show()
    else:
        plt.close()

    # Create summary report
    create_dashboard_summary(persistence_data, output_dir)


def create_importance_persistence_plot(persistence_data, ax):
    """Create feature importance persistence plot."""
    try:
        # Load raw data
        analyzer = FeaturePersistenceAnalyzer()
        df = pd.read_csv(analyzer.importance_csv)

        if df.empty:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            ax.set_title("Feature Importance Persistence")
            return

        # Pivot data for plotting
        pivot_df = df.pivot(index="run_id", columns="feature_name", values="importance")

        # Plot top 10 features
        top_features = pivot_df.mean().nlargest(10).index
        pivot_df[top_features].plot(kind="line", marker="o", alpha=0.7, ax=ax)

        ax.set_title("Feature Importance Persistence Over Time", fontweight="bold")
        ax.set_xlabel("Run ID")
        ax.set_ylabel("Feature Importance")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    except Exception as e:
        logger.error(f"Error creating importance persistence plot: {e}")
        ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")


def create_rank_stability_heatmap(persistence_data, ax):
    """Create rank stability heatmap."""
    try:
        # Load raw data
        analyzer = FeaturePersistenceAnalyzer()
        df = pd.read_csv(analyzer.importance_csv)

        if df.empty:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            ax.set_title("Rank Stability Heatmap")
            return

        # Pivot data for heatmap
        rank_pivot = df.pivot(index="run_id", columns="feature_name", values="rank")

        # Show top 15 features
        top_features = rank_pivot.mean().nsmallest(15).index
        rank_pivot_top = rank_pivot[top_features]

        sns.heatmap(
            rank_pivot_top,
            annot=True,
            cmap="RdYlBu_r",
            center=rank_pivot_top.mean().mean(),
            ax=ax,
            cbar_kws={"label": "Rank"},
        )

        ax.set_title("Feature Rank Stability Heatmap", fontweight="bold")
        ax.set_xlabel("Feature Name")
        ax.set_ylabel("Run ID")

    except Exception as e:
        logger.error(f"Error creating rank stability heatmap: {e}")
        ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")


def create_alpha_generation_plot(persistence_data, ax):
    """Create alpha generation potential plot."""
    try:
        if not persistence_data.get("top_alpha_features"):
            ax.text(0.5, 0.5, "No alpha data available", ha="center", va="center")
            ax.set_title("Alpha Generation Potential")
            return

        # Extract data
        features = [item[0] for item in persistence_data["top_alpha_features"][:10]]
        alpha_means = [
            item[1]["alpha_mean"]
            for item in persistence_data["top_alpha_features"][:10]
        ]
        alpha_stds = [
            item[1]["alpha_std"] for item in persistence_data["top_alpha_features"][:10]
        ]

        # Create bar plot
        bars = ax.bar(
            range(len(features)),
            alpha_means,
            yerr=alpha_stds,
            capsize=5,
            alpha=0.7,
            color="skyblue",
            edgecolor="navy",
        )

        ax.set_title("Top 10 Alpha Generation Features", fontweight="bold")
        ax.set_xlabel("Feature Name")
        ax.set_ylabel("Alpha Generation Score")
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels(features, rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, mean_val in zip(bars, alpha_means):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.001,
                f"{mean_val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    except Exception as e:
        logger.error(f"Error creating alpha generation plot: {e}")
        ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")


def create_performance_correlation_plot(persistence_data, ax):
    """Create performance vs importance correlation plot."""
    try:
        # Load raw data
        analyzer = FeaturePersistenceAnalyzer()
        df = pd.read_csv(analyzer.importance_csv)

        if df.empty:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            ax.set_title("Performance Correlation")
            return

        # Calculate correlations
        correlations = (
            df.groupby("feature_name")
            .apply(lambda x: x["importance"].corr(x["performance_metric"]))
            .sort_values(ascending=False)
        )

        # Plot top 10 correlations
        top_corr = correlations.head(10)

        bars = ax.bar(
            range(len(top_corr)),
            top_corr.values,
            alpha=0.7,
            color="lightgreen",
            edgecolor="darkgreen",
        )

        ax.set_title("Feature Importance vs Performance Correlation", fontweight="bold")
        ax.set_xlabel("Feature Name")
        ax.set_ylabel("Correlation Coefficient")
        ax.set_xticks(range(len(top_corr)))
        ax.set_xticklabels(top_corr.index, rotation=45, ha="right")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.5)

        # Add value labels
        for bar, corr_val in zip(bars, top_corr.values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{corr_val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    except Exception as e:
        logger.error(f"Error creating performance correlation plot: {e}")
        ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")


def create_stability_metrics_plot(persistence_data, ax):
    """Create feature stability metrics plot."""
    try:
        if not persistence_data.get("feature_stats"):
            ax.text(0.5, 0.5, "No stability data available", ha="center", va="center")
            ax.set_title("Feature Stability Metrics")
            return

        # Extract stability metrics
        feature_stats = persistence_data["feature_stats"]
        features = list(feature_stats.keys())[:10]  # Top 10 features

        stability_scores = [feature_stats[f]["rank_stability"] for f in features]
        importance_cv = [feature_stats[f]["importance_cv"] for f in features]

        # Create scatter plot
        scatter = ax.scatter(
            importance_cv,
            stability_scores,
            alpha=0.7,
            s=100,
            c=range(len(features)),
            cmap="viridis",
        )

        ax.set_title("Feature Stability Metrics", fontweight="bold")
        ax.set_xlabel("Importance Coefficient of Variation (Lower = More Stable)")
        ax.set_ylabel("Rank Stability Score (Higher = More Stable)")
        ax.grid(True, alpha=0.3)

        # Add feature labels
        for i, feature in enumerate(features):
            ax.annotate(
                feature,
                (importance_cv[i], stability_scores[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Feature Index")

    except Exception as e:
        logger.error(f"Error creating stability metrics plot: {e}")
        ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")


def create_run_performance_summary(persistence_data, ax):
    """Create run performance summary."""
    try:
        # Load run metadata
        analyzer = FeaturePersistenceAnalyzer()
        if analyzer.checkpoints_index.exists():
            df = pd.read_csv(analyzer.checkpoints_index)

            if not df.empty:
                # Plot performance metrics over time
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp")

                # Create subplots for different metrics
                ax2 = ax.twinx()

                # Plot total trades
                line1 = ax.plot(
                    df["timestamp"],
                    df["total_trades"],
                    "o-",
                    color="blue",
                    label="Total Trades",
                    linewidth=2,
                )
                ax.set_ylabel("Total Trades", color="blue")
                ax.tick_params(axis="y", labelcolor="blue")

                # Plot average profit
                line2 = ax2.plot(
                    df["timestamp"],
                    df["avg_profit"] * 100,
                    "s-",
                    color="red",
                    label="Avg Profit (%)",
                    linewidth=2,
                )
                ax2.set_ylabel("Average Profit (%)", color="red")
                ax2.tick_params(axis="y", labelcolor="red")

                ax.set_title("Run Performance Summary Over Time", fontweight="bold")
                ax.set_xlabel("Timestamp")
                ax.grid(True, alpha=0.3)

                # Add legend
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax.legend(lines, labels, loc="upper left")

                # Rotate x-axis labels
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No run performance data available",
                    ha="center",
                    va="center",
                )
        else:
            ax.text(
                0.5, 0.5, "No run performance data available", ha="center", va="center"
            )

    except Exception as e:
        logger.error(f"Error creating run performance summary: {e}")
        ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")


def create_dashboard_summary(persistence_data, output_dir):
    """Create dashboard summary report."""
    try:
        summary = []
        summary.append("# Feature Persistence Dashboard Summary")
        summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("")

        # Overall metrics
        summary.append("## Overall Metrics")
        summary.append(f"- **Total Runs**: {persistence_data.get('total_runs', 0)}")
        summary.append(
            f"- **Total Features**: {persistence_data.get('total_features', 0)}"
        )
        summary.append(
            f"- **Avg Importance Stability**: {persistence_data.get('avg_importance_stability', 0):.4f}"
        )
        summary.append(
            f"- **Avg Rank Stability**: {persistence_data.get('avg_rank_stability', 0):.4f}"
        )
        summary.append("")

        # Top alpha features
        if persistence_data.get("top_alpha_features"):
            summary.append("## Top Alpha Generation Features")
            for feature_name, stats in persistence_data["top_alpha_features"][:10]:
                summary.append(
                    f"- **{feature_name}**: {stats['alpha_mean']:.4f} ± {stats['alpha_std']:.4f}"
                )
            summary.append("")

        # Most stable features
        if persistence_data.get("most_stable_features"):
            summary.append("## Most Stable Features")
            for feature_name, stats in persistence_data["most_stable_features"][:10]:
                summary.append(
                    f"- **{feature_name}**: Stability {stats['rank_stability']:.4f}"
                )
            summary.append("")

        # Recommendations
        summary.append("## Recommendations")
        summary.append("### For Alpha Generation:")
        if persistence_data.get("top_alpha_features"):
            top_feature = persistence_data["top_alpha_features"][0]
            summary.append(
                f"- Focus on **{top_feature[0]}** for highest alpha potential"
            )

        summary.append("### For Model Stability:")
        if persistence_data.get("most_stable_features"):
            stable_feature = persistence_data["most_stable_features"][0]
            summary.append(f"- Use **{stable_feature[0]}** as a reliable feature")

        summary.append("### For Continual Learning:")
        summary.append("- Monitor feature importance changes over time")
        summary.append("- Use warm-start for new models")
        summary.append("- Apply curriculum learning for underperforming regimes")

        # Save summary
        summary_file = output_dir / "dashboard_summary.md"
        with open(summary_file, "w") as f:
            f.write("\n".join(summary))

        logger.info(f"Dashboard summary saved: {summary_file}")

    except Exception as e:
        logger.error(f"Error creating dashboard summary: {e}")


def main():
    """Main function."""
    args = parse_args()

    print("📊 Feature Persistence Dashboard")
    print("=" * 50)

    success = create_persistence_dashboard(args)

    if success:
        print("\n✅ Dashboard created successfully!")
        print(f"Results saved to: {args.output_dir}")
    else:
        print("\n❌ Dashboard creation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
