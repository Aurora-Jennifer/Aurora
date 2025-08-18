"""
Feature Persistence + Continual Learning Module

This module provides advanced feature importance tracking, persistence analysis,
and continual learning capabilities for the ML trading system.

Features:
- Per-run feature importance logging
- Cross-run persistence analysis
- Warm-start utilities
- Curriculum learning
- Advanced alpha generation
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportance:
    """Feature importance record for a single run."""

    run_id: str
    timestamp: str
    feature_name: str
    importance: float
    coefficient: float
    shap_value: Optional[float] = None
    rank: Optional[int] = None
    regime: Optional[str] = None
    performance_metric: Optional[float] = None
    model_type: str = "ridge"
    alpha_generation_score: Optional[float] = None


@dataclass
class RunMetadata:
    """Metadata for a single ML run."""

    run_id: str
    timestamp: str
    ticker: str
    start_date: str
    end_date: str
    model_type: str
    seed: int
    total_trades: int
    avg_profit: float
    win_rate: float
    sharpe_ratio: float
    checkpoint_path: Optional[str] = None
    importance_path: Optional[str] = None


class FeaturePersistenceAnalyzer:
    """Analyzes feature importance persistence across runs."""

    def __init__(self, runs_dir: str = "runs"):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(exist_ok=True)
        self.checkpoints_dir = self.runs_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

        # File paths
        self.importance_csv = self.runs_dir / "feature_importance.csv"
        self.checkpoints_index = self.checkpoints_dir / "index.csv"
        self.persistence_analysis = self.runs_dir / "persistence_analysis.json"

        # Initialize files if they don't exist
        self._initialize_files()

    def _initialize_files(self):
        """Initialize CSV files with headers if they don't exist."""
        if not self.importance_csv.exists():
            df = pd.DataFrame(
                columns=[
                    "run_id",
                    "timestamp",
                    "feature_name",
                    "importance",
                    "coefficient",
                    "shap_value",
                    "rank",
                    "regime",
                    "performance_metric",
                    "model_type",
                    "alpha_generation_score",
                ]
            )
            df.to_csv(self.importance_csv, index=False)

        if not self.checkpoints_index.exists():
            df = pd.DataFrame(
                columns=[
                    "run_id",
                    "timestamp",
                    "ticker",
                    "start_date",
                    "end_date",
                    "model_type",
                    "seed",
                    "total_trades",
                    "avg_profit",
                    "win_rate",
                    "sharpe_ratio",
                    "checkpoint_path",
                    "importance_path",
                ]
            )
            df.to_csv(self.checkpoints_index, index=False)

    def log_feature_importance(
        self,
        run_id: str,
        feature_importances: Dict[str, float],
        coefficients: Dict[str, float],
        metadata: RunMetadata,
        shap_values: Optional[Dict[str, float]] = None,
        regime: Optional[str] = None,
    ) -> None:
        """Log feature importance for a single run."""
        try:
            # Calculate ranks
            sorted_features = sorted(
                feature_importances.items(), key=lambda x: abs(x[1]), reverse=True
            )
            ranks = {
                feature: rank + 1 for rank, (feature, _) in enumerate(sorted_features)
            }

            # Calculate alpha generation scores
            alpha_scores = self._calculate_alpha_generation_scores(
                feature_importances, coefficients, metadata
            )

            # Create records
            records = []
            for feature_name, importance in feature_importances.items():
                record = FeatureImportance(
                    run_id=run_id,
                    timestamp=metadata.timestamp,
                    feature_name=feature_name,
                    importance=importance,
                    coefficient=coefficients.get(feature_name, 0.0),
                    shap_value=shap_values.get(feature_name) if shap_values else None,
                    rank=ranks.get(feature_name),
                    regime=regime,
                    performance_metric=metadata.avg_profit,
                    model_type=metadata.model_type,
                    alpha_generation_score=alpha_scores.get(feature_name, 0.0),
                )
                records.append(asdict(record))

            # Append to CSV
            df_new = pd.DataFrame(records)
            df_new.to_csv(self.importance_csv, mode="a", header=False, index=False)

            # Save detailed importance data
            importance_file = self.runs_dir / f"{run_id}_importance.json"
            importance_data = {
                "run_id": run_id,
                "timestamp": metadata.timestamp,
                "feature_importances": feature_importances,
                "coefficients": coefficients,
                "shap_values": shap_values,
                "ranks": ranks,
                "alpha_scores": alpha_scores,
                "metadata": asdict(metadata),
            }
            with open(importance_file, "w") as f:
                json.dump(importance_data, f, indent=2)

            # Update checkpoints index
            self._update_checkpoints_index(metadata, importance_file)

            logger.info(
                f"Logged feature importance for run {run_id}: {len(records)} features"
            )

        except Exception as e:
            logger.error(f"Error logging feature importance: {e}")

    def _calculate_alpha_generation_scores(
        self,
        feature_importances: Dict[str, float],
        coefficients: Dict[str, float],
        metadata: RunMetadata,
    ) -> Dict[str, float]:
        """Calculate alpha generation potential scores for features."""
        alpha_scores = {}

        for feature_name in feature_importances.keys():
            importance = abs(feature_importances.get(feature_name, 0.0))
            coefficient = abs(coefficients.get(feature_name, 0.0))

            # Base score: combination of importance and coefficient
            base_score = (importance + coefficient) / 2

            # Performance multiplier: better performance = higher alpha potential
            performance_multiplier = 1.0 + (
                metadata.avg_profit * 10
            )  # Scale profit impact

            # Stability bonus: higher win rate = more stable alpha
            stability_bonus = 1.0 + (metadata.win_rate * 0.5)

            # Risk adjustment: higher Sharpe = better risk-adjusted alpha
            risk_adjustment = 1.0 + max(0, metadata.sharpe_ratio * 0.2)

            # Final alpha score
            alpha_score = (
                base_score * performance_multiplier * stability_bonus * risk_adjustment
            )
            alpha_scores[feature_name] = alpha_score

        return alpha_scores

    def _update_checkpoints_index(
        self, metadata: RunMetadata, importance_path: Path
    ) -> None:
        """Update the checkpoints index CSV."""
        try:
            # Read existing index
            if self.checkpoints_index.exists():
                df = pd.read_csv(self.checkpoints_index)
            else:
                df = pd.DataFrame()

            # Add new record
            new_record = {
                "run_id": metadata.run_id,
                "timestamp": metadata.timestamp,
                "ticker": metadata.ticker,
                "start_date": metadata.start_date,
                "end_date": metadata.end_date,
                "model_type": metadata.model_type,
                "seed": metadata.seed,
                "total_trades": metadata.total_trades,
                "avg_profit": metadata.avg_profit,
                "win_rate": metadata.win_rate,
                "sharpe_ratio": metadata.sharpe_ratio,
                "checkpoint_path": metadata.checkpoint_path,
                "importance_path": str(importance_path),
            }

            df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
            df.to_csv(self.checkpoints_index, index=False)

        except Exception as e:
            logger.error(f"Error updating checkpoints index: {e}")

    def analyze_persistence(self) -> Dict[str, Any]:
        """Analyze feature importance persistence across runs."""
        try:
            if not self.importance_csv.exists():
                return {"error": "No feature importance data found"}

            df = pd.read_csv(self.importance_csv)
            if df.empty:
                return {"error": "No feature importance data found"}

            # Group by feature
            feature_stats = {}
            for feature_name in df["feature_name"].unique():
                feature_data = df[df["feature_name"] == feature_name]

                # Calculate persistence metrics
                importance_mean = feature_data["importance"].mean()
                importance_std = feature_data["importance"].std()
                importance_cv = (
                    importance_std / abs(importance_mean)
                    if importance_mean != 0
                    else float("inf")
                )

                # Rank stability
                rank_mean = feature_data["rank"].mean()
                rank_std = feature_data["rank"].std()
                rank_stability = 1.0 / (1.0 + rank_std)  # Higher = more stable

                # Alpha generation potential
                alpha_mean = feature_data["alpha_generation_score"].mean()
                alpha_std = feature_data["alpha_generation_score"].std()

                # Performance correlation
                performance_corr = feature_data["importance"].corr(
                    feature_data["performance_metric"]
                )

                feature_stats[feature_name] = {
                    "importance_mean": importance_mean,
                    "importance_std": importance_std,
                    "importance_cv": importance_cv,
                    "rank_mean": rank_mean,
                    "rank_std": rank_std,
                    "rank_stability": rank_stability,
                    "alpha_mean": alpha_mean,
                    "alpha_std": alpha_std,
                    "performance_correlation": performance_corr,
                    "appearance_count": len(feature_data),
                    "total_runs": len(df["run_id"].unique()),
                }

            # Calculate overall persistence metrics
            persistence_metrics = {
                "total_runs": len(df["run_id"].unique()),
                "total_features": len(feature_stats),
                "avg_importance_stability": np.mean(
                    [stats["importance_cv"] for stats in feature_stats.values()]
                ),
                "avg_rank_stability": np.mean(
                    [stats["rank_stability"] for stats in feature_stats.values()]
                ),
                "top_alpha_features": sorted(
                    feature_stats.items(),
                    key=lambda x: x[1]["alpha_mean"],
                    reverse=True,
                )[:10],
                "most_stable_features": sorted(
                    feature_stats.items(),
                    key=lambda x: x[1]["rank_stability"],
                    reverse=True,
                )[:10],
                "feature_stats": feature_stats,
            }

            # Save analysis
            with open(self.persistence_analysis, "w") as f:
                json.dump(persistence_metrics, f, indent=2)

            return persistence_metrics

        except Exception as e:
            logger.error(f"Error analyzing persistence: {e}")
            return {"error": str(e)}

    def get_warm_start_data(
        self, feature_names: List[str], n_runs: int = 5
    ) -> Dict[str, Any]:
        """Get warm-start data for model training."""
        try:
            if not self.importance_csv.exists():
                return {"error": "No feature importance data found"}

            df = pd.read_csv(self.importance_csv)
            if df.empty:
                return {"error": "No feature importance data found"}

            # Get recent runs
            recent_runs = df["run_id"].unique()[-n_runs:]

            # Calculate feature priors
            feature_priors = {}
            for feature_name in feature_names:
                feature_data = df[
                    (df["feature_name"] == feature_name)
                    & (df["run_id"].isin(recent_runs))
                ]

                if not feature_data.empty:
                    # EMA of coefficients
                    ema_coefficient = (
                        feature_data["coefficient"].ewm(span=3).mean().iloc[-1]
                    )

                    # Average importance
                    avg_importance = feature_data["importance"].mean()

                    # Alpha generation potential
                    avg_alpha = feature_data["alpha_generation_score"].mean()

                    feature_priors[feature_name] = {
                        "ema_coefficient": ema_coefficient,
                        "avg_importance": avg_importance,
                        "alpha_potential": avg_alpha,
                        "confidence": len(feature_data) / n_runs,
                    }
                else:
                    feature_priors[feature_name] = {
                        "ema_coefficient": 0.0,
                        "avg_importance": 0.0,
                        "alpha_potential": 0.0,
                        "confidence": 0.0,
                    }

            # Get curriculum data
            curriculum_data = self._get_curriculum_data(df, recent_runs)

            return {
                "feature_priors": feature_priors,
                "curriculum_data": curriculum_data,
                "recent_runs": list(recent_runs),
                "total_features": len(feature_names),
            }

        except Exception as e:
            logger.error(f"Error getting warm start data: {e}")
            return {"error": str(e)}

    def _get_curriculum_data(
        self, df: pd.DataFrame, recent_runs: List[str]
    ) -> Dict[str, Any]:
        """Get curriculum learning data based on regime performance."""
        try:
            # Analyze performance by regime
            regime_performance = {}
            for run_id in recent_runs:
                run_data = df[df["run_id"] == run_id]
                if not run_data.empty:
                    regime = run_data["regime"].iloc[0]
                    performance = run_data["performance_metric"].iloc[0]

                    if regime not in regime_performance:
                        regime_performance[regime] = []
                    regime_performance[regime].append(performance)

            # Calculate curriculum weights
            curriculum_weights = {}
            for regime, performances in regime_performance.items():
                avg_performance = np.mean(performances)
                # Higher weight for underperforming regimes
                weight = max(0.1, 1.0 - avg_performance)
                curriculum_weights[regime] = weight

            return {
                "regime_performance": regime_performance,
                "curriculum_weights": curriculum_weights,
                "sampling_strategy": "underperforming_regimes",
            }

        except Exception as e:
            logger.error(f"Error getting curriculum data: {e}")
            return {}

    def generate_persistence_report(self) -> str:
        """Generate a comprehensive persistence analysis report."""
        try:
            persistence_data = self.analyze_persistence()

            if "error" in persistence_data:
                return f"Error: {persistence_data['error']}"

            report = []
            report.append("# Feature Persistence Analysis Report")
            report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")

            # Summary
            report.append("## Summary")
            report.append(f"- Total Runs: {persistence_data['total_runs']}")
            report.append(f"- Total Features: {persistence_data['total_features']}")
            report.append(
                f"- Avg Importance Stability: {persistence_data['avg_importance_stability']:.4f}"
            )
            report.append(
                f"- Avg Rank Stability: {persistence_data['avg_rank_stability']:.4f}"
            )
            report.append("")

            # Top Alpha Features
            report.append("## Top Alpha Generation Features")
            for feature_name, stats in persistence_data["top_alpha_features"]:
                report.append(
                    f"- **{feature_name}**: {stats['alpha_mean']:.4f} ± {stats['alpha_std']:.4f}"
                )
            report.append("")

            # Most Stable Features
            report.append("## Most Stable Features")
            for feature_name, stats in persistence_data["most_stable_features"]:
                report.append(
                    f"- **{feature_name}**: Stability {stats['rank_stability']:.4f}, Rank {stats['rank_mean']:.1f} ± {stats['rank_std']:.1f}"
                )
            report.append("")

            # Detailed Feature Analysis
            report.append("## Detailed Feature Analysis")
            for feature_name, stats in persistence_data["feature_stats"].items():
                report.append(f"### {feature_name}")
                report.append(
                    f"- Importance: {stats['importance_mean']:.4f} ± {stats['importance_std']:.4f}"
                )
                report.append(
                    f"- Rank: {stats['rank_mean']:.1f} ± {stats['rank_std']:.1f}"
                )
                report.append(f"- Alpha Potential: {stats['alpha_mean']:.4f}")
                report.append(
                    f"- Performance Correlation: {stats['performance_correlation']:.4f}"
                )
                report.append(
                    f"- Appearances: {stats['appearance_count']}/{stats['total_runs']}"
                )
                report.append("")

            return "\n".join(report)

        except Exception as e:
            logger.error(f"Error generating persistence report: {e}")
            return f"Error generating report: {e}"

    def create_persistence_plots(
        self, output_dir: str = "results/persistence"
    ) -> List[str]:
        """Create persistence analysis plots."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            if not self.importance_csv.exists():
                return ["No data available for plotting"]

            df = pd.read_csv(self.importance_csv)
            if df.empty:
                return ["No data available for plotting"]

            plots_created = []

            # 1. Feature Importance Over Time
            plt.figure(figsize=(12, 8))
            pivot_df = df.pivot(
                index="run_id", columns="feature_name", values="importance"
            )
            pivot_df.plot(kind="line", marker="o", alpha=0.7)
            plt.title("Feature Importance Persistence Over Time")
            plt.xlabel("Run ID")
            plt.ylabel("Feature Importance")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig(
                output_path / "feature_importance_persistence.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            plots_created.append("feature_importance_persistence.png")

            # 2. Rank Stability Heatmap
            plt.figure(figsize=(10, 8))
            rank_pivot = df.pivot(index="run_id", columns="feature_name", values="rank")
            sns.heatmap(
                rank_pivot, annot=True, cmap="RdYlBu_r", center=rank_pivot.mean().mean()
            )
            plt.title("Feature Rank Stability Heatmap")
            plt.xlabel("Feature Name")
            plt.ylabel("Run ID")
            plt.tight_layout()
            plt.savefig(
                output_path / "rank_stability_heatmap.png", dpi=300, bbox_inches="tight"
            )
            plt.close()
            plots_created.append("rank_stability_heatmap.png")

            # 3. Alpha Generation Potential
            plt.figure(figsize=(12, 6))
            alpha_data = (
                df.groupby("feature_name")["alpha_generation_score"]
                .agg(["mean", "std"])
                .sort_values("mean", ascending=False)
            )
            alpha_data.head(15).plot(kind="bar", y="mean", yerr="std", capsize=5)
            plt.title("Top 15 Alpha Generation Features")
            plt.xlabel("Feature Name")
            plt.ylabel("Alpha Generation Score")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(
                output_path / "alpha_generation_potential.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            plots_created.append("alpha_generation_potential.png")

            # 4. Performance vs Importance Correlation
            plt.figure(figsize=(10, 6))
            corr_data = (
                df.groupby("feature_name")
                .apply(lambda x: x["importance"].corr(x["performance_metric"]))
                .sort_values(ascending=False)
            )
            corr_data.head(15).plot(kind="bar")
            plt.title("Feature Importance vs Performance Correlation")
            plt.xlabel("Feature Name")
            plt.ylabel("Correlation Coefficient")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(
                output_path / "importance_performance_correlation.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            plots_created.append("importance_performance_correlation.png")

            return [f"results/persistence/{plot}" for plot in plots_created]

        except Exception as e:
            logger.error(f"Error creating persistence plots: {e}")
            return [f"Error creating plots: {e}"]


# Global analyzer instance
persistence_analyzer = FeaturePersistenceAnalyzer()
