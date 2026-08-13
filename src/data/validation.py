import os
import json
import logging
from typing import Dict, Any
import numpy as np
import pandas as pd
try:
    import matplotlib
    matplotlib.use('Agg')  # Headless non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

from src.utils.logging_config import setup_logger

logger = setup_logger("dataset_validator")

class DatasetValidator:
    """
    Validates and profiles raw and interim datasets.
    Checks schema, missing values, infinite values, duplicate rows, and class balance.
    Saves validation profiles and class distribution figures.
    """
    def __init__(self, target_column: str = "Attack_type", label_column: str = "Attack_label"):
        self.target_column = target_column
        self.label_column = label_column

    def validate_and_profile(self, df: pd.DataFrame, results_dir: str = "results") -> Dict[str, Any]:
        """
        Executes full dataset audit and profiling.
        Saves dataset_profile.json, dataset_profile.csv, and distribution plots.
        """
        logger.info("Executing dataset validation and profiling...")
        num_rows, num_cols = df.shape

        # 1. Missing Values Analysis
        missing_series = df.isna().sum()
        missing_total = int(missing_series.sum())
        cols_with_missing = {col: int(count) for col, count in missing_series[missing_series > 0].items()}

        # 2. Duplicate Rows Analysis
        duplicates_count = int(df.duplicated().sum())

        # 3. Infinite Values Analysis
        numeric_df = df.select_dtypes(include=[np.number])
        inf_counts = np.isinf(numeric_df).sum()
        inf_total = int(inf_counts.sum())
        cols_with_inf = {col: int(count) for col, count in inf_counts[inf_counts > 0].items()}

        # 4. Label & Class Distribution Analysis
        if self.target_column in df.columns:
            class_counts = df[self.target_column].value_counts().to_dict()
            class_percentages = (df[self.target_column].value_counts(normalize=True) * 100).to_dict()
        else:
            class_counts = {}
            class_percentages = {}

        # 5. Schema Breakdown
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

        profile_data = {
            "dataset_summary": {
                "num_samples": num_rows,
                "num_features": num_cols,
                "missing_values_total": missing_total,
                "duplicate_rows_total": duplicates_count,
                "infinite_values_total": inf_total
            },
            "missing_values_per_column": cols_with_missing,
            "infinite_values_per_column": cols_with_inf,
            "class_distribution_counts": class_counts,
            "class_distribution_percentages": class_percentages,
            "schema_data_types": dtypes
        }

        # Save profile JSON & CSV
        os.makedirs(results_dir, exist_ok=True)
        json_path = os.path.join(results_dir, "dataset_profile.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(profile_data, f, indent=2)
        logger.info(f"Saved dataset profile JSON to: {json_path}")

        # Save profile CSV
        csv_path = os.path.join(results_dir, "dataset_profile.csv")
        profile_df = pd.DataFrame(list(class_counts.items()), columns=["Attack_Category", "Sample_Count"])
        profile_df["Percentage"] = profile_df["Attack_Category"].map(class_percentages)
        profile_df.to_csv(csv_path, index=False)
        logger.info(f"Saved dataset profile CSV to: {csv_path}")

        # Generate Class Distribution Figure
        fig_dir = os.path.join(results_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        self.plot_class_distribution(profile_df, os.path.join(fig_dir, "dataset_class_distribution.png"))

        return profile_data

    def plot_class_distribution(self, profile_df: pd.DataFrame, output_path: str):
        """Generates and saves a clean, styled bar graph of attack class distribution."""
        if not HAS_PLOTTING:
            logger.warning("Matplotlib/Seaborn not installed. Skipping class distribution plot generation.")
            return

        plt.figure(figsize=(12, 6))
        sns.set_theme(style="whitegrid")
        palette = sns.color_palette("viridis", len(profile_df))
        
        ax = sns.barplot(x="Sample_Count", y="Attack_Category", data=profile_df, hue="Attack_Category", palette=palette, legend=False)
        plt.title("Edge-IIoTset Attack Class Sample Distribution", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Number of Samples", fontsize=12)
        plt.ylabel("Attack Category", fontsize=12)

        # Annotate percentages
        for p in ax.patches:
            width = p.get_width()
            pct = (width / profile_df["Sample_Count"].sum()) * 100
            ax.annotate(f"{width:,.0f} ({pct:.1f}%)",
                        (width + (profile_df["Sample_Count"].max() * 0.01), p.get_y() + p.get_height() / 2.),
                        ha='left', va='center', fontsize=9, color='black')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved dataset class distribution plot to: {output_path}")
