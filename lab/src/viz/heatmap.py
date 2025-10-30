"""Heatmap visualization for ablation impact matrices."""
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from pathlib import Path


def save_heatmap(pivot_df, out_path_png: Path):
    """Saves both an interactive Plotly HTML and a static PNG heatmap.

    Args:
        pivot_df: Pandas DataFrame (pivot table) with impact values
        out_path_png: Path to save PNG file
    """

    # 1. Plotly HTML
    try:
        fig = px.imshow(
            pivot_df.values,
            x=list(pivot_df.columns),
            y=[str(i) for i in pivot_df.index],  # Ensure index is string-like
            color_continuous_scale="RdBu_r",  # Red-Blue reversed (blue=positive)
            aspect="auto",
            labels=dict(
                x="Columns", y="Node", color=pivot_df.columns.name or "Value"
            ),
        )
        fig.update_layout(title="Ablation Impact Heatmap")

        out_html = out_path_png.with_suffix(".html")
        fig.write_html(str(out_html))
    except Exception as e:
        print(f"Warning: Plotly heatmap failed: {e}")

    # 2. Static PNG (matplotlib fallback)
    try:
        plt.figure(figsize=(10, max(6, len(pivot_df.index) // 4)))
        plt.imshow(pivot_df.values, aspect="auto", cmap="RdBu_r")
        plt.colorbar(label=pivot_df.columns.name or "value")
        plt.yticks(range(len(pivot_df.index)), pivot_df.index)

        # Handle simple or multi-level columns
        if isinstance(pivot_df.columns, pd.MultiIndex):
            col_labels = pivot_df.columns.get_level_values(0)
        else:
            col_labels = pivot_df.columns
        plt.xticks(range(len(col_labels)), col_labels, rotation=45)

        plt.title("Ablation Impact Heatmap (Static)")
        plt.tight_layout()
        plt.savefig(out_path_png, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Warning: Matplotlib heatmap failed: {e}")
