#!/usr/bin/env python3
"""
Figure 1: Path-Patch DAG
Visualizes the causal mediation structure from H6 battery results.
Shows that 67% of head 0:2's effect travels through layer-11 residual stream.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set publication-quality defaults
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.dpi'] = 300

# Data from Table 3 (H6 Path Patching Results)
paths = {
    'Layer 1': 0.18,
    'Layer 2': 0.20,
    'Layer 11': 0.67,  # Primary mediation path
    'Layer 23': 0.08,
}

# Figure setup
fig, ax = plt.subplots(figsize=(3.5, 4.0))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Colors
primary_color = '#2E86AB'  # Teal blue for layer-11 path
secondary_color = '#A9A9A9'  # Gray for other paths
source_color = '#F77F00'  # Orange for source
output_color = '#06A77D'  # Green for output

# Node positions
source_pos = (5, 10.5)
layer1_pos = (2, 7.5)
layer2_pos = (3.5, 6.0)
layer11_pos = (6.5, 4.5)  # Primary path - rightmost
layer23_pos = (8, 3.0)
output_pos = (5, 0.5)

# Helper function to draw nodes
def draw_node(ax, pos, label, color, highlight=False):
    """Draw a node with optional highlight box"""
    box_width = 1.8
    box_height = 0.6

    if highlight:
        # Draw highlight box behind
        highlight_box = FancyBboxPatch(
            (pos[0] - box_width/2 - 0.15, pos[1] - box_height/2 - 0.15),
            box_width + 0.3, box_height + 0.3,
            boxstyle="round,pad=0.05",
            edgecolor=primary_color,
            facecolor='none',
            linewidth=2.5,
            zorder=1
        )
        ax.add_patch(highlight_box)

    # Main box
    box = FancyBboxPatch(
        (pos[0] - box_width/2, pos[1] - box_height/2),
        box_width, box_height,
        boxstyle="round,pad=0.05",
        edgecolor='black',
        facecolor=color,
        linewidth=1.2,
        zorder=2
    )
    ax.add_patch(box)

    # Label
    ax.text(pos[0], pos[1], label,
            ha='center', va='center',
            fontsize=9, fontweight='bold',
            zorder=3)

# Helper function to draw edges with percentage labels
def draw_edge(ax, start, end, percentage, primary=False):
    """Draw an edge with arrow and percentage label"""
    if primary:
        color = primary_color
        linewidth = 3.5
        alpha = 1.0
        linestyle = '-'
    else:
        color = secondary_color
        linewidth = 1.5
        alpha = 0.6
        linestyle = '--'

    # Draw arrow
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='->,head_width=0.3,head_length=0.3',
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        linestyle=linestyle,
        zorder=1
    )
    ax.add_patch(arrow)

    # Calculate midpoint for label
    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2

    # Offset label slightly to avoid arrow
    offset_x = 0.3 if start[0] < end[0] else -0.3

    # Percentage label
    label_text = f'{int(percentage * 100)}%'
    bbox_props = dict(
        boxstyle='round,pad=0.2',
        facecolor='white',
        edgecolor=color if primary else secondary_color,
        linewidth=1.5 if primary else 1.0,
        alpha=0.95
    )

    ax.text(mid_x + offset_x, mid_y, label_text,
            ha='center', va='center',
            fontsize=8,
            fontweight='bold' if primary else 'normal',
            bbox=bbox_props,
            zorder=4)

# Draw nodes
draw_node(ax, source_pos, 'Head 0:2\n(Layer 0)', source_color)
draw_node(ax, layer1_pos, 'Residual\nLayer 1', 'white')
draw_node(ax, layer2_pos, 'Residual\nLayer 2', 'white')
draw_node(ax, layer11_pos, 'Residual\nLayer 11', 'white', highlight=True)  # Highlight primary path
draw_node(ax, layer23_pos, 'Residual\nLayer 23', 'white')
draw_node(ax, output_pos, 'Factuality\nChange', output_color)

# Draw edges from source to intermediate layers
draw_edge(ax, source_pos, layer1_pos, paths['Layer 1'], primary=False)
draw_edge(ax, source_pos, layer2_pos, paths['Layer 2'], primary=False)
draw_edge(ax, source_pos, layer11_pos, paths['Layer 11'], primary=True)  # PRIMARY
draw_edge(ax, source_pos, layer23_pos, paths['Layer 23'], primary=False)

# Draw edges from intermediate layers to output
draw_edge(ax, layer1_pos, output_pos, paths['Layer 1'], primary=False)
draw_edge(ax, layer2_pos, output_pos, paths['Layer 2'], primary=False)
draw_edge(ax, layer11_pos, output_pos, paths['Layer 11'], primary=True)  # PRIMARY
draw_edge(ax, layer23_pos, output_pos, paths['Layer 23'], primary=False)

# Add title annotation
ax.text(5, 11.5, 'H6 Battery: Path Patching Mediation',
        ha='center', va='center',
        fontsize=11, fontweight='bold')

# Add primary path annotation
ax.text(8.5, 4.5, 'Primary\nMediation\nPath',
        ha='left', va='center',
        fontsize=8,
        color=primary_color,
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor=primary_color, linewidth=1.5))

# Add legend explaining line styles
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=primary_color, linewidth=3, label='Primary path (67%)'),
    Line2D([0], [0], color=secondary_color, linewidth=1.5, linestyle='--',
           label='Alternative paths (8-20%)', alpha=0.6)
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=7, framealpha=0.95)

plt.tight_layout()

# Save figure
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.join(script_dir, '..', '..')
figures_dir = os.path.join(repo_root, 'paper', 'figures')
os.makedirs(figures_dir, exist_ok=True)

output_path = os.path.join(figures_dir, 'path_patch_dag.pdf')
plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
print(f"✓ Figure 1 saved to {output_path}")

# Also save PNG for preview
output_path_png = os.path.join(figures_dir, 'path_patch_dag.png')
plt.savefig(output_path_png, format='png', bbox_inches='tight', dpi=300)
print(f"✓ Preview saved to {output_path_png}")

# plt.show()  # Comment out for non-interactive environments
