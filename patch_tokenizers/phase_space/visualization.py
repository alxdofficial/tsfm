"""
Visualization utilities for phase space embeddings.

Provides functions to plot 2D and 3D phase space trajectories with time coloring.
"""

from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_phase_space_3d(
    embedded: np.ndarray,
    ax: plt.Axes,
    title: str = "Phase Space",
    color_by_time: bool = True,
    alpha: float = 0.7,
    linewidth: float = 1.5
) -> None:
    """
    Plot 3D phase space trajectory.

    Args:
        embedded: (N, 3) array of phase space coordinates
        ax: Matplotlib 3D axis
        title: Plot title
        color_by_time: If True, color trajectory by time progression
        alpha: Transparency
        linewidth: Line width for trajectory
    """
    if embedded.shape[0] == 0:
        ax.text(0.5, 0.5, 0.5, "No data", ha='center', va='center')
        ax.set_title(title)
        return

    if embedded.shape[1] < 3:
        ax.text(0.5, 0.5, 0.5, f"Dim={embedded.shape[1]} < 3", ha='center', va='center')
        ax.set_title(title)
        return

    x, y, z = embedded[:, 0], embedded[:, 1], embedded[:, 2]

    if color_by_time:
        # Color by time progression (viridis: dark blue → green → yellow)
        colors = plt.cm.viridis(np.linspace(0, 1, len(x)))
        for i in range(len(x) - 1):
            ax.plot(x[i:i+2], y[i:i+2], z[i:i+2],
                   color=colors[i], alpha=alpha, linewidth=linewidth)

        # Add start and end markers
        ax.scatter([x[0]], [y[0]], [z[0]], c='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter([x[-1]], [y[-1]], [z[-1]], c='red', s=100, marker='X', label='End', zorder=5)
    else:
        ax.plot(x, y, z, alpha=alpha, linewidth=linewidth)

    ax.set_xlabel('x(t)')
    ax.set_ylabel('x(t+τ)')
    ax.set_zlabel('x(t+2τ)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_phase_space_2d(
    embedded: np.ndarray,
    ax: plt.Axes,
    dims: Tuple[int, int] = (0, 1),
    title: str = "Phase Space 2D",
    color_by_time: bool = True,
    alpha: float = 0.7
) -> None:
    """
    Plot 2D projection of phase space.

    Args:
        embedded: (N, m) array of phase space coordinates
        ax: Matplotlib axis
        dims: Tuple of (x_dim, y_dim) to plot
        title: Plot title
        color_by_time: If True, color trajectory by time
        alpha: Transparency
    """
    if embedded.shape[0] == 0:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return

    x_dim, y_dim = dims
    if x_dim >= embedded.shape[1] or y_dim >= embedded.shape[1]:
        ax.text(0.5, 0.5, f"Invalid dims {dims}", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return

    x, y = embedded[:, x_dim], embedded[:, y_dim]

    if color_by_time:
        colors = plt.cm.viridis(np.linspace(0, 1, len(x)))
        ax.scatter(x, y, c=colors, alpha=alpha, s=10)
        ax.scatter([x[0]], [y[0]], c='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter([x[-1]], [y[-1]], c='red', s=100, marker='X', label='End', zorder=5)
    else:
        ax.plot(x, y, alpha=alpha, linewidth=1.5)
        ax.scatter(x, y, alpha=alpha*0.5, s=5)

    ax.set_xlabel(f'x(t+{x_dim}τ)')
    ax.set_ylabel(f'x(t+{y_dim}τ)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_comparison_grid(
    embeddings_grid: list,
    row_labels: list,
    col_labels: list,
    figsize: Optional[Tuple[float, float]] = None,
    suptitle: str = "Phase Space Comparison"
) -> plt.Figure:
    """
    Create a grid of 3D phase space plots for comparing multiple embeddings.

    Args:
        embeddings_grid: List[List[np.ndarray]] - embeddings_grid[row][col] is (N, 3)
        row_labels: List of labels for rows (e.g., activity names)
        col_labels: List of labels for columns (e.g., patch indices)
        figsize: Figure size (auto-computed if None)
        suptitle: Overall figure title

    Returns:
        fig: Matplotlib figure
    """
    n_rows = len(embeddings_grid)
    n_cols = len(embeddings_grid[0]) if embeddings_grid else 0

    if figsize is None:
        figsize = (6 * n_cols, 5 * n_rows)

    fig = plt.figure(figsize=figsize)

    for i, row_embeddings in enumerate(embeddings_grid):
        for j, embedded in enumerate(row_embeddings):
            ax = fig.add_subplot(n_rows, n_cols, i * n_cols + j + 1, projection='3d')

            if isinstance(embedded, np.ndarray) and embedded.shape[0] > 0:
                plot_phase_space_3d(
                    embedded,
                    ax,
                    title=f"{row_labels[i]}\n{col_labels[j]}",
                    color_by_time=True,
                    alpha=0.8
                )
            else:
                ax.text(0.5, 0.5, 0.5, "No data", ha='center', va='center')
                ax.set_title(f"{row_labels[i]}\n{col_labels[j]}")

    plt.suptitle(suptitle, fontsize=14, y=0.995)
    plt.tight_layout()

    return fig
