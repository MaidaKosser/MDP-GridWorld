"""
Visualization Module for GridWorld MDP
======================================
Renders value functions and policies on the grid.

Features:
- Heatmap visualization of state values
- Policy arrows showing optimal actions
- Clear marking of obstacles and terminals
"""

import numpy as np
import matplotlib.pyplot as plt

# Unicode arrow symbols for policy visualization
ARROW = {
    "U": "↑",
    "D": "↓",
    "L": "←",
    "R": "→"
}


def values_to_grid(mdp, V):
    """
    Convert value function array to 2D grid for visualization.
    
    Args:
        mdp: GridWorldMDP instance
        V: Value function (numpy array indexed by state)
        
    Returns:
        2D numpy array with values (NaN for obstacles)
    """
    grid = np.full((mdp.rows, mdp.cols), np.nan)
    
    for s in mdp.states:
        r, c = s
        idx = mdp.state_to_idx[s]
        grid[r, c] = float(V[idx])
    
    return grid


def plot_value_and_policy(mdp, V, pi, figsize=(4.4, 4.4), dpi=150):
    """
    Create comprehensive visualization of value function and policy.
    
    Displays:
    - Heatmap of state values
    - Policy arrows for each state
    - Obstacle markers
    - Terminal state labels
    
    Args:
        mdp: GridWorldMDP instance
        V: Value function
        pi: Policy (dict: state -> action)
        figsize: Figure size in inches
        dpi: Resolution
        
    Returns:
        matplotlib Figure object
    """
    val_grid = values_to_grid(mdp, V)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Create heatmap
    im = ax.imshow(val_grid, cmap="viridis", interpolation="nearest")

    # Add gridlines
    ax.set_xticks(np.arange(-0.5, mdp.cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mdp.rows, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Annotate each cell
    for r in range(mdp.rows):
        for c in range(mdp.cols):
            s = (r, c)
            
            # Mark obstacles
            if s in mdp.obstacles:
                ax.text(
                    c, r, "■",
                    ha="center", va="center",
                    color="white", fontsize=16, fontweight="bold"
                )
                continue
            
            # Mark goal terminal
            if s == mdp.goal:
                ax.text(
                    c, r, "GOAL",
                    ha="center", va="center",
                    color="green", fontsize=11, fontweight="bold"
                )
                continue
            
            # Mark negative terminal
            if s == mdp.neg_terminal:
                ax.text(
                    c, r, "NEG",
                    ha="center", va="center",
                    color="red", fontsize=11, fontweight="bold"
                )
                continue
            
            # Skip if no value (shouldn't happen for valid states)
            if np.isnan(val_grid[r, c]):
                continue
            
            # Display policy arrow
            a = pi.get(s, None)
            arrow = ARROW.get(a, "")
            ax.text(
                c, r - 0.12, arrow,
                ha="center", va="center",
                color="white", fontsize=18, fontweight="bold"
            )
            
            # Display value
            ax.text(
                c, r + 0.22, f"{val_grid[r, c]:.2f}",
                ha="center", va="center",
                color="white", fontsize=9
            )

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("State Value", fontsize=9)
    
    fig.tight_layout(pad=0.2)
    return fig