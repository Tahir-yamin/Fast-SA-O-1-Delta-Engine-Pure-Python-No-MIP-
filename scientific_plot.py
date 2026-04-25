import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Set Publication Grade Styling (Inspired by R's ggplot2/Tidyverse)
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "axes.edgecolor": "#4a5568",
    "grid.color": "#e2e8f0"
})

try:
    # 1. Load and Process Real Data
    families = pd.read_csv('data/family_data.csv', index_col='family_id')
    submission = pd.read_csv('optimized_submission.csv', index_col='family_id')
    n_people = families['n_people'].to_dict()
    assigned_days = submission['assigned_day'].to_dict()

    occupancy = {d: 0 for d in range(1, 102)}
    for f_id, day in assigned_days.items():
        occupancy[day] += n_people[f_id]
        
    days = np.array(list(range(1, 101)))
    occ_values = np.array([occupancy[d] for d in days])
    
    # Calculate real-time Accounting Penalties for color mapping
    penalties = []
    for d in range(1, 101):
        n_d = occupancy[d]
        n_d_plus_1 = occupancy.get(d + 1, n_d)
        diff = abs(n_d - n_d_plus_1)
        penalties.append(((n_d - 125) / 400.0) * (n_d**(0.5 + diff / 50.0)))
    penalties = np.array(penalties)

    # 2. Create the Visual Asset
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)

    # Panel A: High-Definition Scatter Plot (Choice and Penalty Density)
    ax1 = fig.add_subplot(gs[0, :])
    scatter = ax1.scatter(days, occ_values, c=np.log10(penalties), s=penalties/5, 
                         cmap="magma", alpha=0.7, edgecolors="white", linewidth=0.5, label="Daily State")
    
    # Add a smoothing line (Savitzky-Golay or Lowess style)
    from scipy.interpolate import make_interp_spline
    X_Y_Spline = make_interp_spline(days, occ_values)
    X_Smooth = np.linspace(days.min(), days.max(), 500)
    Y_Smooth = X_Y_Spline(X_Smooth)
    ax1.plot(X_Smooth, Y_Smooth, color="#3182ce", linewidth=2.5, alpha=0.8, label="Occupancy Trend")

    ax1.axhline(125, color="#e53e3e", linestyle="--", alpha=0.5, label="Min Boundary")
    ax1.axhline(300, color="#e53e3e", linestyle="--", alpha=0.5, label="Max Boundary")
    
    ax1.set_title("Fig 1. Industrial Load Portfolio: Occupancy vs. Thermodynamic Stability Penalty", fontweight="bold")
    ax1.set_ylabel("Total Capacity ($N_d$)")
    cbar = plt.colorbar(scatter, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Logarithmic Penalty Intensity")
    ax1.legend(loc="upper right", frameon=True, fontsize=9)

    # Panel B: Penalty Distribution Histogram (R Style)
    ax2 = fig.add_subplot(gs[1, 0])
    sns.histplot(penalties, kde=True, color="#805ad5", ax=ax2, bins=25, alpha=0.6)
    ax2.set_xscale("log")
    ax2.set_title("Cost Distribution Density", fontsize=11)
    ax2.set_xlabel("Penalty Cost (Log Scale)")

    # Panel C: Delta-Smoothing Variance Heatmap (The "Smoothness" Trap)
    ax3 = fig.add_subplot(gs[1, 1])
    diffs = np.abs(np.diff(occ_values))
    # Reshape for a mini-heatmap sequence
    heatmap_data = diffs.reshape(9, 11) if len(diffs) == 99 else diffs[:90].reshape(9, 10)
    sns.heatmap(heatmap_data, cmap="YlOrRd", ax=ax3, cbar=False, annot=False)
    ax3.set_title("Inter-Day Variance Gradient ($|N_d - N_{d+1}|$)", fontsize=11)
    ax3.set_xticks([]); ax3.set_yticks([])

    plt.savefig('assets/scientific_data_viz.png', bbox_inches='tight', dpi=300)
    print("SUCCESS: High-fidelity scientific visualization generated.")

except Exception as e:
    print(f"FAILED to generate plot: {e}")
