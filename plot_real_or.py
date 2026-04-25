import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    "font.family": "serif",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300
})

try:
    # Load Real Data
    families = pd.read_csv('data/family_data.csv', index_col='family_id')
    submission = pd.read_csv('optimized_submission.csv', index_col='family_id')
    n_people = families['n_people'].to_dict()
    assigned_days = submission['assigned_day'].to_dict()

    # Calculate actual occupancy per day
    occupancy = {d: 0 for d in range(1, 101)}
    for f_id, day in assigned_days.items():
        occupancy[day] += n_people[f_id]
        
    days = list(range(1, 101))
    occ_values = [occupancy[d] for d in days]
    
    # Calculate daily accounting penalty for visualization
    accounting_penalties = []
    for d in range(1, 101):
        n_d = occupancy[d]
        n_d_plus_1 = occupancy.get(d + 1, n_d) # Day 101 wraps to Day 100 for exact formula proxy
        diff = abs(n_d - n_d_plus_1)
        # Mathematical formula: (N_d - 125) / 400 * N_d ^ (0.5 + diff / 50)
        penalty = ((n_d - 125) / 400.0) * (n_d**(0.5 + diff / 50.0))
        accounting_penalties.append(penalty)

    # Make Subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)

    # Plot 1: Exact Occupancy Constrained Profile
    ax1.bar(days, occ_values, color='#2c5282', width=1.0, alpha=0.8, edgecolor='#1a365d', linewidth=0.5)
    ax1.axhline(y=125, color='#c53030', linestyle='--', linewidth=1.5, label='Min Capacity bound (125)')
    ax1.axhline(y=300, color='#822727', linestyle='--', linewidth=1.5, label='Max Capacity bound (300)')
    ax1.set_ylabel('Total Daily Occupancy ($N_d$)')
    ax1.set_title('Fig 1: Verified Occupancy Profile vs Capacity Constraint Boundaries')
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', linestyle=':', alpha=0.7)
    
    # Emphasize smoothing variance
    ax1.plot(days, occ_values, color='#ecc94b', linewidth=1.0, alpha=0.9, marker='.', markersize=3, label='Occupancy Curve')

    # Plot 2: Non-Linear Accounting Penalty Matrix
    ax2.plot(days, accounting_penalties, color='#e53e3e', linewidth=1.5, marker='x', markersize=4)
    ax2.fill_between(days, accounting_penalties, color='#feb2b2', alpha=0.3)
    ax2.set_xlabel('Scheduling Day Index ($d$)')
    ax2.set_ylabel('Accounting\nPenalty (Cost)')
    ax2.set_yscale('log') # Log scale because accounting variance is brutal
    ax2.set_title('Non-Linear Variance Exponentiation Topology Matrix ($A$)')
    ax2.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig('assets/optimization_landscape.png', bbox_inches='tight')
    print("Successfully generated real empirical data plot covering the fake generated one.")

except Exception as e:
    print(f"Error plotting real data: {e}")
