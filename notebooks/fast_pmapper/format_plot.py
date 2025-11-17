# Format a matplotlib plot

import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams.update({'font.size': 28})


def format_plot(ax: plt.Axes, grid_axis='both'):
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    # Make the titles bold
    ax.set_title(ax.get_title(), fontweight="bold")
    ax.set_xlabel(ax.get_xlabel(), fontweight="bold")
    ax.set_ylabel(ax.get_ylabel(), fontweight="bold")

    ax.tick_params(axis='both', which='major')#, labelsize=20)
    ax.tick_params(axis='both', which='minor')#, labelsize=20)

    legend = ax.legend(fontsize=24)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    ax.minorticks_on()
    ax.grid(axis=grid_axis, which='major', linestyle='-', linewidth='0.3', color='gray')
    ax.grid(axis=grid_axis, which='minor', linestyle='--', linewidth='0.1', color='lightgray')
