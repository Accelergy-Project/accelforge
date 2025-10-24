import matplotlib.pyplot as plt
import pandas as pd
from math import ceil
import seaborn as sns

from paths import DATA_DIR
from paths import PLOTS_DIR


def generate_plots(workload_name):
    dataframes = [
        f"timeloop_{workload_name}.csv",
        f"timeloop_{workload_name}_hint.csv",
        f"sunstone_{workload_name}.csv",
        f"zigzag_{workload_name}.csv",
        f"ffm_{workload_name}.csv",
    ]
    dataframes = [pd.read_csv(DATA_DIR / df) for df in dataframes]
    names = ["Timeloop", "Timeloop + Util. Hint", "SunStone", "ZigZag + Util. Hint", "FFM"]
    NAMES_SHORT = ["TL", "TL+H", "SS", "ZZ+H", "FFM"]
    colors = ['C0', 'C1', 'C2', 'C4', 'C3']

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(4, 3))

    ax.set_ylabel("EDP (Js)")
    for df, name, color in zip(dataframes, names, colors):
        df.plot(x="evaluations", y="edp", ax=ax, marker='.', label=name, color=color)
    ax.set_yscale("log")
    ax.ticklabel_format(style='sci', axis='x', scilimits=(-2, 2))
    ax.legend(ncols=ceil(len(dataframes)/2), bbox_to_anchor=(0.5,1.4), loc="upper center")

    ax.set_xlabel('Evaluated mappings')

    fig.savefig(PLOTS_DIR / f"{workload_name}_edp_vs_evals.pdf", dpi=400, bbox_inches='tight')
    fig.savefig(PLOTS_DIR / f"{workload_name}_edp_vs_evals.png", dpi=400, bbox_inches='tight')


    df_ffm = dataframes[-1]
    fastfusion_evals = df_ffm.iloc[df_ffm["edp"].argmin()]["evaluations"]
    fastfusion_edp = df_ffm["edp"].min()

    EDPS = []
    for df in dataframes:
        EDPS.append(get_edp_with_closest_evaluations(df, fastfusion_evals)/fastfusion_edp)

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.bar(NAMES_SHORT, EDPS, color=colors)
    ax.set_yscale("log")
    ax.set_xticklabels(NAMES_SHORT, rotation=15)
    ax.grid(axis='y')
    ax.set_ylabel("EDP normalized to FFM")
    sns.despine(ax=ax, left=True, bottom=True)

    fig.savefig(PLOTS_DIR / f"{workload_name}_edp.pdf", dpi=400, bbox_inches="tight")
    fig.savefig(PLOTS_DIR / f"{workload_name}_edp.png", dpi=400, bbox_inches="tight")


def get_rows_with_closest_evaluations(df, evaluations):
    """
    Returns two rows (x, y) where
    x is the row with closest `x["evaluations"] <= evaluations`
    and y is the row with closest `y["evaluations"] >= evaluations.`
    """
    less_than_df = df[df["evaluations"] <= evaluations]
    if len(less_than_df) == 0:
        x = None
    else:
        x = less_than_df.iloc[less_than_df["evaluations"].argmax()]
    greater_than_df = df[df["evaluations"] >= evaluations]
    if len(greater_than_df) == 0:
        y = None
    else:
        y = greater_than_df.iloc[greater_than_df["evaluations"].argmin()]
    return x, y


def get_edp_with_closest_evaluations(df, evaluations):
    x, y = get_rows_with_closest_evaluations(df, 43000)
    if x is None:
        return y["edp"]
    elif y is None:
        return x["edp"]
    else:
        # Interpolate linearly between x and y
        slope = (y["edp"] - x["edp"])/(y["evaluations"] - x["evaluations"])
        return x["edp"] + slope*(evaluations - x["evaluations"])


def get_rows_with_edp(df, edp):
    """
    Returns rows where `x["edp"]` is <= `edp` or None if no such row exists.
    """
    df_less_than = df[df["edp"] <= edp]
    if len(df_less_than) == 0:
        return None
    else:
        return df_less_than.iloc[df_less_than["edp"].argmax()]


generate_plots("mha")