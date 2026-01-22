import matplotlib.axes as mpax
import matplotlib.pyplot as plt
from fastfusion._accelerated_imports import np
from fastfusion._accelerated_imports import pd

from fastfusion.mapper.FFM._join_pmappings.pmapping_dataframe import PmappingDataframe


DATAFLOW_COLUMN = "dataflow"


def plot_ski_slope(
    data: pd.DataFrame,
    categorize_by_dataflow: bool = False,
    split_by_dataflow: bool = False,
    ax: mpax.Axes = None,
    **kwargs
):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if categorize_by_dataflow or split_by_dataflow:
        _add_dataflow_to_data(data)

    if not split_by_dataflow:
        data = PmappingDataframe(data).data

    separated_datas = []
    labels = []
    if categorize_by_dataflow or split_by_dataflow:
        for dataflow, sub_df in data.groupby(by=DATAFLOW_COLUMN):
            separated_datas.append(PmappingDataframe(sub_df).data)
            labels.append(dataflow)
    else:
        separated_datas.append(data)
        labels.append(None)

    for label, sub_df in zip(labels, separated_datas):
        ax.plot(
            *make_staircase(
                sub_df["Occupancy"].to_numpy(), sub_df["Offchip Accesses"].to_numpy()
            ),
            label=label,
            **kwargs
        )

    ax.set_xlabel("Capacity")
    ax.set_ylabel("Off-chip Accesses")
    ax.set_xscale("log")
    ax.set_yscale("log")

    return fig, ax


def make_staircase(x: np.array, y: np.array):
    """Make staircase from x, y coordinates."""
    sort_idx = np.argsort(x)
    x, y = x[sort_idx], y[sort_idx]

    shifted_x = x[1:]
    shifted_y = y[:-1]
    x = np.concat([x, shifted_x])
    y = np.concat([y, shifted_y])

    sort_idx = np.lexsort([x, -y])
    x, y = x[sort_idx], y[sort_idx]

    return x, y


def _add_dataflow_to_data(data: pd.DataFrame):
    data[DATAFLOW_COLUMN] = data["__Mappings"].apply(_dataflow_from_fullmapping)


def _dataflow_from_fullmapping(fullmapping: str):
    fullmapping = fullmapping.strip("[")
    fullmapping = fullmapping.strip("]")
    dataflow = []
    for term in fullmapping.split(","):
        if term[0] == "T":
            dataflow.append(int(term[1:].split(" size ")))
    return tuple(dataflow)


if __name__ == "__main__":
    print(make_staircase(np.array([1, 2, 3, 4]), np.array([4, 3, 2, 1])))
