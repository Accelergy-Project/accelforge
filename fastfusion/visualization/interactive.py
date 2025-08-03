import os
from typing import Optional, Union
import plotly
from IPython.display import SVG, display
import plotly.graph_objs as go
from ipywidgets import Output, VBox, HBox
from fastfusion.accelerated_imports import pd

from fastfusion.accelerated_imports import pd
from fastfusion.mapper.FFM.visualization import make_mapping
from fastfusion.frontend.mapping import Mapping

def mapping2svg(mapping: pd.Series, einsum_names: list[str], rank_variable_bounds: Optional[dict[str, dict[str, int]]] = None):
    mapping: Mapping = make_mapping(mapping, einsum_names, rank_variable_bounds)
    render = mapping.render()
    return SVG(render)


def diplay_mappings_on_fig(
    fig: plotly.graph_objs.FigureWidget,
    data: dict[str, pd.DataFrame],
    mapping_svg: bool,
    einsum_names: list[str],
    rank_variable_bounds: Optional[dict[str, dict[str, int]]] = None,
):
    # fig = go.FigureWidget(fig)
    out = Output()
    DUAL_OUT = True
    if mapping_svg:
        assert DUAL_OUT
    DUAL_OUT = False

    counter = 0

    @out.capture()
    def display_mapping(trace, points, selector):
        if not points.point_inds:
            return
        out.clear_output()
        d = data[trace.name]
        index = points.point_inds[0]
        display(mapping2svg(d.iloc[index], einsum_names, rank_variable_bounds))
        # backing_tensors = set(
        #     t for tn in d.iloc[index][MAPPING_COLUMN].values() for t in tn.tensors
        # )
        # backing_tensors = TensorReservation.get_backing_tensors(backing_tensors)
        # for t in sorted(backing_tensors):
        #     print(f"{t.__repr__()},")
        # for v in d.iloc[index][MAPPING_COLUMN].values():
        #     print(v)

    out2 = Output()

    @out2.capture()
    def display_mapping_2(trace, points, selector):
        if not points.point_inds:
            return
        out2.clear_output()
        d = data[trace.name]
        index = points.point_inds[0]
        display(mapping2svg(d.iloc[index]))
        if mapping_svg:
            os.makedirs("plots", exist_ok=True)
            svg = mapping2svg(d.iloc[index])
            with open(f"plots/{trace.name}.svg", "w") as f:
                f.write(svg.data)
        # backing_tensors = set(
        #     t for tn in d.iloc[index][MAPPING_COLUMN].values() for t in tn.tensors
        # )
        # backing_tensors = TensorReservation.get_backing_tensors(backing_tensors)
        # for t in sorted(backing_tensors):
        #     print(f"{t.__repr__()},")
        # for v in d.iloc[index][MAPPING_COLUMN].values():
        #     print(v)

    for i in fig.data:
        i.on_hover(display_mapping)
        i.on_click(display_mapping_2)
    if not DUAL_OUT:
        return VBox([fig, out])
    out.layout.width = "50%"
    out2.layout.width = "50%"
    return VBox([fig, HBox([out, out2])])


def plotly_show(
    data: Union[pd.DataFrame, dict[str, pd.DataFrame]],
    x: str,
    y: str,
    category: Optional[str] = None,
    title: Optional[str] = None,
    show_mapping: Optional[bool] = True,
    logscales: bool = False,
    mapping_svg: bool = False,
    einsum_names: Optional[list[str]] = None,
    rank_variable_bounds: Optional[dict[str, dict[str, int]]] = None,
):
    fig = go.FigureWidget()
    markers = [
        "circle",
        "square",
        "diamond",
        "cross",
        "x",
        "triangle-up",
        "triangle-down",
        "triangle-left",
        "triangle-right",
    ]
    if isinstance(data, dict):
        for i, (k, v) in enumerate(data.items()):
            v.sort_values(by=[x, y], inplace=True)
            fig.add_scatter(
                x=v[x],
                y=v[y],
                name=k,
                line={"shape": "hv"},
                mode="markers+lines",
                marker={"symbol": markers[i % len(markers)]},
            )
    else:
        data.sort_values(by=[x, y], inplace=True)
        fig.add_scatter(
            x=data[x],
            y=data[y],
            name="",
            line={"shape": "hv"},
            mode="markers+lines",
            marker={"symbol": markers[0]},
        )
        data = {"": data}
    if title is not None:
        fig.update_layout(title=title)
    if logscales:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
    fig.update_xaxes(title_text=x)
    fig.update_yaxes(title_text=y)
    fig.update_layout(showlegend=True)
    # fig = px.scatter(data, x=x, y=y, color=category, title=title, log_x=logscales, log_y=logscales)
    if show_mapping:
        assert einsum_names is not None, (
            f"einsum_names must be provided if show_mapping is True"
        )
        return diplay_mappings_on_fig(fig, data, mapping_svg, einsum_names, rank_variable_bounds)
    return fig
