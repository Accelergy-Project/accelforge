import pydot


def _pydot_graph() -> pydot.Dot:
    graph = pydot.Dot(graph_type="graph", rankdir="TD", ranksep=0.2)
    graph.set_node_defaults(shape="box", fontname="Arial", fontsize="12")
    graph.set_edge_defaults(fontname="Arial", fontsize="10")
    return graph


# =============================================================================
# Color Map for Visualization
# =============================================================================


class ColorMap:

    def __init__(self, keys: list[str]):
        self.keys = keys
        self.color_list = self._make_color_map(len(keys))
        self.color_map = {key: self.color_list[i] for i, key in enumerate(keys)}

    def format_list(self, items: list[str]) -> str:
        result = ['<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0"><TR>']
        for i, item in enumerate(items):
            start = '<TD ALIGN="CENTER">'  # if i < len(items) - 1 else f'</TR><TR><TD ALIGN="CENTER" COLSPAN="100">'
            if item in self.color_map:
                start = f'<TD ALIGN="CENTER" BORDER="5" COLOR="{self.color_map[item]}">'
            end = "</TD>"
            result.append(f"{start}{item}{end}")
        result.append("</TR></TABLE>>")
        return "".join(result)

        # This makes a colored bar under the text
        # result = ['<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">']
        # # First row: text
        # result.append('<TR>')
        # for item in items:
        #     result.append(f'<TD ALIGN="CENTER" STYLE="margin:0;padding:0;">{item}</TD>')
        # result.append('</TR>')
        # # Second row: color bar (height 20, width 40, minimal spacing)
        # result.append('<TR>')
        # for item in items:
        #     if item in self.color_map:
        #         result.append(f'<TD BGCOLOR="{self.color_map[item]}" HEIGHT="10" WIDTH="15" FIXEDSIZE="TRUE" STYLE="margin:0;padding:0;"></TD>')
        #     else:
        #         result.append('<TD HEIGHT="20" WIDTH="40" FIXEDSIZE="TRUE" STYLE="margin:0;padding:0;"></TD>')
        # result.append('</TR>')
        # result.append('</TABLE>>')
        # return ''.join(result)

    def _make_color_map(self, n_colors: int) -> list[str]:
        if n_colors <= 0:
            return []

        # High contrast, distinguishable colors for borders
        base_colors = [
            "#FF0000",  # Red
            "#00FF00",  # Green
            "#0000FF",  # Blue
            "#FFFF00",  # Yellow
            "#FF00FF",  # Magenta
            "#00FFFF",  # Cyan
            "#FF8000",  # Orange
            "#8000FF",  # Purple
            "#008000",  # Dark Green
            "#800000",  # Dark Red
            "#000080",  # Dark Blue
            "#808000",  # Olive
        ]

        if n_colors <= len(base_colors):
            return base_colors[:n_colors]

        # For more colors, generate additional colors with maximum distinction
        colors = base_colors.copy()

        # Use evenly spaced hues for maximum distinction
        for i in range(len(base_colors), n_colors):
            # Evenly space hues around the color wheel
            hue = i / n_colors

            # Use high saturation and value for maximum contrast
            saturation = 1.0  # Full saturation
            value = 1.0  # Full value

            # Convert HSV to RGB
            h = hue * 6
            c = value * saturation
            x = c * (1 - abs(h % 2 - 1))
            m = value - c

            if h < 1:
                r, g, b = c, x, 0
            elif h < 2:
                r, g, b = x, c, 0
            elif h < 3:
                r, g, b = 0, c, x
            elif h < 4:
                r, g, b = 0, x, c
            elif h < 5:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x

            r = int((r + m) * 255)
            g = int((g + m) * 255)
            b = int((b + m) * 255)

            colors.append(f"#{r:02x}{g:02x}{b:02x}")

        return colors
