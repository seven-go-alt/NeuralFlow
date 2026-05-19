from __future__ import annotations


def _escape(text: str) -> str:
    """Escape text for safe embedding in SVG/HTML."""
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def bar_chart(
    labels: list[str],
    values: list[float],
    title: str = "",
    ylabel: str = "",
    width: int = 600,
    height: int = 300,
) -> str:
    """Generate an inline SVG bar chart."""
    if not labels or not values:
        return _empty_chart(width, height, "No data")

    n = len(labels)
    margin = {"top": 40, "right": 20, "bottom": 60, "left": 60}
    chart_w = width - margin["left"] - margin["right"]
    chart_h = height - margin["top"] - margin["bottom"]

    max_val = max(values) if values else 1
    if max_val == 0:
        max_val = 1

    bar_width = chart_w / n * 0.6
    gap = chart_w / n * 0.4
    bars: list[str] = []
    labels_el: list[str] = []

    for i, (label, val) in enumerate(zip(labels, values, strict=True)):
        bar_h = (val / max_val) * chart_h
        x = margin["left"] + i * (bar_width + gap) + gap / 2
        y = margin["top"] + chart_h - bar_h
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_h:.1f}" '
            f'fill="#0f3460" rx="2"><title>{val:.2f}</title></rect>'
        )
        labels_el.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{height - margin["bottom"] + 15}" '
            f'text-anchor="middle" font-size="11" fill="#666" transform="rotate(-30,{x + bar_width / 2:.1f},{height - margin["bottom"] + 15})">'
            f"{_escape(label)}</text>"
        )

    # Y-axis gridlines
    gridlines = ""
    for i in range(5):
        y = margin["top"] + chart_h - (chart_h * i / 4)
        gridlines += f'<line x1="{margin["left"]}" y1="{y:.1f}" x2="{width - margin["right"]}" y2="{y:.1f}" stroke="#eee" stroke-width="1"/>'
        gridlines += f'<text x="{margin["left"] - 5}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#999">{max_val * i / 4:.1f}</text>'

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
{gridlines}
{bars}
<text x="{width / 2}" y="20" text-anchor="middle" font-size="14" font-weight="600" fill="#333">{_escape(title)}</text>
<text x="15" y="{height / 2}" text-anchor="middle" font-size="12" fill="#666" transform="rotate(-90,15,{height / 2})">{_escape(ylabel)}</text>
{labels_el}
</svg>"""


def multi_bar_chart(
    labels: list[str],
    value_sets: list[list[float]],
    series_labels: list[str] | None = None,
    title: str = "",
    ylabel: str = "",
    width: int = 600,
    height: int = 300,
) -> str:
    """Generate an inline SVG grouped bar chart."""
    if not labels or not value_sets:
        return _empty_chart(width, height, "No data")

    n = len(labels)
    m = len(value_sets)
    margin = {"top": 40, "right": 20, "bottom": 60, "left": 60}
    chart_w = width - margin["left"] - margin["right"]
    chart_h = height - margin["top"] - margin["bottom"]

    max_val = max(max(vs) for vs in value_sets) if value_sets else 1
    if max_val == 0:
        max_val = 1

    colors = ["#0f3460", "#e94560", "#533483", "#16213e"]
    group_w = chart_w / n
    bar_w = group_w / m * 0.7
    gap = group_w * 0.15

    bars: list[str] = []
    for i, _label in enumerate(labels):
        for j in range(m):
            val = value_sets[j][i] if i < len(value_sets[j]) else 0
            bar_h = (val / max_val) * chart_h
            x = margin["left"] + i * group_w + gap + j * bar_w
            y = margin["top"] + chart_h - bar_h
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" '
                f'fill="{colors[j % len(colors)]}" rx="2"><title>{val:.2f}</title></rect>'
            )

    gridlines = ""
    for i in range(5):
        y = margin["top"] + chart_h - (chart_h * i / 4)
        gridlines += f'<line x1="{margin["left"]}" y1="{y:.1f}" x2="{width - margin["right"]}" y2="{y:.1f}" stroke="#eee" stroke-width="1"/>'
        gridlines += f'<text x="{margin["left"] - 5}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#999">{max_val * i / 4:.1f}</text>'

    # Legend
    legend = ""
    if series_labels:
        legend_x = width - margin["right"] - 100
        legend_y = margin["top"] + 5
        for j, sl in enumerate(series_labels):
            ly = legend_y + j * 18
            legend += f'<rect x="{legend_x}" y="{ly}" width="12" height="12" fill="{colors[j % len(colors)]}" rx="2"/>'
            legend += (
                f'<text x="{legend_x + 18}" y="{ly + 11}" font-size="11" fill="#666">{sl}</text>'
            )

    label_els = ""
    for i, label in enumerate(labels):
        x = margin["left"] + i * group_w + group_w / 2
        label_els += f'<text x="{x:.1f}" y="{height - margin["bottom"] + 15}" text-anchor="middle" font-size="11" fill="#666" transform="rotate(-30,{x:.1f},{height - margin["bottom"] + 15})">{_escape(label)}</text>'

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
{gridlines}
{legend}
{bars}
<text x="{width / 2}" y="20" text-anchor="middle" font-size="14" font-weight="600" fill="#333">{_escape(title)}</text>
<text x="15" y="{height / 2}" text-anchor="middle" font-size="12" fill="#666" transform="rotate(-90,15,{height / 2})">{_escape(ylabel)}</text>
{label_els}
</svg>"""


def _empty_chart(width: int, height: int, message: str = "No data") -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
<text x="{width / 2}" y="{height / 2}" text-anchor="middle" font-size="14" fill="#999">{message}</text>
</svg>"""
