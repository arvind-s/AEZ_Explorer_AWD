"""
Map builders for the AEZ block classification pipeline.

Folium: straightforward GeoJson layer colored by zone_label.
Plotly: a SINGLE go.Choroplethmap trace (MapLibre-based; the modern
replacement for the deprecated go.Choroplethmapbox) using integer category
codes + a matching discrete colorscale, plus small invisible scatter traces
just to populate the legend. This avoids px.choropleth_map's per-category
GeoJSON duplication, which balloons file size with many categories.
"""
import json

import folium
import plotly.graph_objects as go


def build_folium_map(gdf, color_col="display_color", label_col="zone_label"):
    """gdf must be in EPSG:4326 and have columns: State, District, Block, and label/color cols."""
    bounds = gdf.total_bounds  # minx, miny, maxx, maxy
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    # OpenStreetMap tiles: no API key required (CartoDB's free tiles now do)
    m = folium.Map(location=center, zoom_start=6, tiles="OpenStreetMap")

    for _, row in gdf.iterrows():
        tooltip_text = (
            f"State: {row['State']}<br>"
            f"District: {row['District']}<br>"
            f"Block: {row['Block']}<br>"
            f"AEZ: {row[label_col]}"
        )
        gj = folium.GeoJson(
            row["geometry"].__geo_interface__,
            style_function=lambda feat, col=row[color_col]: {
                "fillColor": col,
                "color": "#555555",
                "weight": 0.4,
                "fillOpacity": 0.75,
            },
            tooltip=folium.Tooltip(tooltip_text),
        )
        gj.add_to(m)

    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    return m


def build_plotly_map(gdf, category_col="zone_label", color_map=None, shared_color="#888888",
                      no_data_color="#CCCCCC"):
    """
    Single go.Choroplethmapbox trace, colored via integer category codes +
    a discrete colorscale built from color_map, to avoid per-category GeoJSON
    duplication (the failure mode of px.choropleth_map with many categories).

    color_map: dict {single_zone_name: hex_color}. Shared / no-data blocks get
    their own dedicated colors (not part of color_map).
    """
    gdf = gdf.reset_index(drop=True).copy()
    gdf["_id"] = gdf.index.astype(str)

    # Build the ordered list of categories actually present, in a stable order:
    # single zones first (in color_map's order), then Shared, then No zone data.
    single_zone_categories = [z for z in color_map.keys() if (gdf[category_col] == z).any()]
    shared_present = gdf[category_col].str.startswith("Shared:").any()
    nodata_present = (gdf[category_col] == "No zone data").any()

    categories = list(single_zone_categories)
    if shared_present:
        categories.append("__SHARED__")
    if nodata_present:
        categories.append("__NODATA__")

    cat_to_code = {c: i for i, c in enumerate(categories)}

    def _code_for(label):
        if label in cat_to_code:
            return cat_to_code[label]
        if label.startswith("Shared:"):
            return cat_to_code["__SHARED__"]
        return cat_to_code.get("__NODATA__", 0)

    gdf["_zcode"] = gdf[category_col].map(_code_for)

    n = len(categories)
    colors = []
    for c in categories:
        if c == "__SHARED__":
            colors.append(shared_color)
        elif c == "__NODATA__":
            colors.append(no_data_color)
        else:
            colors.append(color_map[c])

    # discrete colorscale: flat steps across [0,1]
    if n == 1:
        colorscale = [[0.0, colors[0]], [1.0, colors[0]]]
    else:
        colorscale = []
        for i, col in enumerate(colors):
            lo = i / n
            hi = (i + 1) / n
            colorscale.append([lo, col])
            colorscale.append([hi, col])

    geojson = json.loads(gdf.to_json())

    hover_text = (
        "State: " + gdf["State"].astype(str)
        + "<br>District: " + gdf["District"].astype(str)
        + "<br>Block: " + gdf["Block"].astype(str)
        + "<br>AEZ: " + gdf[category_col].astype(str)
    )

    bounds = gdf.total_bounds
    center = {"lat": (bounds[1] + bounds[3]) / 2, "lon": (bounds[0] + bounds[2]) / 2}

    choropleth = go.Choroplethmap(
        geojson=geojson,
        locations=gdf["_id"],
        z=gdf["_zcode"],
        featureidkey="properties._id",
        colorscale=colorscale,
        zmin=0,
        zmax=n - 1 if n > 1 else 1,
        marker_line_width=0.3,
        marker_line_color="#555555",
        text=hover_text,
        hoverinfo="text",
        showscale=False,
    )

    fig = go.Figure(choropleth)

    # invisible marker traces just to populate a legend with real category names
    display_names = []
    for c in categories:
        if c == "__SHARED__":
            display_names.append("Shared (multiple zones)")
        elif c == "__NODATA__":
            display_names.append("No zone data")
        else:
            display_names.append(c)

    for name, col in zip(display_names, colors):
        fig.add_trace(
            go.Scattermap(
                lat=[None],
                lon=[None],
                mode="markers",
                marker=dict(size=10, color=col),
                name=name,
                showlegend=True,
            )
        )

    fig.update_layout(
        map_style="carto-positron",
        map_zoom=5.5,
        map_center=center,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(title="AEZ Zone", yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=700,
    )
    return fig
