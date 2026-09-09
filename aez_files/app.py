"""
AEZ Block Explorer -- Streamlit app.

Cascading filters: State -> District -> Block -> AEZ Zone.
Shows a live-updating map (Plotly) and a filtered data table with CSV download.

Data lives in ./data/ next to this file (self-contained for git / Streamlit
Cloud deployment). To refresh against new input files: replace the CSV/
shapefile in ./data/ (or point AEZ_CSV_PATH / AEZ_SHP_PATH env vars elsewhere)
and rerun -- everything here is recomputed from scratch on each load, nothing
is cached from a prior dataset.
"""
import os

import pandas as pd
import streamlit as st

from aez_core import (
    load_and_merge, build_zone_column_map, classify_zones,
    build_color_map, assign_display_color, reproject_and_simplify,
)
from aez_maps import build_plotly_map

st.set_page_config(page_title="AEZ Block Explorer", layout="wide")

# ---------------------------------------------------------------------------
# Paths -- resolved relative to this file so the app works after a fresh
# `git clone` regardless of working directory. Override via env vars to
# point at a different CSV/shapefile pair.
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.environ.get(
    "AEZ_CSV_PATH", os.path.join(APP_DIR, "data", "AWD_Sampling_all_blocks_master.csv")
)
SHP_PATH = os.environ.get(
    "AEZ_SHP_PATH", os.path.join(APP_DIR, "data", "all_blocks_master.shp")
)


@st.cache_data(show_spinner="Loading and classifying blocks...")
def load_data(csv_path, shp_path):
    merged, merge_report = load_and_merge(csv_path, shp_path)

    non_zone_cols = ["District", "Block", "State", "geometry"]
    zone_columns = [c for c in merged.columns if c not in non_zone_cols]
    col_to_short, match_report = build_zone_column_map(zone_columns)

    gdf = classify_zones(merged, zone_columns, col_to_short)

    single_zone_names = sorted(
        gdf.loc[gdf["zone_count"] == 1, "zones_list"].map(lambda z: z[0]).unique()
    )
    color_map = build_color_map(single_zone_names)
    gdf["display_color"] = gdf.apply(lambda r: assign_display_color(r, color_map), axis=1)

    gdf_simplified, tol_used = reproject_and_simplify(gdf, tolerance_m=300, max_mb=8)
    for col in ["State", "District", "Block", "zone_label", "zones_list", "zone_count",
                "is_shared", "display_color"]:
        gdf_simplified[col] = gdf[col].values

    return gdf_simplified, merge_report, color_map, single_zone_names


gdf, merge_report, color_map, single_zone_names = load_data(CSV_PATH, SHP_PATH)

st.title("AEZ Block Explorer")
st.caption(
    f"{merge_report['matched_rows']} blocks loaded "
    f"({merge_report['csv_unmatched_count']} CSV rows and "
    f"{merge_report['shp_unmatched_count']} shapefile rows could not be matched)."
)

# ---------------------------------------------------------------------------
# Cascading filters
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    states = ["All"] + sorted(gdf["State"].unique())
    sel_state = st.selectbox("State", states)

df_state = gdf if sel_state == "All" else gdf[gdf["State"] == sel_state]

with col2:
    districts = ["All"] + sorted(df_state["District"].unique())
    sel_district = st.selectbox("District", districts)

df_district = df_state if sel_district == "All" else df_state[df_state["District"] == sel_district]

with col3:
    blocks = ["All"] + sorted(df_district["Block"].unique())
    sel_block = st.selectbox("Block", blocks)

df_block = df_district if sel_block == "All" else df_district[df_district["Block"] == sel_block]

with col4:
    zone_options = ["All"] + single_zone_names + ["Shared (any zones)"]
    sel_zone = st.selectbox("AEZ Zone", zone_options)

if sel_zone == "All":
    df_filtered = df_block
elif sel_zone == "Shared (any zones)":
    df_filtered = df_block[df_block["zone_count"] >= 2]
else:
    # Match any block that touches this zone, including shared blocks --
    # not just blocks where it's the sole zone.
    df_filtered = df_block[df_block["zones_list"].map(lambda zl: sel_zone in zl)]

st.markdown(f"**{len(df_filtered)} block(s) match the current filters.**")

# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
if len(df_filtered) == 0:
    st.warning("No blocks match this combination of filters.")
else:
    fig = build_plotly_map(df_filtered, color_map=color_map)
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Table + download
# ---------------------------------------------------------------------------
st.subheader("Filtered blocks")
table_cols = ["State", "District", "Block", "zone_count", "zone_label"]
table_df = df_filtered[table_cols].reset_index(drop=True)
st.dataframe(table_df, width="stretch")

csv_bytes = table_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered data as CSV",
    data=csv_bytes,
    file_name="aez_filtered_blocks.csv",
    mime="text/csv",
)
