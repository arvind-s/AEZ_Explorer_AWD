"""
Paddy phenology Streamlit app — multi-farm shapefile input, S1/S2 + GDD assessment.

Run from Crop_stage:
  streamlit run streamlit_app/app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phenology_pipeline.config import DEFAULT_GDD_STAGES, PhenologyConfig
from phenology_pipeline.phenology_engine import run_phenology_batch
from streamlit_app.io_utils import save_uploaded_vector

st.set_page_config(page_title="Paddy Phenology", layout="wide")
st.title("Paddy crop phenology assessment")
st.caption("Sentinel-1 VH + Sentinel-2 NDVI (STAC); optional ERA5-Land GDD staging (120-day kharif)")


def _stack_timeseries(timeseries: dict[str, pd.DataFrame], value_col: str) -> pd.DataFrame:
    rows = []
    for farm_id, df in timeseries.items():
        if value_col not in df.columns:
            continue
        part = df[["date", value_col]].copy()
        part["farm_id"] = farm_id
        rows.append(part)
    if not rows:
        return pd.DataFrame(columns=["date", value_col, "farm_id"])
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


def _add_pixel_traces(
    fig: go.Figure,
    dates: pd.DatetimeIndex,
    matrix: np.ndarray,
    row: int,
    *,
    color: str = "rgba(31, 119, 180, 0.35)",
) -> None:
    """Add one trace per pixel — raw values, no smoothing."""
    n_pixels = matrix.shape[0]
    if n_pixels == 0:
        return
    mode = "markers" if n_pixels > 150 else "lines+markers"
    line = dict(width=1, color=color)
    marker = dict(size=3 if n_pixels <= 150 else 2, color=color)
    for i in range(n_pixels):
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=matrix[i],
                mode=mode,
                line=line,
                marker=marker,
                showlegend=False,
                hovertemplate=f"pixel {i}<br>%{{x|%Y-%m-%d}}<br>%{{y:.4f}}<extra></extra>",
            ),
            row=row,
            col=1,
        )


def _plot_satellite_all(
    timeseries: dict[str, pd.DataFrame],
    pixel_timeseries: dict[str, dict] | None = None,
) -> go.Figure:
    pixel_timeseries = pixel_timeseries or {}
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Sentinel-2 NDVI (all farms)", "Sentinel-1 VH (all farms)"),
        vertical_spacing=0.08,
    )

    for farm_id in sorted(timeseries.keys()):
        px = pixel_timeseries.get(farm_id)
        if px and "ndvi" in px:
            _add_pixel_traces(
                fig,
                px["ndvi_dates"],
                px["ndvi"],
                row=1,
                color="rgba(31, 119, 180, 0.2)",
            )
        else:
            ts = timeseries[farm_id]
            fig.add_trace(
                go.Scatter(
                    x=ts["date"],
                    y=ts["ndvi"],
                    mode="lines+markers",
                    name=f"NDVI {farm_id}",
                ),
                row=1,
                col=1,
            )

        if px and "vh" in px:
            _add_pixel_traces(
                fig,
                px["vh_dates"],
                px["vh"],
                row=2,
                color="rgba(214, 39, 40, 0.2)",
            )
        else:
            ts = timeseries[farm_id]
            fig.add_trace(
                go.Scatter(
                    x=ts["date"],
                    y=ts["vh"],
                    mode="lines+markers",
                    name=f"VH {farm_id}",
                ),
                row=2,
                col=1,
            )

    fig.update_yaxes(title_text="NDVI", row=1, col=1)
    fig.update_yaxes(title_text="VH (dB)", row=2, col=1)
    fig.update_layout(height=620, showlegend=False)
    return fig


def _has_gdd_data(timeseries: dict[str, pd.DataFrame]) -> bool:
    for df in timeseries.values():
        if "cumulative_gdd" not in df.columns:
            continue
        if df["cumulative_gdd"].notna().any():
            return True
    return False


def _plot_gdd_all(timeseries: dict[str, pd.DataFrame]) -> go.Figure:
    gdd_long = _stack_timeseries(timeseries, "cumulative_gdd")
    gdd_long = gdd_long.dropna(subset=["cumulative_gdd"])
    if gdd_long.empty:
        return go.Figure().update_layout(title="No GDD data")

    fig = px.line(
        gdd_long,
        x="date",
        y="cumulative_gdd",
        color="farm_id",
        markers=True,
        title="Cumulative GDD from transplant (all farms)",
    )
    fig.update_layout(height=420, yaxis_title="GDD (°C)")
    return fig


def _plot_individual(
    ts: pd.DataFrame,
    farm_id: str,
    assessment: str,
    *,
    gdd_enabled: bool = False,
    pixel_data: dict | None = None,
) -> go.Figure:
    show_gdd = gdd_enabled and "cumulative_gdd" in ts.columns and ts["cumulative_gdd"].notna().any()
    n_rows = 3 if show_gdd else 2
    titles = ("NDVI (S2)", "VH (S1)", "Cumulative GDD") if show_gdd else ("NDVI (S2)", "VH (S1)")
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=titles,
        vertical_spacing=0.06 if show_gdd else 0.08,
    )
    assessment_dt = pd.Timestamp(assessment)

    if pixel_data and "ndvi" in pixel_data:
        _add_pixel_traces(fig, pixel_data["ndvi_dates"], pixel_data["ndvi"], row=1)
    else:
        fig.add_trace(
            go.Scatter(x=ts["date"], y=ts["ndvi"], name="NDVI", mode="lines+markers"),
            row=1,
            col=1,
        )

    if pixel_data and "vh" in pixel_data:
        _add_pixel_traces(
            fig,
            pixel_data["vh_dates"],
            pixel_data["vh"],
            row=2,
            color="rgba(214, 39, 40, 0.35)",
        )
    else:
        fig.add_trace(
            go.Scatter(x=ts["date"], y=ts["vh"], name="VH", mode="lines+markers"),
            row=2,
            col=1,
        )
    if show_gdd:
        fig.add_trace(
            go.Scatter(
                x=ts["date"],
                y=ts["cumulative_gdd"],
                name="GDD",
                mode="lines+markers",
            ),
            row=3,
            col=1,
        )

    for row in range(1, n_rows + 1):
        fig.add_vline(x=assessment_dt, line_dash="dash", line_color="gray", row=row, col=1)

    fig.update_yaxes(title_text="NDVI", row=1, col=1)
    fig.update_yaxes(title_text="VH (dB)", row=2, col=1)
    if show_gdd:
        fig.update_yaxes(title_text="GDD (°C)", row=3, col=1)
    fig.update_layout(height=520 if show_gdd else 480, title=f"Farm {farm_id}", showlegend=False)
    return fig


def _stage_bar(summary: pd.DataFrame) -> go.Figure:
    counts = summary["current_stage"].value_counts().reset_index()
    counts.columns = ["stage", "count"]
    fig = px.bar(counts, x="stage", y="count", title="Farm count by phenological stage")
    fig.update_layout(height=320)
    return fig


with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader(
        "Polygon layer (.zip shapefile or .geojson)",
        type=["zip", "geojson", "json", "shp"],
    )

    work_dir = Path(st.session_state.get("work_dir", tempfile.mkdtemp(prefix="phenology_st_")))
    st.session_state["work_dir"] = str(work_dir)

    gdf_preview: gpd.GeoDataFrame | None = None
    vector_path: Path | None = None

    if uploaded is not None:
        try:
            vector_path = save_uploaded_vector(uploaded.getvalue(), uploaded.name, work_dir)
            gdf_preview = gpd.read_file(vector_path)
            st.success(f"{len(gdf_preview)} polygon(s) loaded")
        except Exception as exc:
            st.error(f"Could not read upload: {exc}")

    col_options = list(gdf_preview.columns) if gdf_preview is not None else []
    non_geom = [c for c in col_options if c != "geometry"]

    farm_id_col = st.selectbox(
        "Farm ID column",
        options=non_geom or ["farm_id"],
        index=0 if non_geom else 0,
    )
    sowing_col = st.selectbox(
        "Sowing date column (optional)",
        options=["— none —"] + non_geom,
    )
    sowing_col = None if sowing_col == "— none —" else sowing_col

    st.header("Season window")
    download_start = st.date_input("Download start", value=pd.Timestamp("2024-05-01"))
    download_end = st.date_input("Download end", value=pd.Timestamp("2024-11-30"))
    peak_search_end = st.date_input("Peak search end", value=pd.Timestamp("2024-09-30"))
    assessment_date = st.date_input("Assessment date", value=pd.Timestamp("2024-10-01"))

    st.header("STAC")
    stac_interval = st.selectbox("Composite interval", ["12D", "30D"], index=0)
    stac_workers = st.slider("STAC workers", 1, 8, 2)
    force_download = st.checkbox("Force new STAC download", value=False)
    skip_download = st.checkbox("Use cached STAC only (no download)", value=False)

    compute_gdd = st.checkbox(
        "Compute GDD (ERA5 weather)",
        value=False,
        help="Off by default — satellite-only staging is much faster.",
    )
    era5_source = "openmeteo"
    era5_spatial = "aoi"
    if compute_gdd:
        with st.expander("Advanced: ERA5 / GDD weather", expanded=False):
            st.caption(
                "Default uses **Open-Meteo** — one fast request for the whole shapefile. "
                "Do not switch to CDS unless required; CDS queues jobs for minutes per farm."
            )
            era5_source = st.selectbox(
                "Weather source",
                options=["openmeteo", "cds"],
                index=0,
            )
            if era5_source == "cds":
                st.warning("CDS is very slow for multi-farm runs. Prefer Open-Meteo.")
            era5_spatial = st.selectbox(
                "Weather coverage",
                options=["aoi", "per_farm"],
                index=0,
                format_func=lambda x: "One point for all farms (fast)" if x == "aoi" else "Per-farm centroid (slower)",
            )

    run_btn = st.button("Run assessment", type="primary", disabled=gdf_preview is None)

if gdf_preview is not None:
    with st.expander("Preview attributes (first 10 rows)", expanded=False):
        st.dataframe(gdf_preview.drop(columns="geometry").head(10), use_container_width=True)

if run_btn and gdf_preview is not None and vector_path is not None:
    out_dir = work_dir / "phenology_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = PhenologyConfig(
        polygon_path=str(vector_path),
        farm_id_col=farm_id_col,
        download_start=download_start.strftime("%Y-%m-%d"),
        download_end=download_end.strftime("%Y-%m-%d"),
        peak_search_end=peak_search_end.strftime("%Y-%m-%d"),
        output_dir=str(out_dir),
        stac_interval=stac_interval,
        stac_max_workers=stac_workers,
        download_data=not skip_download,
        timeseries_cache_dir=str(out_dir / "stac_download"),
        era5_cache_dir=str(work_dir / "era5_cache"),
        era5_source=era5_source,
        era5_spatial_mode=era5_spatial,
        compute_gdd=compute_gdd,
    )

    progress_slot = st.empty()
    try:
        with st.status("Running phenology pipeline…", expanded=True) as status:

            def on_progress(msg: str) -> None:
                progress_slot.info(msg)
                status.update(label=msg)

            batch = run_phenology_batch(
                cfg,
                assessment_date=assessment_date.strftime("%Y-%m-%d"),
                sowing_col=sowing_col,
                force_download=force_download,
                on_progress=on_progress,
            )
            status.update(label="Done", state="complete")
        st.session_state["batch"] = batch
        st.session_state["gdf"] = gdf_preview
        st.session_state["farm_id_col"] = farm_id_col
        progress_slot.success(f"Completed {len(batch['summary'])} farm(s)")
    except Exception as exc:
        progress_slot.error(str(exc))
        st.exception(exc)

batch = st.session_state.get("batch")
if batch is None:
    st.info("Upload a polygon layer and click **Run assessment** in the sidebar.")
    st.stop()

summary = batch["summary"]
timeseries = batch["timeseries"]
pixel_timeseries = batch.get("pixel_timeseries", {})
assessment = batch["assessment_date"]
gdd_enabled = batch.get("gdd_enabled", False)
results_by_id = {r["farm_id"]: r for r in batch["results"] if r.get("farm_id")}

tab_ind, tab_all = st.tabs(["Individual farm", "All farms"])

with tab_all:
    st.subheader("Summary table")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    if timeseries:
        st.plotly_chart(_stage_bar(summary), use_container_width=True)
        st.plotly_chart(
            _plot_satellite_all(timeseries, pixel_timeseries),
            use_container_width=True,
        )
        if gdd_enabled and _has_gdd_data(timeseries):
            st.plotly_chart(_plot_gdd_all(timeseries), use_container_width=True)
        elif not gdd_enabled:
            st.info("GDD not computed — enable **Compute GDD** in the sidebar for weather-based staging.")
    else:
        st.warning("No timeseries available — check STAC download or per-farm errors.")

with tab_ind:
    farm_ids = sorted(summary["farm_id"].dropna().astype(str).tolist())
    if not farm_ids:
        st.warning("No farms in summary.")
    else:
        selected = st.selectbox("Select farm", farm_ids)
        result = results_by_id.get(selected, {})
        ts = timeseries.get(selected)
        px = pixel_timeseries.get(selected)

        if result.get("error"):
            st.error(result["error"])

        if px:
            st.caption(f"Showing **{px.get('n_pixels', 0)}** raw pixel trace(s) — no smoothing applied.")
        elif ts is not None:
            st.caption("Per-pixel CSV not available — showing raw polygon-level series (NetCDF aggregate).")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stage", result.get("current_stage", "—"))
        if gdd_enabled:
            c2.metric("Cumulative GDD", result.get("cumulative_gdd", "—"))
        else:
            c2.metric("Satellite hint", result.get("satellite_stage_hint", "—"))
        c3.metric("Days after transplant", result.get("days_after_transplant", "—"))
        c4.metric("Confidence", result.get("stage_confidence", "—"))

        c5, c6, c7 = st.columns(3)
        c5.metric("Used transplant", result.get("used_transplant", "—"))
        c6.metric("Est. sowing", result.get("estimated_sowing", "—"))
        c7.metric("Est. harvest", result.get("estimated_harvest_date", "—"))

        if result.get("anomaly_flags"):
            st.warning("Flags: " + ", ".join(result["anomaly_flags"]))

        if ts is not None:
            st.plotly_chart(
                _plot_individual(
                    ts,
                    selected,
                    assessment,
                    gdd_enabled=gdd_enabled,
                    pixel_data=px,
                ),
                use_container_width=True,
            )
        else:
            st.info("No timeseries for this farm.")

        with st.expander("Full JSON result"):
            st.json(result)

        if gdd_enabled:
            with st.expander("GDD stage thresholds (120-day variety)"):
                stage_df = pd.DataFrame(
                    [{"stage": s.name, "gdd_min": s.gdd_min, "gdd_max": s.gdd_max} for s in DEFAULT_GDD_STAGES]
                )
                st.dataframe(stage_df, hide_index=True)
