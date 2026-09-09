"""
Core AEZ (Agro-Ecological Zone) block classification pipeline.

Reusable by both the Jupyter notebook and the Streamlit app, and safe to
re-run from scratch against a new CSV/shapefile pair.
"""
import re
import difflib
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib

# ---------------------------------------------------------------------------
# 1. Key normalization (State / District / Block)
# ---------------------------------------------------------------------------

def normalize_key(s: str) -> str:
    """Strip, uppercase, collapse internal whitespace to a single space."""
    if pd.isna(s):
        return ""
    return re.sub(r"\s+", " ", str(s).strip()).upper()


# ---------------------------------------------------------------------------
# 2. AEZ column name normalization + tolerant short-name mapping
# ---------------------------------------------------------------------------

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())


# Canonical (idealized) AEZ eco-region names. New exports may have double
# spaces, a stray space inside a word (e.g. "K ARNATAKA"), or minor spelling
# drift (e.g. "SUBAUMID" for "SUBHUMID", "EGO-REGION" for "ECO-REGION").
# We match on these tolerant of all of the above.
AEZ_REFERENCE = [
    "ASSAM AND BENGAL PLAIN HOT SUBHUMID TO HUMID (INCLUSION OF PERHUMID) ECO-REGION",
    "CENTRAL HIGHLANDS (MALWA) GUJARAT PLAIN AND KATHIAWAR PENINSULA SEMI-ARID ECO-REGION",
    "CENTRAL HIGHLANDS (MALWA AND BUNDELKHAND) HOT SUBHUMID (DRY) ECO-REGION",
    "DECCAN PLATEAU (TELANGANA) AND EASTERN GHATS HOT SEMI ARID ECO-REGION",
    "DECCAN PLATEAU HOT SEMI-ARID ECO-REGION",
    "EASTERN COASTAL PLAIN HOT SUBHUMID TO SEMI-ARID ECO-REGION",
    "EASTERN GHATS AND TAMIL NADU UPLANDS AND DECCAN (KARNATAKA) PLATEAU HOT SEMI-ARID ECO-REGION",
    "EASTERN HIMALAYAS WARM PERHUMID ECO-REGION",
    "EASTERN PLAIN HOT SUBHUMID (MOIST) ECO-REGION",
    "EASTERN PLATEAU (CHHATTISGARH REGION)",
    "EASTERN PLATEAU (CHHOTANAGPUR) AND EASTERN GHATS HOT SUBHUMID ECO-REGION",
    "KARNATAKA PLATEAU (RAYALSEEMA AS INCLUSION)",
    "WESTERN GHATS AND COASTAL PLAIN HOT HUMID-PERHUMID ECO-REGION",
]

SHORT_NAMES = {
    AEZ_REFERENCE[0]: "Assam & Bengal Plain (Subhumid\u2013Humid)",
    AEZ_REFERENCE[1]: "Central Highlands (Malwa)\u2013Gujarat Plain & Kathiawar (Semi-Arid)",
    AEZ_REFERENCE[2]: "Central Highlands (Malwa\u2013Bundelkhand, Subhumid Dry)",
    AEZ_REFERENCE[3]: "Deccan Plateau (Telangana) & Eastern Ghats (Semi-Arid)",
    AEZ_REFERENCE[4]: "Deccan Plateau (Semi-Arid)",
    AEZ_REFERENCE[5]: "Eastern Coastal Plain (Subhumid\u2013Semi-Arid)",
    AEZ_REFERENCE[6]: "Eastern Ghats, TN Uplands & Deccan (Karnataka) Plateau (Semi-Arid)",
    AEZ_REFERENCE[7]: "Eastern Himalayas (Perhumid)",
    AEZ_REFERENCE[8]: "Eastern Plain (Subhumid Moist)",
    AEZ_REFERENCE[9]: "Eastern Plateau (Chhattisgarh)",
    AEZ_REFERENCE[10]: "Eastern Plateau (Chhotanagpur) & Eastern Ghats (Subhumid)",
    AEZ_REFERENCE[11]: "Karnataka Plateau (incl. Rayalaseema)",
    AEZ_REFERENCE[12]: "Western Ghats & Coastal Plain (Humid\u2013Perhumid)",
}


def match_zone_column(raw_col: str, cutoff: float = 0.80):
    """
    Map a raw (possibly messy) AEZ column header to a clean short display name.

    Tolerant of: double/irregular internal whitespace, a stray space inserted
    inside a word, and small spelling drift. Returns (short_name, method) where
    method is one of 'exact', 'fuzzy', 'fallback'. A 'fallback' match means we
    could not confidently match a known AEZ, so we generate a clean title-cased
    label from the (whitespace-normalized) raw name -- we NEVER return the raw
    ugly multi-space string itself.
    """
    key = normalize_ws(raw_col).upper()
    key_nospace = re.sub(r"\s+", "", key)
    ref_nospace = {re.sub(r"\s+", "", r): r for r in AEZ_REFERENCE}

    if key_nospace in ref_nospace:
        ref = ref_nospace[key_nospace]
        return SHORT_NAMES[ref], "exact"

    matches = difflib.get_close_matches(key_nospace, list(ref_nospace.keys()), n=1, cutoff=cutoff)
    if matches:
        ref = ref_nospace[matches[0]]
        return SHORT_NAMES[ref], "fuzzy"

    # Unknown column: never fall back to the raw ugly string. Build a clean
    # readable label instead and flag it so the caller can warn the user.
    clean_fallback = normalize_ws(raw_col).title()
    return clean_fallback, "fallback"


def build_zone_column_map(zone_columns):
    """
    Given the list of raw AEZ column names from the CSV, return:
      - col_to_short: dict {raw_col: short_display_name}
      - match_report: list of dicts describing how each column was matched
    """
    col_to_short = {}
    match_report = []
    for c in zone_columns:
        short, method = match_zone_column(c)
        col_to_short[c] = short
        match_report.append({"raw_column": c, "short_name": short, "match_method": method})
    return col_to_short, match_report


# ---------------------------------------------------------------------------
# 3. Merge CSV <-> shapefile on normalized keys
# ---------------------------------------------------------------------------

KEY_COLS = ["State", "District", "Block"]


def load_and_merge(csv_path, shp_path):
    """
    Load the CSV and shapefile, normalize the three key columns, and merge
    1:1 on those keys. Returns:
      merged (GeoDataFrame), report (dict with match stats + unmatched keys)
    """
    df = pd.read_csv(csv_path)
    gdf = gpd.read_file(shp_path)

    for c in KEY_COLS:
        if c not in df.columns:
            raise ValueError(f"CSV is missing required key column: {c}")
        if c not in gdf.columns:
            raise ValueError(f"Shapefile is missing required key column: {c}")

    for c in KEY_COLS:
        df[f"_{c}_key"] = df[c].map(normalize_key)
        gdf[f"_{c}_key"] = gdf[c].map(normalize_key)

    key_cols_norm = [f"_{c}_key" for c in KEY_COLS]

    # detect duplicate keys on either side (would break 1:1 assumption)
    df_dupes = df[df.duplicated(subset=key_cols_norm, keep=False)]
    gdf_dupes = gdf[gdf.duplicated(subset=key_cols_norm, keep=False)]

    merged = gdf.merge(
        df.drop(columns=KEY_COLS),
        on=key_cols_norm,
        how="outer",
        indicator=True,
        suffixes=("_shp", "_csv"),
    )

    matched = merged[merged["_merge"] == "both"].copy()
    csv_only = merged[merged["_merge"] == "right_only"].copy()
    shp_only = merged[merged["_merge"] == "left_only"].copy()

    def _rowkey(row):
        return " / ".join(row[k] for k in key_cols_norm)

    unmatched_csv_keys = [_rowkey(r) for _, r in csv_only.iterrows()]
    unmatched_shp_keys = [_rowkey(r) for _, r in shp_only.iterrows()]

    report = {
        "csv_rows": len(df),
        "shp_rows": len(gdf),
        "matched_rows": len(matched),
        "csv_unmatched_count": len(csv_only),
        "shp_unmatched_count": len(shp_only),
        "csv_unmatched_keys": unmatched_csv_keys,
        "shp_unmatched_keys": unmatched_shp_keys,
        "csv_duplicate_keys": sorted(set(_rowkey(r) for _, r in df_dupes.assign(**{k: df_dupes[k] for k in key_cols_norm}).iterrows())) if len(df_dupes) else [],
        "shp_duplicate_keys": sorted(set(_rowkey(r) for _, r in gdf_dupes.assign(**{k: gdf_dupes[k] for k in key_cols_norm}).iterrows())) if len(gdf_dupes) else [],
    }

    matched = matched.drop(columns=["_merge"] + key_cols_norm)
    # prefer shapefile's State/District/Block spelling for display (post-merge
    # both _shp/_csv copies may exist if names collided); normalize to single cols
    for c in KEY_COLS:
        shp_col, csv_col = f"{c}_shp", f"{c}_csv"
        if shp_col in matched.columns and csv_col in matched.columns:
            matched[c] = matched[shp_col]
            matched = matched.drop(columns=[shp_col, csv_col])

    matched = gpd.GeoDataFrame(matched, geometry="geometry", crs=gdf.crs)
    return matched, report


# ---------------------------------------------------------------------------
# 4. Zone classification
# ---------------------------------------------------------------------------

def classify_zones(gdf, zone_columns, col_to_short):
    """
    Add zones_list, zone_count, zone_label, is_shared columns to gdf.
    zone_columns: raw AEZ column names present in gdf.
    col_to_short: mapping raw column -> clean short display name.
    """
    gdf = gdf.copy()
    vals = gdf[zone_columns].apply(pd.to_numeric, errors="coerce").fillna(0)

    zones_list = []
    for _, row in vals.iterrows():
        touched = [col_to_short[c] for c in zone_columns if row[c] != 0]
        zones_list.append(touched)

    gdf["zones_list"] = zones_list
    gdf["zone_count"] = gdf["zones_list"].map(len)
    gdf["is_shared"] = gdf["zone_count"] >= 2

    def _label(zl):
        if len(zl) == 0:
            return "No zone data"
        if len(zl) == 1:
            return zl[0]
        return "Shared: " + " + ".join(zl)

    gdf["zone_label"] = gdf["zones_list"].map(_label)
    return gdf


# ---------------------------------------------------------------------------
# 5. Colors: Spectral for single zones, dedicated grey for shared
# ---------------------------------------------------------------------------

SHARED_COLOR = "#888888"
NO_DATA_COLOR = "#CCCCCC"


def build_color_map(single_zone_names):
    """
    single_zone_names: sorted list of unique single-zone display names.
    Returns dict {zone_name: hex_color}, sampled evenly along the Spectral
    diverging colormap so zones are visually well separated.
    """
    n = len(single_zone_names)
    cmap = matplotlib.colormaps["Spectral"]
    if n == 1:
        positions = [0.5]
    else:
        positions = np.linspace(0.0, 1.0, n)
    colors = [matplotlib.colors.to_hex(cmap(p)) for p in positions]
    return dict(zip(single_zone_names, colors))


def assign_display_color(row, color_map):
    if row["zone_count"] >= 2:
        return SHARED_COLOR
    if row["zone_count"] == 0:
        return NO_DATA_COLOR
    return color_map.get(row["zones_list"][0], NO_DATA_COLOR)


# ---------------------------------------------------------------------------
# 6. Geometry: reproject + simplify with size check
# ---------------------------------------------------------------------------

def reproject_and_simplify(gdf, tolerance_m=300, target_crs="EPSG:4326", max_mb=None):
    """
    Ensure gdf is in target_crs. Simplify in a metric CRS (EPSG:3857) by
    tolerance_m meters, then reproject back to target_crs.
    If max_mb is given, halves the tolerance (up to a few iterations) until
    the resulting GeoJSON is under that size, or gives up and warns.
    """
    gdf = gdf.copy()
    if gdf.crs is None:
        raise ValueError("Input geometry has no CRS defined; cannot reproject safely.")

    working_crs = "EPSG:3857"  # meters, good enough for a simplify tolerance in meters
    gdf_m = gdf.to_crs(working_crs)

    def _simplify(tol):
        g = gdf_m.copy()
        g["geometry"] = g["geometry"].simplify(tol, preserve_topology=True)
        return g.to_crs(target_crs)

    tol = tolerance_m
    result = _simplify(tol)

    if max_mb is not None:
        for _ in range(5):
            size_mb = len(result.to_json()) / (1024 * 1024)
            if size_mb <= max_mb:
                break
            tol *= 2
            result = _simplify(tol)
        else:
            warnings.warn(
                f"Could not get simplified geometry under {max_mb} MB "
                f"even after increasing tolerance to {tol} m."
            )

    if gdf.crs.to_string() != target_crs:
        pass  # already reprojected via _simplify's .to_crs(target_crs)

    return result, tol
