import streamlit as st
import pandas as pd
import numpy as np
import pgeocode
import pydeck as pdk
import plotly.express as px
import json
import requests
from pathlib import Path
import tempfile
import os
import re


US_STATES_GEOJSON_URL = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"

st.set_page_config(page_title="Prospect Geo Maps", layout="wide")
st.title("Geographical Distribution Map Builder")

# =========================================================
# Helpers
# =========================================================

STATE_ABBR = {
    "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA","Colorado":"CO",
    "Connecticut":"CT","Delaware":"DE","Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID",
    "Illinois":"IL","Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA",
    "Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI","Minnesota":"MN",
    "Mississippi":"MS","Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV",
    "New Hampshire":"NH","New Jersey":"NJ","New Mexico":"NM","New York":"NY","North Carolina":"NC",
    "North Dakota":"ND","Ohio":"OH","Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA",
    "Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD","Tennessee":"TN","Texas":"TX",
    "Utah":"UT","Vermont":"VT","Virginia":"VA","Washington":"WA","West Virginia":"WV",
    "Wisconsin":"WI","Wyoming":"WY","District of Columbia":"DC"
}


@st.cache_data(show_spinner=False)

def load_us_states_geojson() -> dict:
    r = requests.get(US_STATES_GEOJSON_URL, timeout=30)
    r.raise_for_status()
    return r.json()

def load_map_style(url: str) -> dict:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def apply_line_layer_overrides(style: dict, layer_ids: list, color="#2B2B2B", opacity=0.9, width_px=2.0):
    import json
    style = json.loads(json.dumps(style))  # deep copy

    idset = set(layer_ids or [])
    for layer in style.get("layers", []):
        if layer.get("id") in idset and layer.get("type") == "line":
            paint = layer.setdefault("paint", {})
            layout = layer.setdefault("layout", {})

            # Force solid, darker boundaries
            paint["line-color"] = color
            paint["line-opacity"] = float(opacity)

            # FORCE constant width (this overrides zoom expressions)
            paint["line-width"] = float(width_px)

            # Remove dash if present (dashed borders look "light")
            if "line-dasharray" in paint:
                paint.pop("line-dasharray", None)
            if "line-dasharray" in layout:
                layout.pop("line-dasharray", None)

            # Reduce blur so lines look crisp/darker
            paint["line-blur"] = 0

    return style

def list_line_layers(style: dict) -> pd.DataFrame:
    rows = []
    for lyr in style.get("layers", []):
        if lyr.get("type") == "line":
            lid = lyr.get("id", "")
            src = lyr.get("source", "")
            s_layer = lyr.get("source-layer", "")
            rows.append({"id": lid, "source": src, "source_layer": s_layer})
    return pd.DataFrame(rows)

def load_excel(file) -> dict:
    xls = pd.ExcelFile(file)
    return {s: pd.read_excel(file, sheet_name=s) for s in xls.sheet_names}

@st.cache_data(show_spinner=False)
def zip_to_latlon(zip_series: pd.Series) -> pd.DataFrame:
    z = zip_series.astype(str).str.zfill(5)
    nomi = pgeocode.Nominatim("us")
    geo = nomi.query_postal_code(z.tolist())
    return pd.DataFrame({"zip5": z.values, "lat": geo["latitude"].values, "lon": geo["longitude"].values})

# def parse_capacity_min(value):
#     if pd.isna(value):
#         return np.nan
#     s = str(value).replace("$", "").replace(",", "").replace("+", "").strip()
#     try:
#         return float(s)
#     except Exception:
#         return np.nan

def parse_capacity_min(value):
    """
    Converts capacity bins like:
    $100K, $500K, $1M, $5M, $25M, $5MM
    into numeric values for correct ordering.
    """
    if pd.isna(value):
        return np.inf

    s = str(value).strip().upper()

    # Remove symbols
    s = s.replace("$", "").replace(",", "").replace("+", "").strip()

    # Handle "K"
    if "K" in s:
        return float(s.replace("K", "")) * 1_000

    # Handle "M" or "MM"
    if "MM" in s:
        return float(s.replace("MM", "")) * 1_000_000

    if "M" in s:
        return float(s.replace("M", "")) * 1_000_000

    # Fallback numeric
    digits = re.sub(r"[^0-9.]", "", s)
    if digits == "":
        return np.inf

    return float(digits)

def hex_to_rgb(hex_color: str):
    h = str(hex_color).lstrip("#")
    if len(h) != 6:
        return [153, 153, 153]
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]

def build_color_map_stable(all_bins: list) -> dict:
    """
    Stable, permanent mapping:
    - sort bins by numeric minimum (100K lightest -> 5MM darkest)
    - assign colors in a fixed palette (light blue -> dark blue -> yellow -> light maroon -> dark maroon)
    - mapping depends ONLY on the *full* list of bins, not filtered data
    """
    bins_sorted = sorted(all_bins, key=parse_capacity_min)

    palette = [
    "#7CAAC1",  # light cyan-blue (100K)
    # "#4FD1D9",  # cyan (new, clearer separation)
    "#257298",  # blue-teal (mid)
    "#F7BD88",  # light yellow (new)
    "#7C5F44",  # richer yellow
    "#F48789",  # light maroon / red
    "#FF3838",  # dark maroon (highest)
]


    # If more bins than palette, interpolate by repeating (or you can extend palette)
    return {b: palette[min(i, len(palette) - 1)] for i, b in enumerate(bins_sorted)}

# def render_deck_png_from_html(html: str, width: int = 2400, height: int = 1600, scale: int = 2) -> bytes:
#     """
#     Render a pydeck HTML string into a high-res PNG (cached by html+settings).
#     Requires Playwright installed + chromium installed.
#     """
#     from playwright.sync_api import sync_playwright

#     with tempfile.TemporaryDirectory() as td:
#         html_path = Path(td) / "map.html"
#         html_path.write_text(html, encoding="utf-8")

#         with sync_playwright() as p:
#             browser = p.chromium.launch()
#             page = browser.new_page(
#                 viewport={"width": int(width), "height": int(height)},
#                 device_scale_factor=int(scale),
#             )
#             page.goto(html_path.as_uri())
#             page.wait_for_timeout(1500)  # allow tiles/layers to load
#             png_bytes = page.screenshot(full_page=False)
#             browser.close()

#     return png_bytes

def render_deck_png_from_html(
    html: str,
    width: int = 1200,
    height: int = 700,
    scale: int = 2,
) -> bytes:
    """
    Render pydeck HTML -> PNG of ONLY the map container (not the full page).
    Uses Playwright to screenshot the deck.gl element.
    """
    from playwright.sync_api import sync_playwright
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        html_path = Path(td) / "map.html"
        html_path.write_text(html, encoding="utf-8")

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(
                viewport={"width": int(width), "height": int(height)},
                device_scale_factor=int(scale),
            )

            page.goto(html_path.as_uri())
            page.wait_for_timeout(1500)  # allow tiles/layers to load

            # Try common pydeck/deck.gl container selectors (varies by version)
            selectors = [
                "#deckgl-wrapper",
                "div#deckgl-wrapper",
                "div[data-testid='deckgl-wrapper']",
                ".deckgl-wrapper",
                "canvas",  # fallback (captures only the canvas)
            ]

            shot = None
            for sel in selectors:
                loc = page.locator(sel)
                if loc.count() > 0:
                    try:
                        loc.first.wait_for(state="visible", timeout=2000)
                        shot = loc.first.screenshot()  # <-- element-only screenshot
                        break
                    except Exception:
                        pass

            # Last resort: full viewport (should rarely happen)
            if shot is None:
                shot = page.screenshot(full_page=False)

            browser.close()
            return shot

def normalize_radii_meters(counts: pd.Series, min_m: float, max_m: float) -> pd.Series:
    """
    Radius mapping in METERS (scales naturally with zoom):
    - winsorize at p95 to prevent one group dominating
    - log1p scale
    - normalized to [min_m, max_m]
    - IMPORTANT: if all values are identical, return min_m (keeps single points small)
    """
    c = counts.astype(float).fillna(0.0)

    if len(c) == 0:
        return pd.Series([], dtype=float)

    cap_hi = float(np.quantile(c, 0.95))
    cap_hi = max(cap_hi, 5.0)
    c = c.clip(upper=cap_hi)

    z = np.log1p(c)
    zmin, zmax = float(z.min()), float(z.max())

    if zmax <= zmin:
        return pd.Series(np.full(len(z), min_m), index=counts.index, dtype=float)

    norm = (z - zmin) / (zmax - zmin)
    return min_m + norm * (max_m - min_m)

@st.cache_data(show_spinner=False)
def load_state_label_points(states_geojson: dict) -> pd.DataFrame:
    """
    Label point per state using polygon centroid (area-weighted approximation).
    Falls back to bbox center if geometry is odd.
    """
    rows = []
    for feat in states_geojson.get("features", []):
        props = feat.get("properties", {}) or {}
        name = props.get("name", "")

        geom = feat.get("geometry", {}) or {}
        gtype = geom.get("type", "")
        coords = geom.get("coordinates", [])

        # collect rings
        rings = []
        if gtype == "Polygon":
            rings = coords
        elif gtype == "MultiPolygon":
            for poly in coords:
                rings.extend(poly)
        else:
            continue

        # Use the largest ring by point count as "main" ring (good enough for states)
        main = max(rings, key=lambda r: len(r)) if rings else None
        if not main:
            continue

        # centroid of polygon ring (lon/lat) using shoelace (works fine for these shapes)
        x = [p[0] for p in main]
        y = [p[1] for p in main]
        if len(x) < 3:
            continue

        a = 0.0
        cx = 0.0
        cy = 0.0
        for i in range(len(x) - 1):
            cross = x[i] * y[i + 1] - x[i + 1] * y[i]
            a += cross
            cx += (x[i] + x[i + 1]) * cross
            cy += (y[i] + y[i + 1]) * cross

        if abs(a) < 1e-12:
            # fallback bbox center
            lon = (min(x) + max(x)) / 2.0
            lat = (min(y) + max(y)) / 2.0
        else:
            a *= 0.5
            cx /= (6.0 * a)
            cy /= (6.0 * a)
            lon, lat = cx, cy

        rows.append({"state_name": name, "lon": lon, "lat": lat})

    return pd.DataFrame(rows)


# =========================================================
# Upload + Load
# =========================================================
uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
if not uploaded:
    st.info("Upload an Excel file to begin.")
    st.stop()

sheets = load_excel(uploaded)
sheet_name = st.selectbox("Select sheet", list(sheets.keys()))
df_raw = sheets[sheet_name].copy()

st.subheader("Preview")
st.dataframe(df_raw.head(25), use_container_width=True)

# =========================================================
# Column mapping (defaults to your schema)
# =========================================================
st.sidebar.header("Column Mapping")
cols = df_raw.columns.tolist()

def idx(colname, fallback=0):
    return cols.index(colname) if colname in cols else fallback

id_col = st.sidebar.selectbox("Unique ID column", cols, index=idx("constituent_id", 0))
zip_col = st.sidebar.selectbox("ZIP 5 column", cols, index=idx("zip_5_digit", idx("zip", 0)))
bin_col = st.sidebar.selectbox("Capacity bin column", cols, index=idx("gift_capacity_min_bin", 0))
city_col = st.sidebar.selectbox("City column", cols, index=idx("home_city", 0))
state_col = st.sidebar.selectbox("State column", cols, index=idx("home_state", 0))

# =========================================================
# Prep base data
# =========================================================
df = df_raw.copy()

# Ensure ID is string-like and non-null
df[id_col] = df[id_col].astype(str).str.strip()
df = df[df[id_col].notna() & (df[id_col] != "")].copy()

# ZIP cleanup: ensure 5 digits, drop invalid
df["zip5"] = df[zip_col].astype(str).str.strip().str.zfill(5)
df = df[df["zip5"].str.match(r"^\d{5}$", na=False)].copy()
df = df[df["zip5"] != "00000"].copy()

# Region + bin
df["region"] = df[city_col].astype(str).str.strip() + ", " + df[state_col].astype(str).str.strip()
df["cap_bin"] = df[bin_col].astype(str).str.strip()
bins_master = sorted(df["cap_bin"].dropna().unique().tolist(), key=parse_capacity_min)
color_map = build_color_map_stable(bins_master)

# =========================================================
# Filters
# =========================================================
st.sidebar.header("Filters")
states = sorted(df[state_col].dropna().astype(str).unique().tolist())
state_filter = st.sidebar.multiselect("Filter states", states, default=[])
if state_filter:
    df = df[df[state_col].astype(str).isin(state_filter)].copy()

# ✅ MASTER bins list BEFORE bin filtering (use the current df after state filter)
bins_master = sorted(df["cap_bin"].dropna().unique().tolist(), key=parse_capacity_min)

# ✅ Permanent colors tied to bins_master ONLY
color_map = build_color_map_stable(bins_master)

# ✅ Use master bins for the multiselect so labels/colors stay stable
bins_selected = st.sidebar.multiselect("Capacity bins to show", bins_master, default=bins_master)
df = df[df["cap_bin"].isin(bins_selected)].copy()

# ✅ Display order is just the master order filtered down
bin_order = [b for b in bins_master if b in bins_selected]

# Geocode ZIP centroids
geo = zip_to_latlon(df["zip5"])
df = df.merge(geo, on="zip5", how="left")
df = df.dropna(subset=["lat", "lon"]).copy()

states_for_zoom = (
    df[state_col]
    .dropna()
    .astype(str)
    .sort_values()
    .unique()
    .tolist()
)

# =========================================================
# Aggregation
# =========================================================

st.sidebar.header("Aggregation")

agg_mode = st.sidebar.selectbox(
    "Map aggregation level",
    ["ZIP (zip5 + capacity bin)", "City (city + capacity bin)"],
    index=0  # ✅ default = ZIP
)

if agg_mode.startswith("ZIP"):
    group_cols = ["zip5", "lat", "lon", "region", "cap_bin"]
else:
    # city centroids based on ZIP centroids
    city_centroids = (
        df.groupby("region", dropna=False, observed=True)[["lat", "lon"]]
          .mean()
          .reset_index()
    )
    df2 = df.drop(columns=["lat", "lon"]).merge(city_centroids, on="region", how="left")
    df = df2
    group_cols = ["region", "lat", "lon", "cap_bin"]

# COUNT UNIQUE constituent_id per group
df_agg = (
    df.groupby(group_cols, dropna=False, observed=True)[id_col]
      .nunique(dropna=True)
      .reset_index(name="n_ids")
)

# Colors (same mapping for map + bar chart)
df_agg["color_hex"] = df_agg["cap_bin"].map(color_map).fillna("#999999")
df_agg["fill_rgb"] = df_agg["color_hex"].apply(hex_to_rgb)
df_agg["line_rgb"] = df_agg["fill_rgb"]

# draw order low behind high (stable)
df_agg["cap_bin"] = pd.Categorical(df_agg["cap_bin"], categories=bin_order, ordered=True)
df_agg = df_agg.sort_values("cap_bin", ascending=True).copy()
df_agg_all = df_agg.copy()   # <-- keep full aggregated set for dropdown + camera


# use_px_clamp = st.sidebar.checkbox("Clamp bubble size on screen (px)", value=True)
# if use_px_clamp:
#     radius_min_px = st.sidebar.slider("Min radius on screen (px)", 0, 10, 1)
#     radius_max_px = st.sidebar.slider("Max radius on screen (px)", 10, 200, 80)
# else:
#     radius_min_px = 0
#     radius_max_px = 100000  # effectively off


# =========================================================
# Styling (METERS radii + pixel clamps)
# =========================================================


st.sidebar.header("Basemap Settings")

base_style = st.sidebar.radio(
    "Basemap style",
    ["Light (Positron)", "Light (Voyager)", "Dark"],
    index=0
)

if base_style == "Light (Positron)":
    map_style = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
elif base_style == "Light (Voyager)":
    map_style = "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json"
else:
    map_style = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"


with st.sidebar.expander("🗺 Map Boundaries & Labels", expanded=False):

    darken_states_overlay = st.checkbox(
        "Darker state boundaries (overlay)", value=True
    )

    state_border_width = st.slider(
        "State border thickness (px)",
        0.1, 2.0, 0.35,
        step=0.25
    )

    state_border_opacity = st.slider(
        "State border darkness",
        0.01, 1.0, 0.1,
        step=0.05
    )

    show_state_labels = st.checkbox(
        "Show state labels (overlay)", value=False
    )

    state_label_size = st.slider(
        "State label size (px)",
        0.1, 10.0, 0.8,
        step=0.1
    )

    state_label_opacity = st.slider(
        "State label opacity",
        0.05, 1.0, 0.4,
        step=0.05
    )

# ---------------------------------------------------------
# Bubble Settings
# ---------------------------------------------------------
st.sidebar.header("Bubble Settings")

show_outlines = st.sidebar.checkbox("Show bubble outlines", value=True)
fill_opacity = st.sidebar.slider("Bubble opacity", 0.05, 1.0, 0.20, step=0.05)
outline_width = st.sidebar.slider("Bubble outline width (px)", 0, 6, 2)

# optional declutter
min_ids_to_show = st.sidebar.slider("Hide groups with # of records < ", 1, 20, 1)
df_agg = df_agg[df_agg["n_ids"] >= min_ids_to_show].copy()

# ---------------------------------------------------------
# Zoom-in Settings  (ADD checkbox here)
# ---------------------------------------------------------
st.sidebar.header("Zoom-in Settings")

zoom_in_mode = st.sidebar.checkbox("Zoom in mode", value=False)

# When Zoom in mode is ON, force radii to minimum possible values
if zoom_in_mode:
    min_px = 0.0   # minimum of slider range
    max_px = 1.0   # minimum of slider range
    # show the sliders but lock them (optional; you can also hide them)
    st.sidebar.slider("Min radius (px)", 0.0, 1.0, min_px, step=0.1, disabled=True)
    st.sidebar.slider("Max radius (px)", 1.0, 10.0, max_px, step=0.5, disabled=True)
else:
    min_px = st.sidebar.slider("Min radius (px)", 0.0, 1.0, 0.3, step=0.1)
    max_px = st.sidebar.slider("Max radius (px)", 1.0, 10.0, 2.0, step=0.5)

# ✅ Warning if user crosses the limit (still useful when not locked)
if min_px >= max_px:
    st.sidebar.warning(
        "⚠️ Min radius must stay smaller than Max radius. "
        "Please lower Min or increase Max."
    )

# Controls how much of the total radius range is used by the first 1..pivot IDs
alpha = st.sidebar.slider("Small-ID emphasis", 0.01, 0.10, 0.01, step=0.01)
pivot = st.sidebar.slider("Pivot # (slow growth after n=)", 1, 200, 5)
k_post = st.sidebar.slider("Post-pivot saturation \n (damp big bubbles)", 50, 800, 50)

# # Damping > 1 makes post-pivot growth even slower
# gamma_post = st.sidebar.slider("Post-pivot damping (Impact Big Bubbles)", 1.0, 6.0, 1.0, step=0.25)

n = df_agg["n_ids"].astype(float).clip(lower=0)

# --- Part 1: fast growth from 0..pivot mapped into [0..alpha] ---
t1 = (n / float(pivot)).clip(0, 1)          # linear in [0,1]
t1 = t1 ** 0.7                               # (optional) boosts small values a bit

# --- Part 2: slow growth for n>pivot mapped into [alpha..1] ---
excess = (n - pivot).clip(lower=0)
t2 = 1.0 - np.exp(-excess / float(max(k_post, 1)))   # saturates
# t2 = t2 ** gamma_post

t = alpha * t1 + (1.0 - alpha) * t2   # combine

df_agg["radius_px"] = min_px + t * (max_px - min_px)

# big first so small remain visible on top
df_agg = df_agg.sort_values("radius_px", ascending=False).copy()



# =========================================================
# Layout: Map + Top Regions
# =========================================================
left, right = st.columns([1.35, 1.0], gap="large")

with left:
    st.subheader("Interactive Bubble Map")

    USA_OPTION = "USA view"

    top_regions = (
        df_agg.groupby("region", dropna=False, observed=True)["n_ids"]
            .sum()
            .sort_values(ascending=False)
            .head(40)
            .index
            .tolist()
    )

    # Use the SAME regions as the bar chart (top_n)
    # (top_n is defined in the right column; replicate here to avoid cross-column ordering issues)
    top_n_map = st.session_state.get("top_n_regions", None)

    # If you want it *exactly* tied to the bar chart slider, store it in session_state (see note below).
    # Otherwise, just compute top_n from the same slider default:
    if top_n_map is None:
        top_n_map = 10  # same default as the bar chart slider

    # compute top regions (by total unique IDs across bins)
    top_regions = (
        df_agg.groupby("region", dropna=False, observed=True)["n_ids"]
            .sum()
            .sort_values(ascending=False)
            .head(int(top_n_map))
            .index
            .tolist()
    )

    zoom_region = st.selectbox(
        "Zoom to region",
        [USA_OPTION] + top_regions,
        key="zoom_region_select"
    )

    # ✅ Filter map points
    if zoom_region == USA_OPTION:
        df_map = df_agg.copy()          # show EVERYTHING again
    else:
        df_map = df_agg[df_agg["region"] == zoom_region].copy()

    # ✅ Center/zoom based on the filtered map data
    if len(df_map):
        center_lat = float(df_map["lat"].mean())
        center_lon = float(df_map["lon"].mean())

        if zoom_region == "USA view":
            zoom = 3.3
        else:
            lat_span = float(df_map["lat"].max() - df_map["lat"].min())
            lon_span = float(df_map["lon"].max() - df_map["lon"].min())
            span = max(lat_span, lon_span)

            if span < 0.15:
                zoom = 10.0
            elif span < 0.35:
                zoom = 9.0
            elif span < 0.75:
                zoom = 8.0
            else:
                zoom = 7.0
    else:
        center_lat, center_lon, zoom = 39.5, -98.35, 3.3    

    tooltip = {
        "html": """
        <div style="font-family: sans-serif;">
        <b>{region}</b><br/>
        ZIP: <b>{zip5}</b><br/>
        Capacity: {cap_bin}<br/>
        Records: <b>{n_ids}</b>
        </div>
        """,
        "style": {"backgroundColor": "white", "color": "black"},
    }

    fill_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_map,
    get_position="[lon, lat]",
    radius_units="pixels",
    get_radius="radius_px",
    filled=True,
    stroked=False,
    get_fill_color="fill_rgb",
    pickable=True,
    opacity=fill_opacity,
    )


    layers = [fill_layer]

    if show_outlines and outline_width > 0:
        outline_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_map,
            get_position="[lon, lat]",
            radius_units="pixels",
            get_radius="radius_px",
            filled=False,
            stroked=True,
            get_line_color="line_rgb",
            line_width_min_pixels=outline_width,
            pickable=False,
            opacity=1.0,
        )

        layers.append(outline_layer)

    if darken_states_overlay:
        states_geo = load_us_states_geojson()

        state_border_layer = pdk.Layer(
            "GeoJsonLayer",
            data=states_geo,
            stroked=True,
            filled=False,
            get_line_color=[40, 40, 40],  # dark gray
            line_width_min_pixels=state_border_width,
            opacity=state_border_opacity,
            pickable=False,
        )

        layers.append(state_border_layer)

    if show_state_labels:
        states_geo = load_us_states_geojson()
        labels_df = load_state_label_points(states_geo)
        labels_df["label"] = labels_df["state_name"].map(STATE_ABBR).fillna(labels_df["state_name"])

        alpha = int(255 * float(state_label_opacity))

        state_label_layer = pdk.Layer(
            "TextLayer",
            data=labels_df,
            get_position="[lon, lat]",
            get_text="label",
            get_size=state_label_size,                 # ✅ knob works
            size_units="pixels",
            get_color=[80, 80, 80, alpha],             # ✅ opacity knob works (RGBA)
            get_outline_color=[255, 255, 255, alpha],  # keep halo consistent
            outline_width=3,
            billboard=True,
            pickable=False,
        )
        layers.append(state_label_layer)


    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom),
        tooltip=tooltip,
        map_style=map_style,
        map_provider="mapbox",
    )
    st.pydeck_chart(deck, use_container_width=True)

    # -----------------------------
    # One-click export (lazy render)
    # -----------------------------
    EXPORT_W, EXPORT_H, EXPORT_SCALE = 1400, 800, 2

    # Signature so we know when the map changed and the old export is stale
    map_signature = (
        zoom_region,
        tuple(bins_selected),
        agg_mode,
        float(min_px), float(max_px),
        float(fill_opacity),
        int(outline_width),
        bool(show_outlines),
        bool(darken_states_overlay),
        float(state_border_width),
        float(state_border_opacity),
        bool(show_state_labels),
        float(state_label_size),
        float(state_label_opacity),
        base_style,   # include anything else that changes appearance
    )

    # If map changed, invalidate previously rendered PNG
    if st.session_state.get("export_signature") != map_signature:
        st.session_state["export_signature"] = map_signature
        st.session_state["export_png"] = None

    colA, colB = st.columns([1, 1])

    with colA:
        if st.button("🔄 Process high-res PNG file", key="render_png_btn"):
            with st.spinner("Rendering high-res PNG..."):
                deck_html = deck.to_html(as_string=True)
                st.session_state["export_png"] = render_deck_png_from_html(
                    deck_html, EXPORT_W, EXPORT_H, EXPORT_SCALE
                )

    with colB:
        png_ready = st.session_state.get("export_png") is not None
        st.download_button(
            "⬇️ Download PNG",
            data=st.session_state.get("export_png") or b"",
            file_name="prospect_geo_map.png",
            mime="image/png",
            disabled=not png_ready,
            key="download_png_btn",
        )

    if st.session_state.get("export_png") is None:
        st.caption("Tip: Click **Render high-res PNG** once you’re happy with the map. Then download.")
    else:
        st.caption("✅ PNG is ready. If you tweak any settings, you’ll need to render again.")



with right:
    st.subheader("Top Regions (stacked by capacity bin)")
    top_n = st.slider("How many regions", 5, 25, 10)

    if "region" in df_agg.columns:
        ctab = (
            df_agg.groupby(["region", "cap_bin"], dropna=False, observed=True)["n_ids"]
                .sum()
                .reset_index(name="count")
        )

        totals = ctab.groupby("region")["count"].sum().sort_values(ascending=False)
        keep_regions = totals.head(top_n).index.tolist()
        ctab = ctab[ctab["region"].isin(keep_regions)].copy()

        ctab["cap_bin"] = pd.Categorical(ctab["cap_bin"], categories=bin_order, ordered=True)

        fig = px.bar(
            ctab.sort_values(["region", "cap_bin"]),
            y="region",
            x="count",
            color="cap_bin",
            orientation="h",
            category_orders={"cap_bin": bin_order, "region": keep_regions},
            color_discrete_map=color_map,
            title="",
        )
        fig.update_layout(
            height=560,
            yaxis_title="",
            xaxis_title="Top Regions",
            legend_title_text="Gift Capacity",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Top Regions chart requires a 'region' field (city/state).")



# =========================================================
# Debugging aids
# =========================================================
# with st.expander("Debug: biggest ZIP/bin groups (by unique IDs)"):
#     st.dataframe(df_agg.sort_values("n_ids", ascending=False).head(50), use_container_width=True)

with st.expander("Top [ZIP + Capacity] Groups"):
    df_debug = (
        df_agg
        .sort_values("n_ids", ascending=False)
        .head(50)
        .reset_index(drop=True)
    )

    df_debug.insert(0, "S.No", range(1, len(df_debug) + 1))

    st.dataframe(df_debug, use_container_width=True, hide_index=True)

with st.expander("Missing ZIP geocodes"):
    base = df_raw.copy()
    base[id_col] = base[id_col].astype(str).str.strip()
    base["zip5"] = base[zip_col].astype(str).str.strip().str.zfill(5)
    base = base[base["zip5"].str.match(r"^\d{5}$", na=False)]
    base = base[base["zip5"] != "00000"]
    geo2 = zip_to_latlon(base["zip5"])
    base = base.merge(geo2, on="zip5", how="left")
    st.write(f"Rows after ZIP cleanup: **{len(base):,}**")
    st.write(f"Rows missing coordinates after ZIP lookup: **{base['lat'].isna().sum():,}**")
    st.dataframe(
        base[base["lat"].isna()][[id_col, zip_col, city_col, state_col]].head(50),
        use_container_width=True
    )
