# import streamlit as st
# import pandas as pd
# import numpy as np
# import pgeocode
# import pydeck as pdk
# import plotly.express as px
# import json
# import requests
# from pathlib import Path
# import tempfile
# import os
# import re
# import threading
# from http.server import HTTPServer, BaseHTTPRequestHandler
# from urllib.parse import urlparse, parse_qs


# # ── Viewport capture: tiny local HTTP server + module-level state ─────────────
# # JS inside the pydeck iframe fires an image-beacon GET request to this server
# # whenever the user pans or zooms the interactive map.  Python reads the latest
# # captured viewport when "Process PNG" is clicked — no extra UI needed.

# @st.cache_resource
# def _get_vp_store() -> dict:
#     """
#     Returns a single mutable dict that persists for the lifetime of the
#     Streamlit process.  Unlike a plain module-level assignment (which is
#     reset to its initialiser on every script rerun), a cache_resource object
#     is created once and shared across all reruns and threads.

#     Values start as None.  They are set to real coordinates only when the
#     user actually pans/zooms the map (onViewStateChange beacon).  The export
#     code falls back to the Python-computed initial_view_state when None.
#     """
#     return {"lat": None, "lon": None, "zoom": None}


# @st.cache_resource
# def _get_hm_vp_store() -> dict:
#     """Separate viewport store for the Census Heat Map (key=hm beacons)."""
#     return {"lat": None, "lon": None, "zoom": None}

# # NOTE: do NOT call _get_vp_store() here at module level.
# # Calling any st.cache_resource function before st.set_page_config() raises
# # StreamlitSetPageConfigMustBeFirstCommandError.
# # We call _get_vp_store() only inside the page body (after set_page_config).


# class _VPHandler(BaseHTTPRequestHandler):
#     """
#     Handles GET /vp?lat=…&lon=…&z=… from the injected JS beacon.

#     Chrome's Private Network Access (PNA) policy blocks requests from
#     localhost:8501 → 127.0.0.1:PORT unless the server handles the CORS
#     OPTIONS preflight AND returns Access-Control-Allow-Private-Network: true.
#     """

#     def _cors(self):
#         self.send_header("Access-Control-Allow-Origin", "*")
#         self.send_header("Access-Control-Allow-Private-Network", "true")
#         self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
#         self.send_header("Access-Control-Allow-Headers", "*")

#     def do_OPTIONS(self):
#         """CORS / PNA preflight – must succeed or Chrome blocks the real GET."""
#         self.send_response(204)
#         self._cors()
#         self.end_headers()

#     def do_GET(self):
#         try:
#             q     = parse_qs(urlparse(self.path).query)
#             key   = q.get("key", ["main"])[0]
#             store = _get_hm_vp_store() if key == "hm" else _get_vp_store()
#             store["lat"]  = float(q["lat"][0])
#             store["lon"]  = float(q["lon"][0])
#             store["zoom"] = float(q["z"][0])
#         except Exception:
#             pass
#         self.send_response(200)
#         self._cors()
#         self.end_headers()

#     def log_message(self, *args):
#         pass  # suppress server log noise


# @st.cache_resource
# def _start_vp_server() -> int:
#     """Start the viewport-capture server once per process; return its port."""
#     import socket
#     s = socket.socket()
#     s.bind(("", 0))
#     port = s.getsockname()[1]
#     s.close()
#     server = HTTPServer(("127.0.0.1", port), _VPHandler)
#     threading.Thread(target=server.serve_forever, daemon=True).start()
#     return port


# def _inject_vp_capture(html: str, port: int) -> str:
#     """
#     Reliably capture the interactive viewport by doing two things:

#     1. Inject a global _vp_send() helper into <head> that fires an
#        XMLHttpRequest to the local server whenever called.  XHR is used
#        instead of fetch/Image because it works in every iframe sandbox
#        configuration with zero CORS complexity.

#     2. Inject onViewStateChange DIRECTLY into the createDeck({...}) call
#        that pydeck emits in its <script> block.  This avoids the fragile
#        window.createDeck monkey-patch that ran before createDeck() was called
#        and silently failed when timing was off.  Because we're inserting the
#        key straight into the props literal, it is guaranteed to be present.

#     No "initial beacon" is fired on render — that would reset the stored
#     coordinates every time Streamlit reruns (e.g. on button click).  Instead
#     the store starts as None and is only written by real user pan/zoom events.
#     """
#     # ── Part 1: helper function in <head> ────────────────────────────────────
#     head_script = (
#         "\n<script>\n"
#         "var _vp_port=" + str(port) + ";\n"
#         "var _vp_timer;\n"
#         "function _vp_send(la,lo,z){\n"
#         "  clearTimeout(_vp_timer);\n"
#         "  _vp_timer=setTimeout(function(){\n"
#         "    try{\n"
#         "      var r=new XMLHttpRequest();\n"
#         "      r.open('GET','http://127.0.0.1:'+_vp_port+\n"
#         "        '/vp?lat='+la+'&lon='+lo+'&z='+z,true);\n"
#         "      r.send();\n"
#         "    }catch(e){}\n"
#         "  },300);\n"
#         "}\n"
#         "</script>\n"
#     )

#     # ── Part 2: handleEvent injected INTO the createDeck({ literal ──────────
#     # In deck.gl 9.0.x (@deck.gl/jupyter-widget), createDeck() does NOT accept
#     # onViewStateChange as a prop.  View-state changes are routed through a
#     # handleEvent(eventName, payload) callback.  The event name is
#     # "deck-view-state-change-event" and the payload IS the viewState object
#     # directly (latitude / longitude / zoom at top level).
#     #
#     # The wiring inside the bundle (Swe function):
#     #   onViewStateChange: ({viewState}) =>
#     #       handleEvent("deck-view-state-change-event", viewState)
#     # …but only when handleEvent is truthy — injecting it enables that wiring.
#     vc_prop = (
#         "handleEvent:function(evtName,data){"
#         "if(evtName==='deck-view-state-change-event'"
#         "&&data&&data.latitude!==undefined)"
#         "_vp_send(data.latitude,data.longitude,data.zoom);"
#         "},"
#     )

#     html = html.replace("</head>", head_script + "</head>", 1)
#     # Insert handleEvent as the first prop of the createDeck call
#     html = html.replace("const deckInstance = createDeck({",
#                         "const deckInstance = createDeck({" + vc_prop, 1)
#     return html


# def _inject_hm_vp_capture(html: str, port: int) -> str:
#     """
#     Same as _inject_vp_capture but sends key=hm so the beacon is routed to
#     _get_hm_vp_store() instead of the main map's _get_vp_store().
#     """
#     head_script = (
#         "\n<script>\n"
#         "var _vp_port=" + str(port) + ";\n"
#         "var _vp_timer;\n"
#         "function _vp_send(la,lo,z){\n"
#         "  clearTimeout(_vp_timer);\n"
#         "  _vp_timer=setTimeout(function(){\n"
#         "    try{\n"
#         "      var r=new XMLHttpRequest();\n"
#         "      r.open('GET','http://127.0.0.1:'+_vp_port+\n"
#         "        '/vp?key=hm&lat='+la+'&lon='+lo+'&z='+z,true);\n"
#         "      r.send();\n"
#         "    }catch(e){}\n"
#         "  },300);\n"
#         "}\n"
#         "</script>\n"
#     )
#     vc_prop = (
#         "handleEvent:function(evtName,data){"
#         "if(evtName==='deck-view-state-change-event'"
#         "&&data&&data.latitude!==undefined)"
#         "_vp_send(data.latitude,data.longitude,data.zoom);"
#         "},"
#     )
#     html = html.replace("</head>", head_script + "</head>", 1)
#     html = html.replace("const deckInstance = createDeck({",
#                         "const deckInstance = createDeck({" + vc_prop, 1)
#     return html


# # ──────────────────────────────────────────────────────────────────────────────
# US_STATES_GEOJSON_URL = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"

# st.set_page_config(page_title="Prospect Geo Maps", layout="wide")
# st.title("Geographical Distribution Map Builder")

# # =========================================================
# # Helpers
# # =========================================================

# STATE_ABBR = {
#     "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA","Colorado":"CO",
#     "Connecticut":"CT","Delaware":"DE","Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID",
#     "Illinois":"IL","Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA",
#     "Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI","Minnesota":"MN",
#     "Mississippi":"MS","Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV",
#     "New Hampshire":"NH","New Jersey":"NJ","New Mexico":"NM","New York":"NY","North Carolina":"NC",
#     "North Dakota":"ND","Ohio":"OH","Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA",
#     "Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD","Tennessee":"TN","Texas":"TX",
#     "Utah":"UT","Vermont":"VT","Virginia":"VA","Washington":"WA","West Virginia":"WV",
#     "Wisconsin":"WI","Wyoming":"WY","District of Columbia":"DC"
# }


# @st.cache_data(show_spinner=False)

# def load_us_states_geojson() -> dict:
#     r = requests.get(US_STATES_GEOJSON_URL, timeout=30)
#     r.raise_for_status()
#     return r.json()

# def load_map_style(url: str) -> dict:
#     r = requests.get(url, timeout=30)
#     r.raise_for_status()
#     return r.json()

# def apply_line_layer_overrides(style: dict, layer_ids: list, color="#2B2B2B", opacity=0.9, width_px=2.0):
#     import json
#     style = json.loads(json.dumps(style))  # deep copy

#     idset = set(layer_ids or [])
#     for layer in style.get("layers", []):
#         if layer.get("id") in idset and layer.get("type") == "line":
#             paint = layer.setdefault("paint", {})
#             layout = layer.setdefault("layout", {})

#             # Force solid, darker boundaries
#             paint["line-color"] = color
#             paint["line-opacity"] = float(opacity)

#             # FORCE constant width (this overrides zoom expressions)
#             paint["line-width"] = float(width_px)

#             # Remove dash if present (dashed borders look "light")
#             if "line-dasharray" in paint:
#                 paint.pop("line-dasharray", None)
#             if "line-dasharray" in layout:
#                 layout.pop("line-dasharray", None)

#             # Reduce blur so lines look crisp/darker
#             paint["line-blur"] = 0

#     return style

# def list_line_layers(style: dict) -> pd.DataFrame:
#     rows = []
#     for lyr in style.get("layers", []):
#         if lyr.get("type") == "line":
#             lid = lyr.get("id", "")
#             src = lyr.get("source", "")
#             s_layer = lyr.get("source-layer", "")
#             rows.append({"id": lid, "source": src, "source_layer": s_layer})
#     return pd.DataFrame(rows)

# def load_excel(file) -> dict:
#     xls = pd.ExcelFile(file)
#     return {s: pd.read_excel(file, sheet_name=s) for s in xls.sheet_names}

# @st.cache_data(show_spinner=False)
# def zip_to_latlon(zip_series: pd.Series) -> pd.DataFrame:
#     z = zip_series.astype(str).str.zfill(5)
#     nomi = pgeocode.Nominatim("us")
#     geo = nomi.query_postal_code(z.tolist())
#     return pd.DataFrame({"zip5": z.values, "lat": geo["latitude"].values, "lon": geo["longitude"].values})

# # def parse_capacity_min(value):
# #     if pd.isna(value):
# #         return np.nan
# #     s = str(value).replace("$", "").replace(",", "").replace("+", "").strip()
# #     try:
# #         return float(s)
# #     except Exception:
# #         return np.nan

# def parse_capacity_min(value):
#     """
#     Converts capacity bins like:
#     $100K, $500K, $1M, $5M, $25M, $5MM
#     into numeric values for correct ordering.
#     """
#     if pd.isna(value):
#         return np.inf

#     s = str(value).strip().upper()

#     # Remove symbols
#     s = s.replace("$", "").replace(",", "").replace("+", "").strip()

#     # Handle "K"
#     if "K" in s:
#         return float(s.replace("K", "")) * 1_000

#     # Handle "M" or "MM"
#     if "MM" in s:
#         return float(s.replace("MM", "")) * 1_000_000

#     if "M" in s:
#         return float(s.replace("M", "")) * 1_000_000

#     # Fallback numeric
#     digits = re.sub(r"[^0-9.]", "", s)
#     if digits == "":
#         return np.inf

#     return float(digits)

# def hex_to_rgb(hex_color: str):
#     h = str(hex_color).lstrip("#")
#     if len(h) != 6:
#         return [153, 153, 153]
#     return [int(h[i:i+2], 16) for i in (0, 2, 4)]

# def build_color_map_stable(all_bins: list) -> dict:
#     """
#     Stable, permanent mapping:
#     - sort bins by numeric minimum (100K lightest -> 5MM darkest)
#     - assign colors in a fixed palette (light blue -> dark blue -> yellow -> light maroon -> dark maroon)
#     - mapping depends ONLY on the *full* list of bins, not filtered data
#     """
#     bins_sorted = sorted(all_bins, key=parse_capacity_min)

#     palette = [
#     "#7CAAC1",  # light cyan-blue (100K)
#     # "#4FD1D9",  # cyan (new, clearer separation)
#     "#257298",  # blue-teal (mid)
#     "#F7BD88",  # light yellow (new)
#     "#7C5F44",  # richer yellow
#     "#F48789",  # light maroon / red
#     "#FF3838",  # dark maroon (highest)
# ]


#     # If more bins than palette, interpolate by repeating (or you can extend palette)
#     return {b: palette[min(i, len(palette) - 1)] for i, b in enumerate(bins_sorted)}

# # def render_deck_png_from_html(html: str, width: int = 2400, height: int = 1600, scale: int = 2) -> bytes:
# #     """
# #     Render a pydeck HTML string into a high-res PNG (cached by html+settings).
# #     Requires Playwright installed + chromium installed.
# #     """
# #     from playwright.sync_api import sync_playwright

# #     with tempfile.TemporaryDirectory() as td:
# #         html_path = Path(td) / "map.html"
# #         html_path.write_text(html, encoding="utf-8")

# #         with sync_playwright() as p:
# #             browser = p.chromium.launch()
# #             page = browser.new_page(
# #                 viewport={"width": int(width), "height": int(height)},
# #                 device_scale_factor=int(scale),
# #             )
# #             page.goto(html_path.as_uri())
# #             page.wait_for_timeout(1500)  # allow tiles/layers to load
# #             png_bytes = page.screenshot(full_page=False)
# #             browser.close()

# #     return png_bytes

# def render_deck_png_from_html(
#     html: str,
#     width: int = 1200,
#     height: int = 700,
#     scale: int = 2,
# ) -> bytes:
#     """Render a pydeck HTML string to a PNG using a headless Playwright browser."""
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

#             # Try common pydeck/deck.gl container selectors (varies by version)
#             selectors = [
#                 "#deckgl-wrapper",
#                 "div#deckgl-wrapper",
#                 "div[data-testid='deckgl-wrapper']",
#                 ".deckgl-wrapper",
#                 "canvas",
#             ]
#             shot = None
#             for sel in selectors:
#                 loc = page.locator(sel)
#                 if loc.count() > 0:
#                     try:
#                         loc.first.wait_for(state="visible", timeout=2000)
#                         shot = loc.first.screenshot()
#                         break
#                     except Exception:
#                         pass

#             if shot is None:
#                 shot = page.screenshot(full_page=False)

#             browser.close()
#             return shot

# def normalize_radii_meters(counts: pd.Series, min_m: float, max_m: float) -> pd.Series:
#     """
#     Radius mapping in METERS (scales naturally with zoom):
#     - winsorize at p95 to prevent one group dominating
#     - log1p scale
#     - normalized to [min_m, max_m]
#     - IMPORTANT: if all values are identical, return min_m (keeps single points small)
#     """
#     c = counts.astype(float).fillna(0.0)

#     if len(c) == 0:
#         return pd.Series([], dtype=float)

#     cap_hi = float(np.quantile(c, 0.95))
#     cap_hi = max(cap_hi, 5.0)
#     c = c.clip(upper=cap_hi)

#     z = np.log1p(c)
#     zmin, zmax = float(z.min()), float(z.max())

#     if zmax <= zmin:
#         return pd.Series(np.full(len(z), min_m), index=counts.index, dtype=float)

#     norm = (z - zmin) / (zmax - zmin)
#     return min_m + norm * (max_m - min_m)

# @st.cache_data(show_spinner=False)
# def load_msa_mapping():
#     """
#     Load the bundled ZIP-to-MSA reference table (zip_to_msa.xlsx).
#     Columns: 'Zip Code', 'MSA'.  Returns ['zip5', 'msa'] or None.
#     """
#     path = Path(__file__).parent / "zip_to_msa.xlsx"
#     if not path.exists():
#         return None
#     raw = pd.read_excel(path, dtype=str)
#     raw.columns = raw.columns.str.strip()
#     zip_c = next((c for c in raw.columns if "zip" in c.lower()), None)
#     msa_c = next((c for c in raw.columns if "msa" in c.lower()), None)
#     if zip_c is None or msa_c is None:
#         return None
#     out = raw[[zip_c, msa_c]].copy().dropna()
#     out["zip5"] = out[zip_c].astype(str).str.strip().str.zfill(5)
#     out["msa"]  = out[msa_c].astype(str).str.strip()
#     return out[["zip5", "msa"]].drop_duplicates("zip5")


# @st.cache_data(show_spinner=False)
# def load_state_label_points(states_geojson: dict) -> pd.DataFrame:
#     """
#     Label point per state using polygon centroid (area-weighted approximation).
#     Falls back to bbox center if geometry is odd.
#     """
#     rows = []
#     for feat in states_geojson.get("features", []):
#         props = feat.get("properties", {}) or {}
#         name = props.get("name", "")

#         geom = feat.get("geometry", {}) or {}
#         gtype = geom.get("type", "")
#         coords = geom.get("coordinates", [])

#         # collect rings
#         rings = []
#         if gtype == "Polygon":
#             rings = coords
#         elif gtype == "MultiPolygon":
#             for poly in coords:
#                 rings.extend(poly)
#         else:
#             continue

#         # Use the largest ring by point count as "main" ring (good enough for states)
#         main = max(rings, key=lambda r: len(r)) if rings else None
#         if not main:
#             continue

#         # centroid of polygon ring (lon/lat) using shoelace (works fine for these shapes)
#         x = [p[0] for p in main]
#         y = [p[1] for p in main]
#         if len(x) < 3:
#             continue

#         a = 0.0
#         cx = 0.0
#         cy = 0.0
#         for i in range(len(x) - 1):
#             cross = x[i] * y[i + 1] - x[i + 1] * y[i]
#             a += cross
#             cx += (x[i] + x[i + 1]) * cross
#             cy += (y[i] + y[i + 1]) * cross

#         if abs(a) < 1e-12:
#             # fallback bbox center
#             lon = (min(x) + max(x)) / 2.0
#             lat = (min(y) + max(y)) / 2.0
#         else:
#             a *= 0.5
#             cx /= (6.0 * a)
#             cy /= (6.0 * a)
#             lon, lat = cx, cy

#         rows.append({"state_name": name, "lon": lon, "lat": lat})

#     return pd.DataFrame(rows)


# # =========================================================
# # Census Heat Map – helpers
# # =========================================================

# CENSUS_VARIABLES = {
#     "Median Household Income":        "Median Income",
#     "Income Per Household":           "IncomePerHousehold",
#     "Population":                     "Population",
#     "Average House Value":            "AverageHouseValue",
#     "Median Household Value":         "Median Household Value",
#     "Labor Participation Rate (%)":   "Labor Participation Rate",
#     "Median Age":                     "Median age",
#     "Owner-Occupied Housing (%)":     "Owner-occupied housing units Percent",
#     "Bachelor's Degree or Higher (%)":"Percent Population 25 years and over Percent bachelors degree or higher",
#     "High School Grad or Higher (%)": "Percent Population 25 years and over Percent high school graduate or higher",
#     "Less Than 9th Grade (%)":        "Percent Population 25 years and over Less than 9th grade",
# }


# @st.cache_data(show_spinner=False)
# def load_census_data():
#     """Load the bundled CensusData.xlsx; returns DataFrame or None."""
#     path = Path(__file__).parent / "CensusData.xlsx"
#     if not path.exists():
#         return None
#     df = pd.read_excel(path, dtype={"ZipCode5Digits": str})
#     df["zip5"] = df["ZipCode5Digits"].astype(str).str.strip().str.zfill(5)
#     df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
#     return df


# @st.cache_data(show_spinner=False)
# def load_zip_geojson_for_states(state_pairs: tuple):
#     """
#     Download and merge ZIP/ZCTA boundary GeoJSON from OpenDataDE GitHub.
#     state_pairs: tuple of (abbr, full_state_name) pairs, e.g. (("IL", "Illinois"),)
#     Filename pattern: {abbr_lower}_{name_lower_underscored}_zip_codes_geo.min.json
#     Returns a FeatureCollection dict or None on failure.
#     """
#     all_features = []
#     for abbr, full_name in state_pairs:
#         name_slug = full_name.lower().replace(" ", "_")
#         filename  = f"{abbr.lower()}_{name_slug}_zip_codes_geo.min.json"
#         url = (
#             "https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON"
#             f"/master/{filename}"
#         )
#         try:
#             r = requests.get(url, timeout=30)
#             r.raise_for_status()
#             all_features.extend(r.json().get("features", []))
#         except Exception:
#             pass
#     return {"type": "FeatureCollection", "features": all_features} if all_features else None


# def get_zip_from_props(props: dict) -> str:
#     """Extract a 5-digit ZIP/ZCTA code from GeoJSON feature properties."""
#     for key in ["ZCTA5CE10", "ZCTA5CE20", "ZCTA5", "ZIP_CODE", "zip", "ZIP5", "ZIPCODE", "postalCode"]:
#         val = props.get(key)
#         if val:
#             return str(val).strip().zfill(5)
#     return ""


# def _lerp(c1, c2, t):
#     """Linearly interpolate between two RGB triplets."""
#     return [int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3)]


# def value_to_rgba(val: float, vmin: float, vmax: float, scale: str, opacity: float) -> list:
#     """Map a scalar value in [vmin, vmax] to an RGBA list for pydeck."""
#     t = float(np.clip((val - vmin) / (vmax - vmin), 0, 1)) if vmax > vmin else 0.5
#     a = int(opacity * 255)

#     stops2 = {
#         "blue_dark": ([210, 235, 255], [10,  50, 150]),
#         "warm":      ([255, 250, 200], [200,  30,  30]),
#         "purple":    ([240, 220, 255], [ 70,  10, 130]),
#         "orange":    ([255, 245, 200], [180,  60,   0]),
#     }
#     if scale in stops2:
#         c1, c2 = stops2[scale]
#         return _lerp(c1, c2, t) + [a]

#     # 3-stop diverging scales through a yellow mid-point
#     mid = [255, 230, 80]
#     if scale == "green_red":
#         c_lo, c_hi = [20, 150, 70], [200, 30, 30]
#     else:  # "red_green"
#         c_lo, c_hi = [200, 30, 30], [20, 150, 70]
#     rgb = _lerp(c_lo, mid, t * 2) if t < 0.5 else _lerp(mid, c_hi, (t - 0.5) * 2)
#     return rgb + [a]


# def fmt_census_val(val, var_label: str) -> str:
#     """Format a census value for tooltip/legend display."""
#     if val is None or (isinstance(val, float) and np.isnan(val)):
#         return "N/A"
#     if any(k in var_label for k in ("Income", "Value", "House")):
#         return f"${val:,.0f}"
#     if any(k in var_label for k in ("%", "Rate", "Higher", "Grade", "Occupied")):
#         return f"{val:.1f}%"
#     if "Age" in var_label:
#         return f"{val:.1f} yrs"
#     return f"{val:,.0f}"


# # =========================================================

# # =========================================================
# # Page tabs — Distribution Map  |  Census Heat Map
# # =========================================================
# _tab_dist, _tab_hm = st.tabs([
#     "📍 Distribution Map",
#     "📊 Census Heat Map",
# ])

# # ── Render Tab 2 (Census Heat Map) first so it is always visible,
# # ── even when no file has been uploaded (Tab 1 may call st.stop()).
# _tab_hm.__enter__()

# # Census Demographics Heat Map
# # =========================================================
# st.header("📊 Census Demographics Heat Map")
# st.caption(
#     "Visualize demographic variables at the ZIP code level for any US state or metro area, "
#     "using the bundled CensusData.xlsx."
# )

# _df_census = load_census_data()

# if _df_census is None:
#     st.warning(
#         "CensusData.xlsx not found. Place it in the same directory as map_app.py to enable this feature."
#     )
# else:
#     _hm_left, _hm_right = st.columns([1, 2.8])

#     with _hm_left:
#         st.subheader("Settings")

#         # ── State selector ────────────────────────────────────────────────────
#         _all_states = sorted(STATE_ABBR.keys())
#         _def_state  = "Illinois" if "Illinois" in _all_states else _all_states[0]
#         hm_state    = st.selectbox("State", _all_states,
#                                    index=_all_states.index(_def_state), key="hm_state")
#         hm_abbr     = STATE_ABBR[hm_state]

#         # ── City / MSA filter (optional) ──────────────────────────────────────
#         _msa_lkp   = load_msa_mapping()
#         _msa_opts  = ["— All ZIPs in state —"]
#         if _msa_lkp is not None:
#             _state_msas = sorted({
#                 m for m in _msa_lkp["msa"].dropna()
#                 if f", {hm_abbr}" in str(m)
#             })
#             _msa_opts += _state_msas
#         hm_msa = st.selectbox("City / MSA (optional)", _msa_opts, key="hm_msa")

#         # ── Variable ──────────────────────────────────────────────────────────
#         hm_var_label = st.selectbox("Variable to map",
#                                     list(CENSUS_VARIABLES.keys()), key="hm_var")
#         hm_var_col   = CENSUS_VARIABLES[hm_var_label]

#         # ── Color scale ───────────────────────────────────────────────────────
#         hm_cscale = st.selectbox(
#             "Color scale",
#             [
#                 "Blue (light→dark)",
#                 "Warm (yellow→red)",
#                 "Green→Yellow→Red",
#                 "Red→Yellow→Green",
#                 "Purple (light→dark)",
#                 "Orange (light→dark)",
#             ],
#             key="hm_cscale",
#         )
#         _cscale_key = {
#             "Blue (light→dark)":   "blue_dark",
#             "Warm (yellow→red)":   "warm",
#             "Green→Yellow→Red":    "green_red",
#             "Red→Yellow→Green":    "red_green",
#             "Purple (light→dark)": "purple",
#             "Orange (light→dark)": "orange",
#         }[hm_cscale]

#         hm_opacity      = st.slider("ZIP fill opacity", 0.1, 1.0, 0.7, step=0.05, key="hm_opacity")
#         hm_border       = st.checkbox("Show ZIP borders", value=True, key="hm_border")
#         hm_show_labels  = st.checkbox("Show ZIP code labels", value=True, key="hm_show_labels")
#         hm_label_size   = st.slider("ZIP label font size", 50, 100, 70, step=1,
#                                     key="hm_label_size",
#                                     disabled=not hm_show_labels)

#         with st.expander("Advanced (percentile clip)"):
#             hm_p_lo = st.slider("Clip low %ile",  0,  20,  2, key="hm_p_lo")
#             hm_p_hi = st.slider("Clip high %ile", 80, 100, 98, key="hm_p_hi")

#         hm_basemap = st.radio("Basemap", ["Light", "Dark"],
#                               horizontal=True, key="hm_basemap")
#         _hm_style  = (
#             "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
#             if hm_basemap == "Light"
#             else "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
#         )

#         _gen_hm = st.button("🗺 Generate Heat Map", type="primary", key="gen_hm")

#     # ── Right column: map output ──────────────────────────────────────────────
#     with _hm_right:
#         # Auto-regenerate when "data" settings (variable, state, MSA, color scale)
#         # change — GeoJSON is cached so only the color pass re-runs (fast).
#         # Continuous sliders (opacity, font size) stay live without full regen.
#         _hp_cur = st.session_state.get("hm_params")
#         _auto_regen = _hp_cur is not None and (
#             _hp_cur.get("var_col")  != hm_var_col  or
#             _hp_cur.get("abbr")     != hm_abbr      or
#             _hp_cur.get("msa")      != hm_msa       or
#             _hp_cur.get("cscale")   != _cscale_key  or
#             _hp_cur.get("p_lo")     != hm_p_lo      or
#             _hp_cur.get("p_hi")     != hm_p_hi
#         )
#         if _gen_hm or _auto_regen:
#             st.session_state["hm_params"] = dict(
#                 state=hm_state, abbr=hm_abbr, msa=hm_msa,
#                 var_col=hm_var_col, var_label=hm_var_label,
#                 cscale=_cscale_key, opacity=hm_opacity,
#                 border=hm_border, show_labels=hm_show_labels,
#                 label_size=hm_label_size,
#                 p_lo=hm_p_lo, p_hi=hm_p_hi,
#                 style=_hm_style,
#             )

#         _hp = st.session_state.get("hm_params")

#         if _hp is None:
#             st.info("Configure settings on the left and click **Generate Heat Map**.")
#         else:
#             with st.spinner(f"Loading ZIP boundary data for {_hp['state']}…"):
#                 _gj = load_zip_geojson_for_states(((_hp["abbr"], _hp["state"]),))

#             if _gj is None:
#                 st.error(
#                     f"Could not load ZIP boundary data for {_hp['state']}. "
#                     "Check your internet connection."
#                 )
#             else:
#                 # ── Collect ZIP codes present in the GeoJSON ──────────────────
#                 _gj_zips = set()
#                 for _f in _gj["features"]:
#                     _z = get_zip_from_props(_f.get("properties", {}) or {})
#                     if _z:
#                         _gj_zips.add(_z)

#                 # ── Filter census data to this state ──────────────────────────
#                 _df_st = _df_census[_df_census["zip5"].isin(_gj_zips)].copy()

#                 # ── Optionally narrow to selected MSA ─────────────────────────
#                 _active_zips = _gj_zips
#                 if _hp["msa"] != "— All ZIPs in state —" and _msa_lkp is not None:
#                     _mz = set(_msa_lkp[_msa_lkp["msa"] == _hp["msa"]]["zip5"].tolist())
#                     _active_zips = _gj_zips & _mz
#                     _df_st = _df_st[_df_st["zip5"].isin(_active_zips)].copy()

#                 _vc = _hp["var_col"]
#                 if _vc not in _df_st.columns:
#                     st.error(f"Column '{_vc}' not found in CensusData.xlsx.")
#                 elif _df_st.empty or _df_st[_vc].replace(0, np.nan).dropna().empty:
#                     st.warning("No census data found for the selected area and variable.")
#                 else:
#                     # ── Value range (winsorized) ──────────────────────────────
#                     _valid = _df_st[_vc].replace(0, np.nan).dropna()
#                     _vmin  = float(_valid.quantile(_hp["p_lo"] / 100))
#                     _vmax  = float(_valid.quantile(_hp["p_hi"] / 100))

#                     _zip_val = {
#                         row["zip5"]: float(row[_vc])
#                         for _, row in _df_st.iterrows()
#                         if pd.notna(row[_vc]) and row[_vc] != 0
#                     }

#                     # ── Enrich GeoJSON features with color + display value ─────
#                     # NOTE: avoid leading-underscore property names — pydeck's
#                     # tooltip template engine does not interpolate them correctly.
#                     _feats_out = []
#                     for _f in _gj["features"]:
#                         _props = dict(_f.get("properties", {}) or {})
#                         _z     = get_zip_from_props(_props)
#                         if _z not in _active_zips:
#                             continue
#                         _v = _zip_val.get(_z)
#                         if _v is not None:
#                             _props["fill_color"] = value_to_rgba(_v, _vmin, _vmax,
#                                                                   _hp["cscale"], _hp["opacity"])
#                             _props["value_str"]  = fmt_census_val(_v, _hp["var_label"])
#                         else:
#                             _props["fill_color"] = [200, 200, 200, 50]
#                             _props["value_str"]  = "N/A"
#                         _props["line_color"] = [60, 60, 60, 180] if _hp["border"] else [0, 0, 0, 0]
#                         _props["zip_code"]   = _z
#                         _feats_out.append({
#                             "type":       "Feature",
#                             "geometry":   _f["geometry"],
#                             "properties": _props,
#                         })

#                     _gj_out = {"type": "FeatureCollection", "features": _feats_out}

#                     # ── Map center & zoom ─────────────────────────────────────
#                     _lat_s = _df_st["Latitude"].dropna()
#                     _lon_s = _df_st["Longitude"].dropna()
#                     if not _lat_s.empty:
#                         _clat  = float(_lat_s.mean())
#                         _clon  = float(_lon_s.mean())
#                         _span  = max(float(_lat_s.max() - _lat_s.min()),
#                                      float(_lon_s.max() - _lon_s.min()))
#                         _czoom = (10.5 if _span < 0.3
#                                   else 9.0 if _span < 0.8
#                                   else 8.0 if _span < 2.0
#                                   else 7.0 if _span < 4.0
#                                   else 6.0 if _span < 8.0
#                                   else 5.0)
#                     else:
#                         _clat, _clon, _czoom = 39.5, -98.35, 4.0

#                     # ── pydeck GeoJsonLayer ───────────────────────────────────
#                     _hm_layer = pdk.Layer(
#                         "GeoJsonLayer",
#                         data=_gj_out,
#                         stroked=True,
#                         filled=True,
#                         get_fill_color="properties.fill_color",
#                         get_line_color="properties.line_color",
#                         line_width_min_pixels=0.5,
#                         pickable=True,
#                         auto_highlight=True,
#                     )

#                     # ── ZIP label TextLayer (uses census centroids) ───────────
#                     _label_df = (
#                         _df_st[_df_st["zip5"].isin(_active_zips)]
#                         [["zip5", "Latitude", "Longitude"]]
#                         .dropna(subset=["Latitude", "Longitude"])
#                         .rename(columns={"Latitude": "lat", "Longitude": "lon"})
#                         .reset_index(drop=True)
#                     )
#                     # Read display params directly from session state for live updates.
#                     _lsz         = int(st.session_state.get("hm_label_size", _hp.get("label_size", 16)))
#                     _show_labels = st.session_state.get("hm_show_labels", _hp.get("show_labels", True))

#                     # Use size_units="meters" so labels scale with zoom automatically:
#                     #   - at state zoom (5-7): labels ≈ 1-4px → invisible (below size_min_pixels)
#                     #   - at city zoom (8-9):  labels grow to size_min_pixels → size_max_pixels
#                     #   - fully zoomed in:     labels capped at size_max_pixels (= user's slider)
#                     # This naturally prevents the black blob without collision detection.
#                     _zip_label_layer = pdk.Layer(
#                         "TextLayer",
#                         data=_label_df,
#                         get_position="[lon, lat]",
#                         get_text="zip5",
#                         get_size=4000,           # nominal 4 km → invisible at state zoom
#                         size_units="meters",
#                         size_min_pixels=8,       # appear only when zoomed in enough
#                         size_max_pixels=_lsz,    # slider controls the max rendered size
#                         get_color=[15, 15, 15, 230],
#                         get_outline_color=[255, 255, 255, 220],
#                         outline_width=2,
#                         font_weight="bold",
#                         billboard=False,
#                         pickable=False,
#                     )

#                     _hm_layers = [_hm_layer]
#                     if _show_labels:
#                         _hm_layers.append(_zip_label_layer)

#                     _hm_deck = pdk.Deck(
#                         layers=_hm_layers,
#                         initial_view_state=pdk.ViewState(
#                             latitude=_clat, longitude=_clon, zoom=_czoom,
#                         ),
#                         tooltip={
#                             "html": (
#                                 "<div style='font-family:sans-serif;padding:4px'>"
#                                 "<b>ZIP {zip_code}</b><br/>"
#                                 "<b>" + _hp["var_label"] + ":</b> {value_str}"
#                                 "</div>"
#                             ),
#                             "style": {"backgroundColor": "white", "color": "black"},
#                         },
#                         map_style=_hp["style"],
#                     )

#                     _area_title = (
#                         _hp["msa"] if _hp["msa"] != "— All ZIPs in state —"
#                         else _hp["state"]
#                     )
#                     st.subheader(f"{_hp['var_label']} — {_area_title}")
#                     # Render via injected HTML so pan/zoom fires viewport beacons
#                     # (key=hm) → _get_hm_vp_store(), used by the PNG export below.
#                     _hm_vp_port  = _start_vp_server()
#                     _hm_disp_html = _hm_deck.to_html(as_string=True)
#                     _hm_disp_html = _inject_hm_vp_capture(_hm_disp_html, _hm_vp_port)
#                     st.components.v1.html(_hm_disp_html, height=520, scrolling=False)

#                     # ── Color legend bar ──────────────────────────────────────
#                     _N = 7
#                     _legend_html = (
#                         "<div style='display:flex;align-items:center;gap:4px;margin:6px 0 2px'>"
#                         f"<span style='font-size:11px;white-space:nowrap'>"
#                         f"{fmt_census_val(_vmin, _hp['var_label'])}</span>"
#                     )
#                     for _i in range(_N):
#                         _t  = _i / (_N - 1)
#                         _vi = _vmin + _t * (_vmax - _vmin)
#                         _ci = value_to_rgba(_vi, _vmin, _vmax, _hp["cscale"], 1.0)
#                         _legend_html += (
#                             f"<div style='flex:1;height:16px;"
#                             f"background:rgb({_ci[0]},{_ci[1]},{_ci[2]});"
#                             f"border-radius:2px'></div>"
#                         )
#                     _legend_html += (
#                         f"<span style='font-size:11px;white-space:nowrap'>"
#                         f"{fmt_census_val(_vmax, _hp['var_label'])}</span>"
#                         "</div>"
#                     )
#                     st.markdown(_legend_html, unsafe_allow_html=True)
#                     st.caption(
#                         f"Gray = no census data | "
#                         f"Clipped to {_hp['p_lo']}–{_hp['p_hi']}th percentile | "
#                         f"{len(_feats_out):,} ZIP codes rendered"
#                     )

#                     # ── PNG export (same Playwright pipeline as main map) ─────
#                     _safe_area = _area_title.replace(" ", "_").replace(",", "").replace("/", "-")
#                     _safe_var  = (_hp["var_label"]
#                                   .replace(" ", "_").replace("(", "").replace(")", "")
#                                   .replace("%", "pct").replace("'", ""))
#                     _dl_stem   = f"census_heatmap_{_safe_area}_{_safe_var}"

#                     # Invalidate cached PNG whenever the map parameters change
#                     _hm_sig = (
#                         _hp["abbr"], _hp["msa"], _hp["var_col"], _hp["cscale"],
#                         _hp["opacity"], _hp["border"], _hp["p_lo"], _hp["p_hi"], _hp["style"],
#                     )
#                     if st.session_state.get("hm_export_sig") != _hm_sig:
#                         st.session_state["hm_export_sig"] = _hm_sig
#                         st.session_state["hm_export_png"] = None

#                     _exp_col1, _exp_col2 = st.columns(2)
#                     with _exp_col1:
#                         if st.button("🔄 Render PNG", key="hm_render_png",
#                                      use_container_width=True):
#                             with st.spinner("Rendering high-res PNG…"):
#                                 # Use whatever viewport the user currently has on screen.
#                                 # Falls back to the Python-computed initial view if the
#                                 # user hasn't panned/zoomed yet.
#                                 _hm_vp   = _get_hm_vp_store()
#                                 _exp_lat  = _hm_vp["lat"]  if _hm_vp["lat"]  is not None else _clat
#                                 _exp_lon  = _hm_vp["lon"]  if _hm_vp["lon"]  is not None else _clon
#                                 _exp_zoom = _hm_vp["zoom"] if _hm_vp["zoom"] is not None else _czoom
#                                 # +1.0 compensates for export canvas (~2×) vs widget width
#                                 _deck_exp = pdk.Deck(
#                                     layers=_hm_layers,
#                                     initial_view_state=pdk.ViewState(
#                                         latitude=_exp_lat,
#                                         longitude=_exp_lon,
#                                         zoom=_exp_zoom + 1.0,
#                                     ),
#                                     map_style=_hp["style"],
#                                 )
#                                 st.session_state["hm_export_png"] = render_deck_png_from_html(
#                                     _deck_exp.to_html(as_string=True), 1400, 800, 2
#                                 )
#                     with _exp_col2:
#                         _hm_png_ready = st.session_state.get("hm_export_png") is not None
#                         st.download_button(
#                             "⬇️ Download PNG",
#                             data=st.session_state.get("hm_export_png") or b"",
#                             file_name=f"{_dl_stem}.png",
#                             mime="image/png",
#                             disabled=not _hm_png_ready,
#                             key="hm_download_png",
#                             use_container_width=True,
#                         )
#                     if not _hm_png_ready:
#                         st.caption("Click **Render PNG** once you're happy with the map, then download.")

# _tab_hm.__exit__(None, None, None)

# # ── Distribution Map tab ─────────────────────────────────────────────────
# _tab_dist.__enter__()

# # Upload + Load
# # =========================================================
# uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
# if not uploaded:
#     st.info("Upload an Excel file to begin.")
#     _tab_dist.__exit__(None, None, None)
#     st.stop()
# sheets = load_excel(uploaded)
# sheet_name = st.selectbox("Select sheet", list(sheets.keys()))
# df_raw = sheets[sheet_name].copy()

# st.subheader("Preview")
# st.dataframe(df_raw.head(25), use_container_width=True)

# # =========================================================
# # Column mapping (defaults to your schema)
# # =========================================================
# st.sidebar.header("Column Mapping")
# cols = df_raw.columns.tolist()

# def idx(colname, fallback=0):
#     return cols.index(colname) if colname in cols else fallback

# id_col = st.sidebar.selectbox("Unique ID column", cols, index=idx("constituent_id", 0))
# zip_col = st.sidebar.selectbox("ZIP 5 column", cols, index=idx("zip_5_digit", idx("zip", 0)))
# bin_col = st.sidebar.selectbox("Capacity bin column", cols, index=idx("gift_capacity_min_bin", 0))
# city_col = st.sidebar.selectbox("City column", cols, index=idx("home_city", 0))
# state_col = st.sidebar.selectbox("State column", cols, index=idx("home_state", 0))

# # =========================================================
# # Prep base data
# # =========================================================
# df = df_raw.copy()

# # Ensure ID is string-like and non-null
# df[id_col] = df[id_col].astype(str).str.strip()
# df = df[df[id_col].notna() & (df[id_col] != "")].copy()

# # ZIP cleanup: ensure 5 digits, drop invalid
# df["zip5"] = df[zip_col].astype(str).str.strip().str.zfill(5)
# df = df[df["zip5"].str.match(r"^\d{5}$", na=False)].copy()
# df = df[df["zip5"] != "00000"].copy()

# # Region + bin
# df["region"] = df[city_col].astype(str).str.strip() + ", " + df[state_col].astype(str).str.strip()
# df["cap_bin"] = df[bin_col].astype(str).str.strip()
# bins_master = sorted(df["cap_bin"].dropna().unique().tolist(), key=parse_capacity_min)
# color_map = build_color_map_stable(bins_master)

# # =========================================================
# # Filters
# # =========================================================
# st.sidebar.header("Filters")
# states = sorted(df[state_col].dropna().astype(str).unique().tolist())
# # Sanitise any previously-loaded state_filter against the current data
# if "p_state_filter" in st.session_state:
#     st.session_state["p_state_filter"] = [
#         s for s in st.session_state["p_state_filter"] if s in states
#     ]
# state_filter = st.sidebar.multiselect("Filter states", states, key="p_state_filter")
# if state_filter:
#     df = df[df[state_col].astype(str).isin(state_filter)].copy()

# # ✅ MASTER bins list BEFORE bin filtering (use the current df after state filter)
# bins_master = sorted(df["cap_bin"].dropna().unique().tolist(), key=parse_capacity_min)

# # ✅ Permanent colors tied to bins_master ONLY
# color_map = build_color_map_stable(bins_master)

# # ── Apply settings loaded from a params JSON file ────────────────────────────
# # Both `states` and `bins_master` are now known, so multiselect params can be
# # validated here. We use the widget key "params_file_uploader" (set by Streamlit
# # before the rerun starts) so this works even though the widget renders later.
# _pfile = st.session_state.get("params_file_uploader")
# if _pfile is not None:
#     try:
#         _pfile.seek(0)
#         _raw_bytes = _pfile.read()
#         _phash = hash(_raw_bytes)
#         if st.session_state.get("_applied_params_hash") != _phash:
#             _lp = json.loads(_raw_bytes).get("params", {})
#             # --- simple (non-data-dependent) params ---
#             for _k in [
#                 "p_agg_mode", "p_base_style",
#                 "p_darken_states", "p_border_width", "p_border_opacity",
#                 "p_show_labels", "p_label_size", "p_label_opacity",
#                 "p_show_outlines", "p_fill_opacity", "p_outline_width",
#                 "p_min_ids", "p_zoom_in_mode", "p_min_px",
#                 "p_alpha", "p_pivot", "p_k_post", "p_top_n",
#             ]:
#                 if _k in _lp:
#                     st.session_state[_k] = _lp[_k]
#             # p_max_px: must be ≥ p_min_px + 0.5
#             if "p_max_px" in _lp:
#                 _min_v = float(st.session_state.get("p_min_px", 0.3))
#                 st.session_state["p_max_px"] = max(float(_lp["p_max_px"]), _min_v + 0.5)
#             # multiselects: validate against current data options
#             if "p_state_filter" in _lp:
#                 st.session_state["p_state_filter"] = [
#                     s for s in _lp["p_state_filter"] if s in states
#                 ]
#             if "p_bins_selected" in _lp:
#                 st.session_state["p_bins_selected"] = [
#                     b for b in _lp["p_bins_selected"] if b in bins_master
#                 ]
#             st.session_state["_applied_params_hash"] = _phash
#             st.rerun()
#     except Exception:
#         pass
# else:
#     # File removed — reset hash so re-uploading the same file works
#     st.session_state.pop("_applied_params_hash", None)

# # ✅ Use master bins for the multiselect so labels/colors stay stable
# # Pre-seed session state so we never need default= on the widget (avoids the
# # "created with default but also set via Session State API" Streamlit warning).
# if "p_bins_selected" not in st.session_state:
#     # First run — select all bins by default
#     st.session_state["p_bins_selected"] = bins_master
# else:
#     # Sanitise any previously-loaded value against the current data
#     st.session_state["p_bins_selected"] = [
#         b for b in st.session_state["p_bins_selected"] if b in bins_master
#     ]
# bins_selected = st.sidebar.multiselect(
#     "Capacity bins to show", bins_master, key="p_bins_selected"
# )
# df = df[df["cap_bin"].isin(bins_selected)].copy()

# # ✅ Display order is just the master order filtered down
# bin_order = [b for b in bins_master if b in bins_selected]

# # Geocode ZIP centroids
# geo = zip_to_latlon(df["zip5"])
# df = df.merge(geo, on="zip5", how="left")
# df = df.dropna(subset=["lat", "lon"]).copy()

# states_for_zoom = (
#     df[state_col]
#     .dropna()
#     .astype(str)
#     .sort_values()
#     .unique()
#     .tolist()
# )

# # =========================================================
# # Aggregation
# # =========================================================

# st.sidebar.header("Aggregation")

# if "p_agg_mode" not in st.session_state:
#     st.session_state["p_agg_mode"] = "ZIP (zip5 + capacity bin)"
# agg_mode = st.sidebar.selectbox(
#     "Map aggregation level",
#     ["ZIP (zip5 + capacity bin)", "City (city + capacity bin)", "MSA (MSA + capacity bin)"],
#     key="p_agg_mode",
# )

# # Load bundled ZIP→MSA reference table once (cached)
# df_zip_msa = load_msa_mapping()

# if agg_mode.startswith("ZIP"):
#     group_cols = ["zip5", "lat", "lon", "region", "cap_bin"]

# elif agg_mode.startswith("MSA") and df_zip_msa is not None:
#     # Merge zip -> MSA, override region with MSA name
#     df = df.merge(df_zip_msa, on="zip5", how="left")
#     df["msa"] = df["msa"].fillna("Unknown MSA")
#     df["region"] = df["msa"]

#     # MSA centroid = mean lat/lon of all constituent ZIPs
#     msa_centroids = (
#         df.groupby("region", dropna=False, observed=True)[["lat", "lon"]]
#           .mean()
#           .reset_index()
#     )
#     df = df.drop(columns=["lat", "lon"]).merge(msa_centroids, on="region", how="left")
#     group_cols = ["region", "lat", "lon", "cap_bin"]

# else:
#     # City mode (also used as fallback if MSA selected but no file uploaded)
#     if agg_mode.startswith("MSA"):
#         st.warning("zip_to_msa.xlsx not found or could not be parsed. Showing City mode instead.")
#     city_centroids = (
#         df.groupby("region", dropna=False, observed=True)[["lat", "lon"]]
#           .mean()
#           .reset_index()
#     )
#     df2 = df.drop(columns=["lat", "lon"]).merge(city_centroids, on="region", how="left")
#     df = df2
#     group_cols = ["region", "lat", "lon", "cap_bin"]

# # COUNT UNIQUE constituent_id per group
# df_agg = (
#     df.groupby(group_cols, dropna=False, observed=True)[id_col]
#       .nunique(dropna=True)
#       .reset_index(name="n_ids")
# )

# # Colors (same mapping for map + bar chart)
# df_agg["color_hex"] = df_agg["cap_bin"].map(color_map).fillna("#999999")
# df_agg["fill_rgb"] = df_agg["color_hex"].apply(hex_to_rgb)
# df_agg["line_rgb"] = df_agg["fill_rgb"]

# # draw order low behind high (stable)
# df_agg["cap_bin"] = pd.Categorical(df_agg["cap_bin"], categories=bin_order, ordered=True)
# df_agg = df_agg.sort_values("cap_bin", ascending=True).copy()
# df_agg_all = df_agg.copy()   # <-- keep full aggregated set for dropdown + camera


# # =========================================================
# # Styling (METERS radii + pixel clamps)
# # =========================================================


# st.sidebar.header("Basemap Settings")

# if "p_base_style" not in st.session_state:
#     st.session_state["p_base_style"] = "Light (Positron)"
# base_style = st.sidebar.radio(
#     "Basemap style",
#     ["Light (Positron)", "Light (Voyager)", "Dark"],
#     key="p_base_style",
# )

# if base_style == "Light (Positron)":
#     map_style = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
# elif base_style == "Light (Voyager)":
#     map_style = "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json"
# else:
#     map_style = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"


# with st.sidebar.expander("🗺 Map Boundaries & Labels", expanded=False):

#     if "p_darken_states" not in st.session_state:
#         st.session_state["p_darken_states"] = True
#     darken_states_overlay = st.checkbox(
#         "Darker state boundaries (overlay)", key="p_darken_states"
#     )

#     if "p_border_width" not in st.session_state:
#         st.session_state["p_border_width"] = 0.35
#     state_border_width = st.slider(
#         "State border thickness (px)",
#         0.1, 2.0,
#         step=0.25, key="p_border_width",
#     )

#     if "p_border_opacity" not in st.session_state:
#         st.session_state["p_border_opacity"] = 0.1
#     state_border_opacity = st.slider(
#         "State border darkness",
#         0.01, 1.0,
#         step=0.05, key="p_border_opacity",
#     )

#     if "p_show_labels" not in st.session_state:
#         st.session_state["p_show_labels"] = False
#     show_state_labels = st.checkbox(
#         "Show state labels (overlay)", key="p_show_labels"
#     )

#     if "p_label_size" not in st.session_state:
#         st.session_state["p_label_size"] = 0.8
#     state_label_size = st.slider(
#         "State label size (px)",
#         0.1, 10.0,
#         step=0.1, key="p_label_size",
#     )

#     if "p_label_opacity" not in st.session_state:
#         st.session_state["p_label_opacity"] = 0.4
#     state_label_opacity = st.slider(
#         "State label opacity",
#         0.05, 1.0,
#         step=0.05, key="p_label_opacity",
#     )

# # ---------------------------------------------------------
# # Bubble Settings
# # ---------------------------------------------------------
# st.sidebar.header("Bubble Settings")

# if "p_show_outlines" not in st.session_state:
#     st.session_state["p_show_outlines"] = True
# show_outlines    = st.sidebar.checkbox("Show bubble outlines", key="p_show_outlines")
# if "p_fill_opacity" not in st.session_state:
#     st.session_state["p_fill_opacity"] = 0.20
# fill_opacity     = st.sidebar.slider("Bubble opacity", 0.05, 1.0, step=0.05, key="p_fill_opacity")
# if "p_outline_width" not in st.session_state:
#     st.session_state["p_outline_width"] = 2
# outline_width    = st.sidebar.slider("Bubble outline width (px)", 0, 6, key="p_outline_width")

# # optional declutter
# if "p_min_ids" not in st.session_state:
#     st.session_state["p_min_ids"] = 1
# min_ids_to_show  = st.sidebar.slider("Hide groups with # of records < ", 1, 20, key="p_min_ids")
# df_agg = df_agg[df_agg["n_ids"] >= min_ids_to_show].copy()

# # ---------------------------------------------------------
# # Zoom-in Settings  (ADD checkbox here)
# # ---------------------------------------------------------
# st.sidebar.header("Zoom-in Settings")

# if "p_zoom_in_mode" not in st.session_state:
#     st.session_state["p_zoom_in_mode"] = False
# zoom_in_mode = st.sidebar.checkbox("Zoom in mode", key="p_zoom_in_mode")

# # When Zoom in mode is ON, force radii to minimum possible values
# if zoom_in_mode:
#     min_px = 0.0   # minimum of slider range
#     max_px = 1.0   # minimum of slider range
#     # show the sliders but lock them
#     st.sidebar.slider("Min radius (px)", 0.0, 1.0, min_px, step=0.1, disabled=True)
#     st.sidebar.slider("Max radius (px)", 1.0, 10.0, max_px, step=0.5, disabled=True)
# else:
#     if "p_min_px" not in st.session_state:
#         st.session_state["p_min_px"] = 0.3
#     min_px = st.sidebar.slider("Min radius (px)", 0.0, 1.0, step=0.1, key="p_min_px")
#     # Clamp any loaded max_px to be above min_px before the slider renders
#     if "p_max_px" in st.session_state:
#         st.session_state["p_max_px"] = max(float(st.session_state["p_max_px"]), float(min_px) + 0.5)
#     else:
#         st.session_state["p_max_px"] = max(2.0, float(min_px) + 0.5)
#     # Max radius lower bound is always at least min_px + one step above it
#     max_px = st.sidebar.slider("Max radius (px)", min_px + 0.5, 10.0, step=0.5, key="p_max_px")

# # Hard clamp: guarantee max_px is strictly greater than min_px at all times
# max_px = max(max_px, min_px + 0.1)

# # Controls how much of the total radius range is used by the first 1..pivot IDs
# if "p_alpha" not in st.session_state:
#     st.session_state["p_alpha"] = 0.01
# alpha  = st.sidebar.slider("Small-ID emphasis", 0.01, 0.10, step=0.01, key="p_alpha")
# if "p_pivot" not in st.session_state:
#     st.session_state["p_pivot"] = 5
# pivot  = st.sidebar.slider("Pivot # (slow growth after n=)", 1, 200, key="p_pivot")
# if "p_k_post" not in st.session_state:
#     st.session_state["p_k_post"] = 50
# k_post = st.sidebar.slider("Post-pivot saturation \n (damp big bubbles)", 50, 800, key="p_k_post")

# # # Damping > 1 makes post-pivot growth even slower
# # gamma_post = st.sidebar.slider("Post-pivot damping (Impact Big Bubbles)", 1.0, 6.0, 1.0, step=0.25)

# n = df_agg["n_ids"].astype(float).clip(lower=0)

# # --- Part 1: fast growth from 0..pivot mapped into [0..alpha] ---
# t1 = (n / float(pivot)).clip(0, 1)          # linear in [0,1]
# t1 = t1 ** 0.7                               # (optional) boosts small values a bit

# # --- Part 2: slow growth for n>pivot mapped into [alpha..1] ---
# excess = (n - pivot).clip(lower=0)
# t2 = 1.0 - np.exp(-excess / float(max(k_post, 1)))   # saturates
# # t2 = t2 ** gamma_post

# t = alpha * t1 + (1.0 - alpha) * t2   # combine

# df_agg["radius_px"] = min_px + t * (max_px - min_px)

# # big first so small remain visible on top
# df_agg = df_agg.sort_values("radius_px", ascending=False).copy()



# # =========================================================
# # ⚙️  Save / Load Settings
# # =========================================================
# st.sidebar.markdown("---")
# st.sidebar.header("⚙️ Save / Load Settings")

# # Keys that participate in save/load
# _PARAM_KEYS = [
#     "p_agg_mode", "p_base_style",
#     "p_darken_states", "p_border_width", "p_border_opacity",
#     "p_show_labels", "p_label_size", "p_label_opacity",
#     "p_show_outlines", "p_fill_opacity", "p_outline_width",
#     "p_min_ids", "p_zoom_in_mode", "p_min_px", "p_max_px",
#     "p_alpha", "p_pivot", "p_k_post",
#     "p_state_filter", "p_bins_selected", "p_top_n",
# ]

# def _build_params_json() -> bytes:
#     """Serialise all current widget values to a JSON byte-string."""
#     params = {}
#     for _k in _PARAM_KEYS:
#         if _k in st.session_state:
#             _v = st.session_state[_k]
#             if hasattr(_v, "item"):                    # numpy scalar → Python native
#                 _v = _v.item()
#             elif isinstance(_v, list):
#                 _v = [x.item() if hasattr(x, "item") else x for x in _v]
#             params[_k] = _v
#     return json.dumps({"version": 1, "params": params}, indent=2).encode("utf-8")

# _settings_stem = Path(uploaded.name).stem
# st.sidebar.download_button(
#     "💾 Save current settings",
#     data=_build_params_json(),
#     file_name=f"{_settings_stem}_settings.json",
#     mime="application/json",
#     key="save_params_btn",
# )
# st.sidebar.file_uploader(
#     "📂 Load settings (.json)", type=["json"], key="params_file_uploader"
# )

# # =========================================================
# # Layout: Map + Top Regions
# # =========================================================
# left, right = st.columns([1.35, 1.0], gap="large")

# with left:
#     st.subheader("Interactive Bubble Map")

#     USA_OPTION = "USA view"

#     top_regions = (
#         df_agg.groupby("region", dropna=False, observed=True)["n_ids"]
#             .sum()
#             .sort_values(ascending=False)
#             .head(40)
#             .index
#             .tolist()
#     )

#     # Use the SAME regions as the bar chart (top_n)
#     # (top_n is defined in the right column; replicate here to avoid cross-column ordering issues)
#     top_n_map = st.session_state.get("top_n_regions", None)

#     # If you want it *exactly* tied to the bar chart slider, store it in session_state (see note below).
#     # Otherwise, just compute top_n from the same slider default:
#     if top_n_map is None:
#         top_n_map = 10  # same default as the bar chart slider

#     # compute top regions (by total unique IDs across bins)
#     top_regions = (
#         df_agg.groupby("region", dropna=False, observed=True)["n_ids"]
#             .sum()
#             .sort_values(ascending=False)
#             .head(int(top_n_map))
#             .index
#             .tolist()
#     )

#     zoom_region = st.selectbox(
#         "Zoom to region",
#         [USA_OPTION] + top_regions,
#         key="zoom_region_select"
#     )

#     # ✅ Filter map points
#     if zoom_region == USA_OPTION:
#         df_map = df_agg.copy()          # show EVERYTHING again
#     else:
#         df_map = df_agg[df_agg["region"] == zoom_region].copy()

#     # ✅ Center/zoom based on the filtered map data
#     if len(df_map):
#         center_lat = float(df_map["lat"].mean())
#         center_lon = float(df_map["lon"].mean())

#         if zoom_region == "USA view":
#             zoom = 3.3
#         else:
#             lat_span = float(df_map["lat"].max() - df_map["lat"].min())
#             lon_span = float(df_map["lon"].max() - df_map["lon"].min())
#             span = max(lat_span, lon_span)

#             if span < 0.15:
#                 zoom = 10.0
#             elif span < 0.35:
#                 zoom = 9.0
#             elif span < 0.75:
#                 zoom = 8.0
#             else:
#                 zoom = 7.0
#     else:
#         center_lat, center_lon, zoom = 39.5, -98.35, 3.3    

#     if agg_mode.startswith("ZIP"):
#         tooltip_html = """
#         <div style="font-family: sans-serif;">
#         <b>{region}</b><br/>
#         ZIP: <b>{zip5}</b><br/>
#         Capacity: {cap_bin}<br/>
#         Records: <b>{n_ids}</b>
#         </div>
#         """
#     else:
#         tooltip_html = """
#         <div style="font-family: sans-serif;">
#         <b>{region}</b><br/>
#         Capacity: {cap_bin}<br/>
#         Records: <b>{n_ids}</b>
#         </div>
#         """
#     tooltip = {
#         "html": tooltip_html,
#         "style": {"backgroundColor": "white", "color": "black"},
#     }

#     # fill_layer = pdk.Layer(
#     # "ScatterplotLayer",
#     # data=df_map,
#     # get_position="[lon, lat]",
#     # radius_units="pixels",
#     # get_radius="radius_px",
#     # filled=True,
#     # stroked=False,
#     # get_fill_color="fill_rgb",
#     # pickable=True,
#     # opacity=fill_opacity,
#     # )


#     # layers = [fill_layer]

#     # if show_outlines and outline_width > 0:
#     #     outline_layer = pdk.Layer(
#     #         "ScatterplotLayer",
#     #         data=df_map,
#     #         get_position="[lon, lat]",
#     #         radius_units="pixels",
#     #         get_radius="radius_px",
#     #         filled=False,
#     #         stroked=True,
#     #         get_line_color="line_rgb",
#     #         line_width_min_pixels=outline_width,
#     #         pickable=False,
#     #         opacity=1.0,
#     #     )

#     #     layers.append(outline_layer)

#     layers = []

#     # draw lower-priority bins first, higher-priority bins last
#     # so higher-priority colors (like red) remain visible on top
#     for b in sorted(bin_order, key=parse_capacity_min):
#         df_bin = df_map[df_map["cap_bin"] == b].copy()
#         if df_bin.empty:
#             continue

#         # keep the existing bubble size logic exactly as-is
#         layers.append(
#             pdk.Layer(
#                 "ScatterplotLayer",
#                 data=df_bin,
#                 get_position="[lon, lat]",
#                 radius_units="pixels",
#                 get_radius="radius_px",
#                 filled=True,
#                 stroked=False,
#                 get_fill_color="fill_rgb",
#                 pickable=True,
#                 opacity=fill_opacity,
#             )
#         )

#         if show_outlines and outline_width > 0:
#             layers.append(
#                 pdk.Layer(
#                     "ScatterplotLayer",
#                     data=df_bin,
#                     get_position="[lon, lat]",
#                     radius_units="pixels",
#                     get_radius="radius_px",
#                     filled=False,
#                     stroked=True,
#                     get_line_color="line_rgb",
#                     line_width_min_pixels=outline_width,
#                     pickable=False,
#                     opacity=1.0,
#                 )
#             )

#     if darken_states_overlay:
#         states_geo = load_us_states_geojson()

#         state_border_layer = pdk.Layer(
#             "GeoJsonLayer",
#             data=states_geo,
#             stroked=True,
#             filled=False,
#             get_line_color=[40, 40, 40],  # dark gray
#             line_width_min_pixels=state_border_width,
#             opacity=state_border_opacity,
#             pickable=False,
#         )

#         layers.append(state_border_layer)

#     if show_state_labels:
#         states_geo = load_us_states_geojson()
#         labels_df = load_state_label_points(states_geo)
#         labels_df["label"] = labels_df["state_name"].map(STATE_ABBR).fillna(labels_df["state_name"])

#         alpha = int(255 * float(state_label_opacity))

#         state_label_layer = pdk.Layer(
#             "TextLayer",
#             data=labels_df,
#             get_position="[lon, lat]",
#             get_text="label",
#             get_size=state_label_size,                 # knob works
#             size_units="pixels",
#             get_color=[80, 80, 80, alpha],             # opacity knob works (RGBA)
#             get_outline_color=[255, 255, 255, alpha],  # keep halo consistent
#             outline_width=3,
#             billboard=True,
#             pickable=False,
#         )
#         layers.append(state_label_layer)


#     # For USA view, use the stored viewport (updated by user pan/zoom) as
#     # initial_view_state so the map does NOT visually reset to full-USA
#     # every time the user clicks "Process PNG" or changes a filter.
#     # On first load (store is None) we fall back to the Python-computed values.
#     if zoom_region == USA_OPTION:
#         _vs = _get_vp_store()
#         _iv_lat  = _vs["lat"]  if _vs["lat"]  is not None else float(center_lat)
#         _iv_lon  = _vs["lon"]  if _vs["lon"]  is not None else float(center_lon)
#         _iv_zoom = _vs["zoom"] if _vs["zoom"] is not None else float(zoom)
#     else:
#         _iv_lat, _iv_lon, _iv_zoom = float(center_lat), float(center_lon), float(zoom)

#     deck = pdk.Deck(
#         layers=layers,
#         initial_view_state=pdk.ViewState(latitude=_iv_lat, longitude=_iv_lon, zoom=_iv_zoom),
#         tooltip=tooltip,
#         map_style=map_style,
#         map_provider="mapbox",
#     )

#     if zoom_region == USA_OPTION:
#         # Render via injected HTML so we can capture the interactive viewport.
#         # onViewStateChange fires on every pan/zoom and sends a fetch() beacon
#         # to the background server, updating the persistent _get_vp_store() dict.
#         _vp_port    = _start_vp_server()
#         _disp_html  = deck.to_html(as_string=True)
#         _disp_html  = _inject_vp_capture(_disp_html, _vp_port)
#         st.components.v1.html(_disp_html, height=520, scrolling=False)
#     else:
#         st.pydeck_chart(deck, use_container_width=True)

#     # -----------------------------
#     # One-click export (lazy render)
#     # -----------------------------
#     EXPORT_W, EXPORT_H, EXPORT_SCALE = 1400, 800, 2
#     import math as _math

#     # ── Compute export viewport ───────────────────────────────────────────────
#     if zoom_region == USA_OPTION:
#         # Use whatever viewport the user currently has on screen.
#         # _get_vp_store() is updated by the JS onViewStateChange beacon.
#         # Values are None until the user actually pans or zooms; fall back to
#         # the Python-computed initial_view_state in that case.
#         _vp = _get_vp_store()
#         exp_center_lat = _vp["lat"]  if _vp["lat"]  is not None else float(center_lat)
#         exp_center_lon = _vp["lon"]  if _vp["lon"]  is not None else float(center_lon)
#         _raw_zoom      = _vp["zoom"] if _vp["zoom"] is not None else float(zoom)
#         # +1.0 compensates for the export canvas being ~2× wider than the widget
#         # (log2(1400/700) = 1.0), keeping the same visible area.
#         exp_zoom_val   = _raw_zoom + 0.4
#     else:
#         # City / region view: viewport is already auto-computed from selected region
#         exp_center_lat = float(center_lat)
#         exp_center_lon = float(center_lon)
#         exp_zoom_val   = float(zoom)

#     # Signature so we know when the map changed and the old export is stale
#     map_signature = (
#         zoom_region,
#         tuple(sorted(state_filter)),
#         tuple(bins_selected),
#         agg_mode,
#         float(min_px), float(max_px),
#         float(fill_opacity),
#         int(outline_width),
#         bool(show_outlines),
#         bool(darken_states_overlay),
#         float(state_border_width),
#         float(state_border_opacity),
#         bool(show_state_labels),
#         float(state_label_size),
#         float(state_label_opacity),
#         base_style,
#     )

#     # If map changed, invalidate previously rendered PNG
#     if st.session_state.get("export_signature") != map_signature:
#         st.session_state["export_signature"] = map_signature
#         st.session_state["export_png"] = None

#     colA, colB = st.columns([1, 1])

#     with colA:
#         if st.button("🔄 Process high-res PNG file", key="render_png_btn"):
#             with st.spinner("Rendering high-res PNG..."):
#                 # exp_center_lat / exp_center_lon / exp_zoom_val already set above
#                 # (either from the override controls or auto-computed)
#                 deck_export = pdk.Deck(
#                     layers=layers,
#                     initial_view_state=pdk.ViewState(
#                         latitude=exp_center_lat,
#                         longitude=exp_center_lon,
#                         zoom=exp_zoom_val,
#                     ),
#                     tooltip=tooltip,
#                     map_style=map_style,
#                     map_provider="mapbox",
#                 )
#                 deck_export_html = deck_export.to_html(as_string=True)
#                 st.session_state["export_png"] = render_deck_png_from_html(
#                     deck_export_html, EXPORT_W, EXPORT_H, EXPORT_SCALE
#                 )

#     with colB:
#         png_ready = st.session_state.get("export_png") is not None
#         _png_fname = f"{Path(uploaded.name).stem}_MAPS_PNG.png"
#         st.download_button(
#             "⬇️ Download PNG",
#             data=st.session_state.get("export_png") or b"",
#             file_name=_png_fname,
#             mime="image/png",
#             disabled=not png_ready,
#             key="download_png_btn",
#         )

#     if st.session_state.get("export_png") is None:
#         st.caption("Tip: Click **Render high-res PNG** once you’re happy with the map. Then download.")
#     else:
#         st.caption("✅ PNG is ready. If you tweak any settings, you’ll need to render again.")



# with right:
#     st.subheader("Top Regions (stacked by capacity bin)")
#     if "p_top_n" not in st.session_state:
#         st.session_state["p_top_n"] = 10
#     top_n = st.slider("How many regions", 5, 25, key="p_top_n")

#     if "region" in df_agg.columns:
#         ctab = (
#             df_agg.groupby(["region", "cap_bin"], dropna=False, observed=True)["n_ids"]
#                 .sum()
#                 .reset_index(name="count")
#         )

#         totals = ctab.groupby("region")["count"].sum().sort_values(ascending=False)
#         keep_regions = totals.head(top_n).index.tolist()

#         # ── Secondary exclusion: remove specific regions from the bar chart only ──
#         # (does NOT affect the interactive map or any other data)
#         with st.expander("Remove regions from bar chart only", expanded=False):
#             exclude_from_chart = st.multiselect(
#                 "Select regions to hide from bar chart (map is unaffected)",
#                 options=keep_regions,
#                 default=[],
#                 key="bar_chart_exclude",
#             )
#         if exclude_from_chart:
#             keep_regions = [r for r in keep_regions if r not in exclude_from_chart]

#         ctab = ctab[ctab["region"].isin(keep_regions)].copy()
#         ctab["cap_bin"] = pd.Categorical(ctab["cap_bin"], categories=bin_order, ordered=True)

#         fig = px.bar(
#             ctab.sort_values(["region", "cap_bin"]),
#             y="region",
#             x="count",
#             color="cap_bin",
#             orientation="h",
#             category_orders={"cap_bin": bin_order, "region": keep_regions},
#             color_discrete_map=color_map,
#             title="",
#         )
#         fig.update_layout(
#             height=560,
#             yaxis_title="",
#             xaxis_title="Top Regions",
#             legend_title_text="Gift Capacity",
#             margin=dict(l=10, r=10, t=10, b=10),
#         )
#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.info("Top Regions chart requires a 'region' field (city/state).")



# # =========================================================
# # Debugging aids
# # =========================================================
# # with st.expander("Debug: biggest ZIP/bin groups (by unique IDs)"):
# #     st.dataframe(df_agg.sort_values("n_ids", ascending=False).head(50), use_container_width=True)

# with st.expander("Top [ZIP + Capacity] Groups"):
#     df_debug = (
#         df_agg
#         .sort_values("n_ids", ascending=False)
#         .head(50)
#         .reset_index(drop=True)
#     )

#     df_debug.insert(0, "S.No", range(1, len(df_debug) + 1))

#     st.dataframe(df_debug, use_container_width=True, hide_index=True)

# with st.expander("Missing ZIP geocodes"):
#     base = df_raw.copy()
#     base[id_col] = base[id_col].astype(str).str.strip()
#     base["zip5"] = base[zip_col].astype(str).str.strip().str.zfill(5)
#     base = base[base["zip5"].str.match(r"^\d{5}$", na=False)]
#     base = base[base["zip5"] != "00000"]
#     geo2 = zip_to_latlon(base["zip5"])
#     base = base.merge(geo2, on="zip5", how="left")
#     st.write(f"Rows after ZIP cleanup: **{len(base):,}**")
#     st.write(f"Rows missing coordinates after ZIP lookup: **{base['lat'].isna().sum():,}**")
#     st.dataframe(
#         base[base["lat"].isna()][[id_col, zip_col, city_col, state_col]].head(50),
#         use_container_width=True
#     )


# # =========================================================

# _tab_dist.__exit__(None, None, None)



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
import hashlib
import streamlit.components.v1 as components
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs


# ── Viewport capture: per-session state + V1 custom component ───────────────

def _get_vp(key: str) -> dict:
    """Per-session viewport state for one map instance."""
    ss_key = f"_viewport_{key}"
    if ss_key not in st.session_state:
        st.session_state[ss_key] = {"lat": None, "lon": None, "zoom": None}
    return st.session_state[ss_key]


def _save_vp(key: str, vp) -> None:
    """Persist viewport returned by the custom component."""
    if not isinstance(vp, dict):
        return
    try:
        lat = float(vp["lat"])
        lon = float(vp["lon"])
        zoom = float(vp["zoom"])
    except Exception:
        return
    st.session_state[f"_viewport_{key}"] = {"lat": lat, "lon": lon, "zoom": zoom}

def _stable_component_hash(parts) -> str:
    """
    Hash only the inputs that should force the iframe to reload.
    Excludes live viewport values so pan/zoom does not cause remount loops.
    """
    return hashlib.sha1(
        json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

def _inject_vp_postmessage(html: str, instance_id: str) -> str:
    """
    Inject a postMessage-based viewport capture hook into pydeck HTML.
    The deck HTML posts its latest pan/zoom state to the parent custom
    component, which forwards it to Python via Streamlit's V1 component bridge.
    """
    head_script = """
<script>
var _vp_timer;
function _vp_send(la,lo,z){
  clearTimeout(_vp_timer);
  _vp_timer=setTimeout(function(){
    try{
      window.parent.postMessage({
        type:'deck_viewport',
        instance_id:'__INSTANCE_ID__',
        viewport:{lat:la,lon:lo,zoom:z}
      }, '*');
    }catch(e){}
  },300);
}
</script>
""".replace("__INSTANCE_ID__", instance_id)

    vc_prop = (
        "handleEvent:function(evtName,data){"
        "if(evtName==='deck-view-state-change-event'"
        "&&data&&data.latitude!==undefined)"
        "_vp_send(data.latitude,data.longitude,data.zoom);"
        "},"
    )

    html = html.replace("</head>", head_script + "</head>", 1)
    html = html.replace(
        "const deckInstance = createDeck({",
        "const deckInstance = createDeck({" + vc_prop,
        1,
    )
    return html


def render_synced_deck(
    deck: pdk.Deck,
    *,
    state_key: str,
    component_key: str,
    render_hash: str,
    height: int = 520,
):
    html = _inject_vp_postmessage(deck.to_html(as_string=True), component_key)
    vp = deck_with_state(
        key=component_key,
        html=html,
        html_hash=render_hash,
        height=height,
        instance_id=component_key,
        default=None,
    )
    _save_vp(state_key, vp)


# ──────────────────────────────────────────────────────────────────────────────
US_STATES_GEOJSON_URL = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"

st.set_page_config(page_title="Prospect Geo Maps", layout="wide")
st.title("Geographical Distribution Map Builder")

_COMPONENT_DIR = (Path(__file__).parent / "deck_sync_component").resolve()
deck_with_state = components.declare_component(
    "deck_with_state",
    path=str(_COMPONENT_DIR),
)

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
    """Render a pydeck HTML string to a PNG using a headless Playwright browser."""
    from playwright.sync_api import sync_playwright

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
                "canvas",
            ]
            shot = None
            for sel in selectors:
                loc = page.locator(sel)
                if loc.count() > 0:
                    try:
                        loc.first.wait_for(state="visible", timeout=2000)
                        shot = loc.first.screenshot()
                        break
                    except Exception:
                        pass

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
def load_msa_mapping():
    """
    Load the bundled ZIP-to-MSA reference table (zip_to_msa.xlsx).
    Columns: 'Zip Code', 'MSA'.  Returns ['zip5', 'msa'] or None.
    """
    path = Path(__file__).parent / "zip_to_msa.xlsx"
    if not path.exists():
        return None
    raw = pd.read_excel(path, dtype=str)
    raw.columns = raw.columns.str.strip()
    zip_c = next((c for c in raw.columns if "zip" in c.lower()), None)
    msa_c = next((c for c in raw.columns if "msa" in c.lower()), None)
    if zip_c is None or msa_c is None:
        return None
    out = raw[[zip_c, msa_c]].copy().dropna()
    out["zip5"] = out[zip_c].astype(str).str.strip().str.zfill(5)
    out["msa"]  = out[msa_c].astype(str).str.strip()
    return out[["zip5", "msa"]].drop_duplicates("zip5")


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
# Census Heat Map – helpers
# =========================================================

CENSUS_VARIABLES = {
    "Median Household Income":        "Median Income",
    "Income Per Household":           "IncomePerHousehold",
    "Population":                     "Population",
    "Average House Value":            "AverageHouseValue",
    "Median Household Value":         "Median Household Value",
    "Labor Participation Rate (%)":   "Labor Participation Rate",
    "Median Age":                     "Median age",
    "Owner-Occupied Housing (%)":     "Owner-occupied housing units Percent",
    "Bachelor's Degree or Higher (%)":"Percent Population 25 years and over Percent bachelors degree or higher",
    "High School Grad or Higher (%)": "Percent Population 25 years and over Percent high school graduate or higher",
    "Less Than 9th Grade (%)":        "Percent Population 25 years and over Less than 9th grade",
}


@st.cache_data(show_spinner=False)
def load_census_data():
    """Load the bundled CensusData.xlsx; returns DataFrame or None."""
    path = Path(__file__).parent / "CensusData.xlsx"
    if not path.exists():
        return None
    df = pd.read_excel(path, dtype={"ZipCode5Digits": str})
    df["zip5"] = df["ZipCode5Digits"].astype(str).str.strip().str.zfill(5)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df


@st.cache_data(show_spinner=False)
def load_zip_geojson_for_states(state_pairs: tuple):
    """
    Download and merge ZIP/ZCTA boundary GeoJSON from OpenDataDE GitHub.
    state_pairs: tuple of (abbr, full_state_name) pairs, e.g. (("IL", "Illinois"),)
    Filename pattern: {abbr_lower}_{name_lower_underscored}_zip_codes_geo.min.json
    Returns a FeatureCollection dict or None on failure.
    """
    all_features = []
    for abbr, full_name in state_pairs:
        name_slug = full_name.lower().replace(" ", "_")
        filename  = f"{abbr.lower()}_{name_slug}_zip_codes_geo.min.json"
        url = (
            "https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON"
            f"/master/{filename}"
        )
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            all_features.extend(r.json().get("features", []))
        except Exception:
            pass
    return {"type": "FeatureCollection", "features": all_features} if all_features else None


def get_zip_from_props(props: dict) -> str:
    """Extract a 5-digit ZIP/ZCTA code from GeoJSON feature properties."""
    for key in ["ZCTA5CE10", "ZCTA5CE20", "ZCTA5", "ZIP_CODE", "zip", "ZIP5", "ZIPCODE", "postalCode"]:
        val = props.get(key)
        if val:
            return str(val).strip().zfill(5)
    return ""


def _lerp(c1, c2, t):
    """Linearly interpolate between two RGB triplets."""
    return [int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3)]


def value_to_rgba(val: float, vmin: float, vmax: float, scale: str, opacity: float) -> list:
    """Map a scalar value in [vmin, vmax] to an RGBA list for pydeck."""
    t = float(np.clip((val - vmin) / (vmax - vmin), 0, 1)) if vmax > vmin else 0.5
    a = int(opacity * 255)

    stops2 = {
        "blue_dark": ([210, 235, 255], [10,  50, 150]),
        "warm":      ([255, 250, 200], [200,  30,  30]),
        "purple":    ([240, 220, 255], [ 70,  10, 130]),
        "orange":    ([255, 245, 200], [180,  60,   0]),
    }
    if scale in stops2:
        c1, c2 = stops2[scale]
        return _lerp(c1, c2, t) + [a]

    # 3-stop diverging scales through a yellow mid-point
    mid = [255, 230, 80]
    if scale == "green_red":
        c_lo, c_hi = [20, 150, 70], [200, 30, 30]
    else:  # "red_green"
        c_lo, c_hi = [200, 30, 30], [20, 150, 70]
    rgb = _lerp(c_lo, mid, t * 2) if t < 0.5 else _lerp(mid, c_hi, (t - 0.5) * 2)
    return rgb + [a]


def fmt_census_val(val, var_label: str) -> str:
    """Format a census value for tooltip/legend display."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if any(k in var_label for k in ("Income", "Value", "House")):
        return f"${val:,.0f}"
    if any(k in var_label for k in ("%", "Rate", "Higher", "Grade", "Occupied")):
        return f"{val:.1f}%"
    if "Age" in var_label:
        return f"{val:.1f} yrs"
    return f"{val:,.0f}"


# =========================================================

# =========================================================
# Page tabs — Distribution Map  |  Census Heat Map
# =========================================================
_tab_dist, _tab_hm = st.tabs([
    "📍 Distribution Map",
    "📊 Census Heat Map",
])

# ── Render Tab 2 (Census Heat Map) first so it is always visible,
# ── even when no file has been uploaded (Tab 1 may call st.stop()).
_tab_hm.__enter__()

# Census Demographics Heat Map
# =========================================================
st.header("📊 Census Demographics Heat Map")
st.caption(
    "Visualize demographic variables at the ZIP code level for any US state or metro area, "
    "using the bundled CensusData.xlsx."
)

_df_census = load_census_data()

if _df_census is None:
    st.warning(
        "CensusData.xlsx not found. Place it in the same directory as map_app.py to enable this feature."
    )
else:
    _hm_left, _hm_right = st.columns([1, 2.8])

    with _hm_left:
        st.subheader("Settings")

        # ── State selector ────────────────────────────────────────────────────
        _all_states = sorted(STATE_ABBR.keys())
        _def_state  = "Illinois" if "Illinois" in _all_states else _all_states[0]
        hm_state    = st.selectbox("State", _all_states,
                                   index=_all_states.index(_def_state), key="hm_state")
        hm_abbr     = STATE_ABBR[hm_state]

        # ── City / MSA filter (optional) ──────────────────────────────────────
        _msa_lkp   = load_msa_mapping()
        _msa_opts  = ["— All ZIPs in state —"]
        if _msa_lkp is not None:
            _state_msas = sorted({
                m for m in _msa_lkp["msa"].dropna()
                if f", {hm_abbr}" in str(m)
            })
            _msa_opts += _state_msas
        hm_msa = st.selectbox("City / MSA (optional)", _msa_opts, key="hm_msa")

        # ── Variable ──────────────────────────────────────────────────────────
        hm_var_label = st.selectbox("Variable to map",
                                    list(CENSUS_VARIABLES.keys()), key="hm_var")
        hm_var_col   = CENSUS_VARIABLES[hm_var_label]

        # ── Color scale ───────────────────────────────────────────────────────
        hm_cscale = st.selectbox(
            "Color scale",
            [
                "Blue (light→dark)",
                "Warm (yellow→red)",
                "Green→Yellow→Red",
                "Red→Yellow→Green",
                "Purple (light→dark)",
                "Orange (light→dark)",
            ],
            key="hm_cscale",
        )
        _cscale_key = {
            "Blue (light→dark)":   "blue_dark",
            "Warm (yellow→red)":   "warm",
            "Green→Yellow→Red":    "green_red",
            "Red→Yellow→Green":    "red_green",
            "Purple (light→dark)": "purple",
            "Orange (light→dark)": "orange",
        }[hm_cscale]

        hm_opacity      = st.slider("ZIP fill opacity", 0.1, 1.0, 0.7, step=0.05, key="hm_opacity")
        hm_border       = st.checkbox("Show ZIP borders", value=True, key="hm_border")
        hm_show_labels  = st.checkbox("Show ZIP code labels", value=True, key="hm_show_labels")
        hm_label_size   = st.slider("ZIP label font size", 50, 100, 70, step=1,
                                    key="hm_label_size",
                                    disabled=not hm_show_labels)

        with st.expander("Advanced (percentile clip)"):
            hm_p_lo = st.slider("Clip low %ile",  0,  20,  2, key="hm_p_lo")
            hm_p_hi = st.slider("Clip high %ile", 80, 100, 98, key="hm_p_hi")

        hm_basemap = st.radio("Basemap", ["Light", "Dark"],
                              horizontal=True, key="hm_basemap")
        _hm_style  = (
            "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
            if hm_basemap == "Light"
            else "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
        )

        _gen_hm = st.button("🗺 Generate Heat Map", type="primary", key="gen_hm")

    # ── Right column: map output ──────────────────────────────────────────────
    with _hm_right:
        # Auto-regenerate when "data" settings (variable, state, MSA, color scale)
        # change — GeoJSON is cached so only the color pass re-runs (fast).
        # Continuous sliders (opacity, font size) stay live without full regen.
        _hp_cur = st.session_state.get("hm_params")
        _auto_regen = _hp_cur is not None and (
            _hp_cur.get("var_col")  != hm_var_col  or
            _hp_cur.get("abbr")     != hm_abbr      or
            _hp_cur.get("msa")      != hm_msa       or
            _hp_cur.get("cscale")   != _cscale_key  or
            _hp_cur.get("p_lo")     != hm_p_lo      or
            _hp_cur.get("p_hi")     != hm_p_hi
        )
        if _gen_hm or _auto_regen:
            st.session_state["hm_params"] = dict(
                state=hm_state, abbr=hm_abbr, msa=hm_msa,
                var_col=hm_var_col, var_label=hm_var_label,
                cscale=_cscale_key, opacity=hm_opacity,
                border=hm_border, show_labels=hm_show_labels,
                label_size=hm_label_size,
                p_lo=hm_p_lo, p_hi=hm_p_hi,
                style=_hm_style,
            )

        _hp = st.session_state.get("hm_params")

        if _hp is None:
            st.info("Configure settings on the left and click **Generate Heat Map**.")
        else:
            with st.spinner(f"Loading ZIP boundary data for {_hp['state']}…"):
                _gj = load_zip_geojson_for_states(((_hp["abbr"], _hp["state"]),))

            if _gj is None:
                st.error(
                    f"Could not load ZIP boundary data for {_hp['state']}. "
                    "Check your internet connection."
                )
            else:
                # ── Collect ZIP codes present in the GeoJSON ──────────────────
                _gj_zips = set()
                for _f in _gj["features"]:
                    _z = get_zip_from_props(_f.get("properties", {}) or {})
                    if _z:
                        _gj_zips.add(_z)

                # ── Filter census data to this state ──────────────────────────
                _df_st = _df_census[_df_census["zip5"].isin(_gj_zips)].copy()

                # ── Optionally narrow to selected MSA ─────────────────────────
                _active_zips = _gj_zips
                if _hp["msa"] != "— All ZIPs in state —" and _msa_lkp is not None:
                    _mz = set(_msa_lkp[_msa_lkp["msa"] == _hp["msa"]]["zip5"].tolist())
                    _active_zips = _gj_zips & _mz
                    _df_st = _df_st[_df_st["zip5"].isin(_active_zips)].copy()

                _vc = _hp["var_col"]
                if _vc not in _df_st.columns:
                    st.error(f"Column '{_vc}' not found in CensusData.xlsx.")
                elif _df_st.empty or _df_st[_vc].replace(0, np.nan).dropna().empty:
                    st.warning("No census data found for the selected area and variable.")
                else:
                    # ── Value range (winsorized) ──────────────────────────────
                    _valid = _df_st[_vc].replace(0, np.nan).dropna()
                    _vmin  = float(_valid.quantile(_hp["p_lo"] / 100))
                    _vmax  = float(_valid.quantile(_hp["p_hi"] / 100))

                    _zip_val = {
                        row["zip5"]: float(row[_vc])
                        for _, row in _df_st.iterrows()
                        if pd.notna(row[_vc]) and row[_vc] != 0
                    }

                    # ── Enrich GeoJSON features with color + display value ─────
                    # NOTE: avoid leading-underscore property names — pydeck's
                    # tooltip template engine does not interpolate them correctly.
                    _feats_out = []
                    for _f in _gj["features"]:
                        _props = dict(_f.get("properties", {}) or {})
                        _z     = get_zip_from_props(_props)
                        if _z not in _active_zips:
                            continue
                        _v = _zip_val.get(_z)
                        if _v is not None:
                            _props["fill_color"] = value_to_rgba(_v, _vmin, _vmax,
                                                                  _hp["cscale"], _hp["opacity"])
                            _props["value_str"]  = fmt_census_val(_v, _hp["var_label"])
                        else:
                            _props["fill_color"] = [200, 200, 200, 50]
                            _props["value_str"]  = "N/A"
                        _props["line_color"] = [60, 60, 60, 180] if _hp["border"] else [0, 0, 0, 0]
                        _props["zip_code"]   = _z
                        _feats_out.append({
                            "type":       "Feature",
                            "geometry":   _f["geometry"],
                            "properties": _props,
                        })

                    _gj_out = {"type": "FeatureCollection", "features": _feats_out}

                    # ── Map center & zoom ─────────────────────────────────────
                    _lat_s = _df_st["Latitude"].dropna()
                    _lon_s = _df_st["Longitude"].dropna()
                    if not _lat_s.empty:
                        _clat  = float(_lat_s.mean())
                        _clon  = float(_lon_s.mean())
                        _span  = max(float(_lat_s.max() - _lat_s.min()),
                                     float(_lon_s.max() - _lon_s.min()))
                        _czoom = (10.5 if _span < 0.3
                                  else 9.0 if _span < 0.8
                                  else 8.0 if _span < 2.0
                                  else 7.0 if _span < 4.0
                                  else 6.0 if _span < 8.0
                                  else 5.0)
                    else:
                        _clat, _clon, _czoom = 39.5, -98.35, 4.0

                    # ── pydeck GeoJsonLayer ───────────────────────────────────
                    _hm_layer = pdk.Layer(
                        "GeoJsonLayer",
                        data=_gj_out,
                        stroked=True,
                        filled=True,
                        get_fill_color="properties.fill_color",
                        get_line_color="properties.line_color",
                        line_width_min_pixels=0.5,
                        pickable=True,
                        auto_highlight=True,
                    )

                    # ── ZIP label TextLayer (uses census centroids) ───────────
                    _label_df = (
                        _df_st[_df_st["zip5"].isin(_active_zips)]
                        [["zip5", "Latitude", "Longitude"]]
                        .dropna(subset=["Latitude", "Longitude"])
                        .rename(columns={"Latitude": "lat", "Longitude": "lon"})
                        .reset_index(drop=True)
                    )
                    # Read display params directly from session state for live updates.
                    _lsz         = int(st.session_state.get("hm_label_size", _hp.get("label_size", 16)))
                    _show_labels = st.session_state.get("hm_show_labels", _hp.get("show_labels", True))

                    # Use size_units="meters" so labels scale with zoom automatically:
                    #   - at state zoom (5-7): labels ≈ 1-4px → invisible (below size_min_pixels)
                    #   - at city zoom (8-9):  labels grow to size_min_pixels → size_max_pixels
                    #   - fully zoomed in:     labels capped at size_max_pixels (= user's slider)
                    # This naturally prevents the black blob without collision detection.
                    _zip_label_layer = pdk.Layer(
                        "TextLayer",
                        data=_label_df,
                        get_position="[lon, lat]",
                        get_text="zip5",
                        get_size=4000,           # nominal 4 km → invisible at state zoom
                        size_units="meters",
                        size_min_pixels=8,       # appear only when zoomed in enough
                        size_max_pixels=_lsz,    # slider controls the max rendered size
                        get_color=[15, 15, 15, 230],
                        get_outline_color=[255, 255, 255, 220],
                        outline_width=2,
                        font_weight="bold",
                        billboard=False,
                        pickable=False,
                    )

                    _hm_layers = [_hm_layer]
                    if _show_labels:
                        _hm_layers.append(_zip_label_layer)

                    _hm_vp_key = f"hm_vp::{_hp['abbr']}::{_hp['msa']}"
                    _hm_vs = _get_vp(_hm_vp_key)
                    _hm_iv_lat = _hm_vs["lat"] if _hm_vs["lat"] is not None else _clat
                    _hm_iv_lon = _hm_vs["lon"] if _hm_vs["lon"] is not None else _clon
                    _hm_iv_zoom = _hm_vs["zoom"] if _hm_vs["zoom"] is not None else _czoom

                    _hm_deck = pdk.Deck(
                        layers=_hm_layers,
                        initial_view_state=pdk.ViewState(
                            latitude=_hm_iv_lat, longitude=_hm_iv_lon, zoom=_hm_iv_zoom,
                        ),
                        tooltip={
                            "html": (
                                "<div style='font-family:sans-serif;padding:4px'>"
                                "<b>ZIP {zip_code}</b><br/>"
                                "<b>" + _hp["var_label"] + ":</b> {value_str}"
                                "</div>"
                            ),
                            "style": {"backgroundColor": "white", "color": "black"},
                        },
                        map_style=_hp["style"],
                    )

                    _area_title = (
                        _hp["msa"] if _hp["msa"] != "— All ZIPs in state —"
                        else _hp["state"]
                    )
                    st.subheader(f"{_hp['var_label']} — {_area_title}")
                    _hm_render_hash = _stable_component_hash({
                        "abbr": _hp["abbr"],
                        "msa": _hp["msa"],
                        "var_col": _hp["var_col"],
                        "var_label": _hp["var_label"],
                        "cscale": _hp["cscale"],
                        "opacity": float(_hp["opacity"]),
                        "border": bool(_hp["border"]),
                        "show_labels": bool(_show_labels),
                        "label_size": int(_lsz),
                        "p_lo": int(_hp["p_lo"]),
                        "p_hi": int(_hp["p_hi"]),
                        "style": _hp["style"],
                        "feature_count": len(_feats_out),
                    })
                    
                    render_synced_deck(
                        _hm_deck,
                        state_key="hm_vp",
                        component_key="hm_deck",
                        render_hash=_hm_render_hash,
                        height=520,
                    )
                    # ── Color legend bar ──────────────────────────────────────
                    _N = 7
                    _legend_html = (
                        "<div style='display:flex;align-items:center;gap:4px;margin:6px 0 2px'>"
                        f"<span style='font-size:11px;white-space:nowrap'>"
                        f"{fmt_census_val(_vmin, _hp['var_label'])}</span>"
                    )
                    for _i in range(_N):
                        _t  = _i / (_N - 1)
                        _vi = _vmin + _t * (_vmax - _vmin)
                        _ci = value_to_rgba(_vi, _vmin, _vmax, _hp["cscale"], 1.0)
                        _legend_html += (
                            f"<div style='flex:1;height:16px;"
                            f"background:rgb({_ci[0]},{_ci[1]},{_ci[2]});"
                            f"border-radius:2px'></div>"
                        )
                    _legend_html += (
                        f"<span style='font-size:11px;white-space:nowrap'>"
                        f"{fmt_census_val(_vmax, _hp['var_label'])}</span>"
                        "</div>"
                    )
                    st.markdown(_legend_html, unsafe_allow_html=True)
                    st.caption(
                        f"Gray = no census data | "
                        f"Clipped to {_hp['p_lo']}–{_hp['p_hi']}th percentile | "
                        f"{len(_feats_out):,} ZIP codes rendered"
                    )

                    # ── PNG export (same Playwright pipeline as main map) ─────
                    _safe_area = _area_title.replace(" ", "_").replace(",", "").replace("/", "-")
                    _safe_var  = (_hp["var_label"]
                                  .replace(" ", "_").replace("(", "").replace(")", "")
                                  .replace("%", "pct").replace("'", ""))
                    _dl_stem   = f"census_heatmap_{_safe_area}_{_safe_var}"

                    # Invalidate cached PNG whenever the map parameters change
                    _hm_sig = (
                        _hp["abbr"], _hp["msa"], _hp["var_col"], _hp["cscale"],
                        _hp["opacity"], _hp["border"], _hp["p_lo"], _hp["p_hi"], _hp["style"],
                    )
                    if st.session_state.get("hm_export_sig") != _hm_sig:
                        st.session_state["hm_export_sig"] = _hm_sig
                        st.session_state["hm_export_png"] = None

                    _exp_col1, _exp_col2 = st.columns(2)
                    with _exp_col1:
                        if st.button("🔄 Render PNG", key="hm_render_png",
                                     use_container_width=True):
                            with st.spinner("Rendering high-res PNG…"):
                                # Use whatever viewport the user currently has on screen.
                                # Falls back to the Python-computed initial view if the
                                # user hasn't panned/zoomed yet.
                                _hm_vp   = _get_vp(_hm_vp_key)
                                _exp_lat  = _hm_vp["lat"]  if _hm_vp["lat"]  is not None else _clat
                                _exp_lon  = _hm_vp["lon"]  if _hm_vp["lon"]  is not None else _clon
                                _exp_zoom = _hm_vp["zoom"] if _hm_vp["zoom"] is not None else _czoom
                                # +1.0 compensates for export canvas (~2×) vs widget width
                                _deck_exp = pdk.Deck(
                                    layers=_hm_layers,
                                    initial_view_state=pdk.ViewState(
                                        latitude=_exp_lat,
                                        longitude=_exp_lon,
                                        zoom=_exp_zoom + 1.0,
                                    ),
                                    map_style=_hp["style"],
                                )
                                st.session_state["hm_export_png"] = render_deck_png_from_html(
                                    _deck_exp.to_html(as_string=True), 1400, 800, 2
                                )
                    with _exp_col2:
                        _hm_png_ready = st.session_state.get("hm_export_png") is not None
                        st.download_button(
                            "⬇️ Download PNG",
                            data=st.session_state.get("hm_export_png") or b"",
                            file_name=f"{_dl_stem}.png",
                            mime="image/png",
                            disabled=not _hm_png_ready,
                            key="hm_download_png",
                            use_container_width=True,
                        )
                    if not _hm_png_ready:
                        st.caption("Click **Render PNG** once you're happy with the map, then download.")

_tab_hm.__exit__(None, None, None)

# ── Distribution Map tab ─────────────────────────────────────────────────
_tab_dist.__enter__()

# Upload + Load
# =========================================================
uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
if not uploaded:
    st.info("Upload an Excel file to begin.")
    _tab_dist.__exit__(None, None, None)
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
# Sanitise any previously-loaded state_filter against the current data
if "p_state_filter" in st.session_state:
    st.session_state["p_state_filter"] = [
        s for s in st.session_state["p_state_filter"] if s in states
    ]
state_filter = st.sidebar.multiselect("Filter states", states, key="p_state_filter")
if state_filter:
    df = df[df[state_col].astype(str).isin(state_filter)].copy()

# ✅ MASTER bins list BEFORE bin filtering (use the current df after state filter)
bins_master = sorted(df["cap_bin"].dropna().unique().tolist(), key=parse_capacity_min)

# ✅ Permanent colors tied to bins_master ONLY
color_map = build_color_map_stable(bins_master)

# ── Apply settings loaded from a params JSON file ────────────────────────────
# Both `states` and `bins_master` are now known, so multiselect params can be
# validated here. We use the widget key "params_file_uploader" (set by Streamlit
# before the rerun starts) so this works even though the widget renders later.
_pfile = st.session_state.get("params_file_uploader")
if _pfile is not None:
    try:
        _pfile.seek(0)
        _raw_bytes = _pfile.read()
        _phash = hash(_raw_bytes)
        if st.session_state.get("_applied_params_hash") != _phash:
            _lp = json.loads(_raw_bytes).get("params", {})
            # --- simple (non-data-dependent) params ---
            for _k in [
                "p_agg_mode", "p_base_style",
                "p_darken_states", "p_border_width", "p_border_opacity",
                "p_show_labels", "p_label_size", "p_label_opacity",
                "p_show_outlines", "p_fill_opacity", "p_outline_width",
                "p_min_ids", "p_zoom_in_mode", "p_min_px",
                "p_alpha", "p_pivot", "p_k_post", "p_top_n",
            ]:
                if _k in _lp:
                    st.session_state[_k] = _lp[_k]
            # p_max_px: must be ≥ p_min_px + 0.5
            if "p_max_px" in _lp:
                _min_v = float(st.session_state.get("p_min_px", 0.3))
                st.session_state["p_max_px"] = max(float(_lp["p_max_px"]), _min_v + 0.5)
            # multiselects: validate against current data options
            if "p_state_filter" in _lp:
                st.session_state["p_state_filter"] = [
                    s for s in _lp["p_state_filter"] if s in states
                ]
            if "p_bins_selected" in _lp:
                st.session_state["p_bins_selected"] = [
                    b for b in _lp["p_bins_selected"] if b in bins_master
                ]
            st.session_state["_applied_params_hash"] = _phash
            st.rerun()
    except Exception:
        pass
else:
    # File removed — reset hash so re-uploading the same file works
    st.session_state.pop("_applied_params_hash", None)

# ✅ Use master bins for the multiselect so labels/colors stay stable
# Pre-seed session state so we never need default= on the widget (avoids the
# "created with default but also set via Session State API" Streamlit warning).
if "p_bins_selected" not in st.session_state:
    # First run — select all bins by default
    st.session_state["p_bins_selected"] = bins_master
else:
    # Sanitise any previously-loaded value against the current data
    st.session_state["p_bins_selected"] = [
        b for b in st.session_state["p_bins_selected"] if b in bins_master
    ]
bins_selected = st.sidebar.multiselect(
    "Capacity bins to show", bins_master, key="p_bins_selected"
)
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

if "p_agg_mode" not in st.session_state:
    st.session_state["p_agg_mode"] = "ZIP (zip5 + capacity bin)"
agg_mode = st.sidebar.selectbox(
    "Map aggregation level",
    ["ZIP (zip5 + capacity bin)", "City (city + capacity bin)", "MSA (MSA + capacity bin)"],
    key="p_agg_mode",
)

# Load bundled ZIP→MSA reference table once (cached)
df_zip_msa = load_msa_mapping()

if agg_mode.startswith("ZIP"):
    group_cols = ["zip5", "lat", "lon", "region", "cap_bin"]

elif agg_mode.startswith("MSA") and df_zip_msa is not None:
    # Merge zip -> MSA, override region with MSA name
    df = df.merge(df_zip_msa, on="zip5", how="left")
    df["msa"] = df["msa"].fillna("Unknown MSA")
    df["region"] = df["msa"]

    # MSA centroid = mean lat/lon of all constituent ZIPs
    msa_centroids = (
        df.groupby("region", dropna=False, observed=True)[["lat", "lon"]]
          .mean()
          .reset_index()
    )
    df = df.drop(columns=["lat", "lon"]).merge(msa_centroids, on="region", how="left")
    group_cols = ["region", "lat", "lon", "cap_bin"]

else:
    # City mode (also used as fallback if MSA selected but no file uploaded)
    if agg_mode.startswith("MSA"):
        st.warning("zip_to_msa.xlsx not found or could not be parsed. Showing City mode instead.")
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


# =========================================================
# Styling (METERS radii + pixel clamps)
# =========================================================


st.sidebar.header("Basemap Settings")

if "p_base_style" not in st.session_state:
    st.session_state["p_base_style"] = "Light (Positron)"
base_style = st.sidebar.radio(
    "Basemap style",
    ["Light (Positron)", "Light (Voyager)", "Dark"],
    key="p_base_style",
)

if base_style == "Light (Positron)":
    map_style = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
elif base_style == "Light (Voyager)":
    map_style = "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json"
else:
    map_style = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"


with st.sidebar.expander("🗺 Map Boundaries & Labels", expanded=False):

    if "p_darken_states" not in st.session_state:
        st.session_state["p_darken_states"] = True
    darken_states_overlay = st.checkbox(
        "Darker state boundaries (overlay)", key="p_darken_states"
    )

    if "p_border_width" not in st.session_state:
        st.session_state["p_border_width"] = 0.35
    state_border_width = st.slider(
        "State border thickness (px)",
        0.1, 2.0,
        step=0.25, key="p_border_width",
    )

    if "p_border_opacity" not in st.session_state:
        st.session_state["p_border_opacity"] = 0.1
    state_border_opacity = st.slider(
        "State border darkness",
        0.01, 1.0,
        step=0.05, key="p_border_opacity",
    )

    if "p_show_labels" not in st.session_state:
        st.session_state["p_show_labels"] = False
    show_state_labels = st.checkbox(
        "Show state labels (overlay)", key="p_show_labels"
    )

    if "p_label_size" not in st.session_state:
        st.session_state["p_label_size"] = 0.8
    state_label_size = st.slider(
        "State label size (px)",
        0.1, 10.0,
        step=0.1, key="p_label_size",
    )

    if "p_label_opacity" not in st.session_state:
        st.session_state["p_label_opacity"] = 0.4
    state_label_opacity = st.slider(
        "State label opacity",
        0.05, 1.0,
        step=0.05, key="p_label_opacity",
    )

# ---------------------------------------------------------
# Bubble Settings
# ---------------------------------------------------------
st.sidebar.header("Bubble Settings")

if "p_show_outlines" not in st.session_state:
    st.session_state["p_show_outlines"] = True
show_outlines    = st.sidebar.checkbox("Show bubble outlines", key="p_show_outlines")
if "p_fill_opacity" not in st.session_state:
    st.session_state["p_fill_opacity"] = 0.20
fill_opacity     = st.sidebar.slider("Bubble opacity", 0.05, 1.0, step=0.05, key="p_fill_opacity")
if "p_outline_width" not in st.session_state:
    st.session_state["p_outline_width"] = 2
outline_width    = st.sidebar.slider("Bubble outline width (px)", 0, 6, key="p_outline_width")

# optional declutter
if "p_min_ids" not in st.session_state:
    st.session_state["p_min_ids"] = 1
min_ids_to_show  = st.sidebar.slider("Hide groups with # of records < ", 1, 20, key="p_min_ids")
df_agg = df_agg[df_agg["n_ids"] >= min_ids_to_show].copy()

# ---------------------------------------------------------
# Zoom-in Settings  (ADD checkbox here)
# ---------------------------------------------------------
st.sidebar.header("Zoom-in Settings")

if "p_zoom_in_mode" not in st.session_state:
    st.session_state["p_zoom_in_mode"] = False
zoom_in_mode = st.sidebar.checkbox("Zoom in mode", key="p_zoom_in_mode")

# When Zoom in mode is ON, force radii to minimum possible values
if zoom_in_mode:
    min_px = 0.0   # minimum of slider range
    max_px = 1.0   # minimum of slider range
    # show the sliders but lock them
    st.sidebar.slider("Min radius (px)", 0.0, 1.0, min_px, step=0.1, disabled=True)
    st.sidebar.slider("Max radius (px)", 1.0, 10.0, max_px, step=0.5, disabled=True)
else:
    if "p_min_px" not in st.session_state:
        st.session_state["p_min_px"] = 0.3
    min_px = st.sidebar.slider("Min radius (px)", 0.0, 1.0, step=0.1, key="p_min_px")
    # Clamp any loaded max_px to be above min_px before the slider renders
    if "p_max_px" in st.session_state:
        st.session_state["p_max_px"] = max(float(st.session_state["p_max_px"]), float(min_px) + 0.5)
    else:
        st.session_state["p_max_px"] = max(2.0, float(min_px) + 0.5)
    # Max radius lower bound is always at least min_px + one step above it
    max_px = st.sidebar.slider("Max radius (px)", min_px + 0.5, 10.0, step=0.5, key="p_max_px")

# Hard clamp: guarantee max_px is strictly greater than min_px at all times
max_px = max(max_px, min_px + 0.1)

# Controls how much of the total radius range is used by the first 1..pivot IDs
if "p_alpha" not in st.session_state:
    st.session_state["p_alpha"] = 0.01
alpha  = st.sidebar.slider("Small-ID emphasis", 0.01, 0.10, step=0.01, key="p_alpha")
if "p_pivot" not in st.session_state:
    st.session_state["p_pivot"] = 5
pivot  = st.sidebar.slider("Pivot # (slow growth after n=)", 1, 200, key="p_pivot")
if "p_k_post" not in st.session_state:
    st.session_state["p_k_post"] = 50
k_post = st.sidebar.slider("Post-pivot saturation \n (damp big bubbles)", 50, 800, key="p_k_post")

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
# ⚙️  Save / Load Settings
# =========================================================
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Save / Load Settings")

# Keys that participate in save/load
_PARAM_KEYS = [
    "p_agg_mode", "p_base_style",
    "p_darken_states", "p_border_width", "p_border_opacity",
    "p_show_labels", "p_label_size", "p_label_opacity",
    "p_show_outlines", "p_fill_opacity", "p_outline_width",
    "p_min_ids", "p_zoom_in_mode", "p_min_px", "p_max_px",
    "p_alpha", "p_pivot", "p_k_post",
    "p_state_filter", "p_bins_selected", "p_top_n",
]

def _build_params_json() -> bytes:
    """Serialise all current widget values to a JSON byte-string."""
    params = {}
    for _k in _PARAM_KEYS:
        if _k in st.session_state:
            _v = st.session_state[_k]
            if hasattr(_v, "item"):                    # numpy scalar → Python native
                _v = _v.item()
            elif isinstance(_v, list):
                _v = [x.item() if hasattr(x, "item") else x for x in _v]
            params[_k] = _v
    return json.dumps({"version": 1, "params": params}, indent=2).encode("utf-8")

_settings_stem = Path(uploaded.name).stem
st.sidebar.download_button(
    "💾 Save current settings",
    data=_build_params_json(),
    file_name=f"{_settings_stem}_settings.json",
    mime="application/json",
    key="save_params_btn",
)
st.sidebar.file_uploader(
    "📂 Load settings (.json)", type=["json"], key="params_file_uploader"
)

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

    if agg_mode.startswith("ZIP"):
        tooltip_html = """
        <div style="font-family: sans-serif;">
        <b>{region}</b><br/>
        ZIP: <b>{zip5}</b><br/>
        Capacity: {cap_bin}<br/>
        Records: <b>{n_ids}</b>
        </div>
        """
    else:
        tooltip_html = """
        <div style="font-family: sans-serif;">
        <b>{region}</b><br/>
        Capacity: {cap_bin}<br/>
        Records: <b>{n_ids}</b>
        </div>
        """
    tooltip = {
        "html": tooltip_html,
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


    # For USA view, use the stored viewport (updated by user pan/zoom) as
    # initial_view_state so the map does NOT visually reset to full-USA
    # every time the user clicks "Process PNG" or changes a filter.
    # On first load (store is None) we fall back to the Python-computed values.
    if zoom_region == USA_OPTION:
        _vs = _get_vp("main_vp")
        _iv_lat  = _vs["lat"]  if _vs["lat"]  is not None else float(center_lat)
        _iv_lon  = _vs["lon"]  if _vs["lon"]  is not None else float(center_lon)
        _iv_zoom = _vs["zoom"] if _vs["zoom"] is not None else float(zoom)
    else:
        _iv_lat, _iv_lon, _iv_zoom = float(center_lat), float(center_lon), float(zoom)

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=_iv_lat, longitude=_iv_lon, zoom=_iv_zoom),
        tooltip=tooltip,
        map_style=map_style,
        map_provider="mapbox",
    )

    if zoom_region == USA_OPTION:
        _main_render_hash = _stable_component_hash({
            "zoom_region": zoom_region,
            "state_filter": tuple(sorted(state_filter)),
            "bins_selected": tuple(bins_selected),
            "agg_mode": agg_mode,
            "min_px": float(min_px),
            "max_px": float(max_px),
            "fill_opacity": float(fill_opacity),
            "outline_width": int(outline_width),
            "show_outlines": bool(show_outlines),
            "darken_states_overlay": bool(darken_states_overlay),
            "state_border_width": float(state_border_width),
            "state_border_opacity": float(state_border_opacity),
            "show_state_labels": bool(show_state_labels),
            "state_label_size": float(state_label_size),
            "state_label_opacity": float(state_label_opacity),
            "base_style": base_style,
            "min_ids_to_show": int(min_ids_to_show),
            "top_n_map": int(top_n_map),
            "map_rows": len(df_map),
        })
    
        render_synced_deck(
            deck,
            state_key="main_vp",
            component_key="main_deck",
            render_hash=_main_render_hash,
            height=520,
        )
    else:
        st.pydeck_chart(deck, use_container_width=True)

    # -----------------------------
    # One-click export (lazy render)
    # -----------------------------
    EXPORT_W, EXPORT_H, EXPORT_SCALE = 1400, 800, 2
    import math as _math

    # ── Compute export viewport ───────────────────────────────────────────────
    if zoom_region == USA_OPTION:
        # Use whatever viewport the user currently has on screen.
        # _get_vp_store() is updated by the JS onViewStateChange beacon.
        # Values are None until the user actually pans or zooms; fall back to
        # the Python-computed initial_view_state in that case.
        _vp = _get_vp("main_vp")
        exp_center_lat = _vp["lat"]  if _vp["lat"]  is not None else float(center_lat)
        exp_center_lon = _vp["lon"]  if _vp["lon"]  is not None else float(center_lon)
        _raw_zoom      = _vp["zoom"] if _vp["zoom"] is not None else float(zoom)
        # +0.4 keeps the export framing closer to the on-screen widget view.
        exp_zoom_val   = _raw_zoom + 0.4
    else:
        # City / region view: viewport is already auto-computed from selected region
        exp_center_lat = float(center_lat)
        exp_center_lon = float(center_lon)
        exp_zoom_val   = float(zoom)

    # Signature so we know when the map changed and the old export is stale
    map_signature = (
        zoom_region,
        tuple(sorted(state_filter)),
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
        base_style,
    )

    # If map changed, invalidate previously rendered PNG
    if st.session_state.get("export_signature") != map_signature:
        st.session_state["export_signature"] = map_signature
        st.session_state["export_png"] = None

    colA, colB = st.columns([1, 1])

    with colA:
        if st.button("🔄 Process high-res PNG file", key="render_png_btn"):
            with st.spinner("Rendering high-res PNG..."):
                # exp_center_lat / exp_center_lon / exp_zoom_val already set above
                # (either from the override controls or auto-computed)
                deck_export = pdk.Deck(
                    layers=layers,
                    initial_view_state=pdk.ViewState(
                        latitude=exp_center_lat,
                        longitude=exp_center_lon,
                        zoom=exp_zoom_val,
                    ),
                    tooltip=tooltip,
                    map_style=map_style,
                    map_provider="mapbox",
                )
                deck_export_html = deck_export.to_html(as_string=True)
                st.session_state["export_png"] = render_deck_png_from_html(
                    deck_export_html, EXPORT_W, EXPORT_H, EXPORT_SCALE
                )

    with colB:
        png_ready = st.session_state.get("export_png") is not None
        _png_fname = f"{Path(uploaded.name).stem}_MAPS_PNG.png"
        st.download_button(
            "⬇️ Download PNG",
            data=st.session_state.get("export_png") or b"",
            file_name=_png_fname,
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
    if "p_top_n" not in st.session_state:
        st.session_state["p_top_n"] = 10
    top_n = st.slider("How many regions", 5, 25, key="p_top_n")

    if "region" in df_agg.columns:
        ctab = (
            df_agg.groupby(["region", "cap_bin"], dropna=False, observed=True)["n_ids"]
                .sum()
                .reset_index(name="count")
        )

        totals = ctab.groupby("region")["count"].sum().sort_values(ascending=False)
        keep_regions = totals.head(top_n).index.tolist()

        # ── Secondary exclusion: remove specific regions from the bar chart only ──
        # (does NOT affect the interactive map or any other data)
        with st.expander("Remove regions from bar chart only", expanded=False):
            exclude_from_chart = st.multiselect(
                "Select regions to hide from bar chart (map is unaffected)",
                options=keep_regions,
                default=[],
                key="bar_chart_exclude",
            )
        if exclude_from_chart:
            keep_regions = [r for r in keep_regions if r not in exclude_from_chart]

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


# =========================================================

_tab_dist.__exit__(None, None, None)

