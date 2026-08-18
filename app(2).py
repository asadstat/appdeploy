"""
app.py - Automated Rice Detection with Streamlit + Earth Engine
Python port of the original Shiny + rgee application (pixel_count.R)
"""

import json
import os
import tempfile
import zipfile
from datetime import date

import ee
import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Automated Rice Monitoring",
    page_icon="\U0001F33E",
    layout="wide",
)

# ------------------------------------------------------------
# Optional default Earth Engine asset
# ------------------------------------------------------------
# Example:
# EE_ASSET_DEFAULT = "projects/ee-your-project/assets/table12"
EE_ASSET_DEFAULT = ""


# ============================================================
# Google Earth Engine initialization
# ============================================================
@st.cache_resource(show_spinner=False)
def initialize_gee():
    """Initialize Earth Engine once per server process."""
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()
    return True


# ============================================================
# File handling helpers
# ============================================================
def resolve_roi_filepath(uploaded_file):
    """
    Write a Streamlit UploadedFile to disk and return (file_path, ext),
    unzipping ZIPs and locating the .shp / GeoJSON / GPKG inside.
    """
    ext = os.path.splitext(uploaded_file.name)[1].lower().lstrip(".")

    tmp_dir = tempfile.mkdtemp(prefix="roi_")
    raw_path = os.path.join(tmp_dir, uploaded_file.name)

    with open(raw_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if ext == "zip":
        with zipfile.ZipFile(raw_path) as zf:
            zf.extractall(tmp_dir)

        shp_files = []
        geo_files = []
        for root, _dirs, files in os.walk(tmp_dir):
            for name in files:
                lower = name.lower()
                if lower.endswith(".shp"):
                    shp_files.append(os.path.join(root, name))
                elif lower.endswith((".geojson", ".json", ".gpkg")):
                    geo_files.append(os.path.join(root, name))

        if shp_files:
            return shp_files[0], "shp"
        if geo_files:
            found_ext = os.path.splitext(geo_files[0])[1].lower().lstrip(".")
            return geo_files[0], found_ext

        raise ValueError("No shapefile or GeoJSON was found in the ZIP.")

    return raw_path, ext


def guess_epsg_from_bbox(bounds):
    """
    Guess an EPSG code from raw (CRS-less) coordinates.

    - If coordinates already fall inside valid lon/lat ranges, assume
      geographic WGS84 (EPSG:4326).
    - Otherwise assume a projected, metre-based CRS (e.g. UTM) and fall
      back to the UTM zone covering the app's default study region
      (Bangladesh, ~90.4E/23.8N -> zone 46N / EPSG:32646), since projected
      eastings/northings alone can't reveal the zone or hemisphere
      without extra hints.
    """
    minx, miny, maxx, maxy = bounds

    looks_geographic = (
        -180 <= minx <= 180
        and -180 <= maxx <= 180
        and -90 <= miny <= 90
        and -90 <= maxy <= 90
    )

    if looks_geographic:
        return 4326, "coordinates fall within lon/lat bounds"

    return (
        32646,
        "coordinates look projected; defaulted to UTM zone 46N (Bangladesh)",
    )


def load_roi_gdf(uploaded_file, manual_epsg_input):
    """
    Read the uploaded file into a GeoDataFrame reprojected to EPSG:4326,
    applying the same missing-CRS fallback logic as the original R app.
    """
    file_path, ext = resolve_roi_filepath(uploaded_file)

    gdf = gpd.read_file(file_path)

    if gdf.crs is None:

        if ext in ("geojson", "json"):
            gdf = gdf.set_crs(4326)
            st.warning(
                "No CRS found in the GeoJSON; assumed WGS84 (EPSG:4326)."
            )

        else:
            manual_epsg = None
            try:
                manual_epsg = int(str(manual_epsg_input).strip())
            except (TypeError, ValueError):
                manual_epsg = None

            if manual_epsg:
                gdf = gdf.set_crs(epsg=manual_epsg)
                st.warning(
                    f"No CRS detected; applied manually entered EPSG:{manual_epsg}."
                )
            else:
                raise ValueError(
                    "The uploaded file has no CRS. If this is a shapefile ZIP, "
                    "make sure it includes the .prj file. Otherwise, enter the "
                    "correct EPSG code in the \"EPSG code if CRS can't be "
                    "detected\" field and re-upload."
                )

    gdf["geometry"] = gdf.geometry.buffer(0)  # ~ sf::st_make_valid
    gdf = gdf.to_crs(4326)
    return gdf


def sf_to_ee_geometry(gdf):
    """Convert a GeoDataFrame (EPSG:4326) to an ee.Geometry."""
    fc = ee.FeatureCollection(json.loads(gdf.to_json()))
    return fc.geometry()


# ============================================================
# Rice classification algorithm (mirrors run_rice_analysis in R)
# ============================================================
def speckle_correction(image):
    return (
        image.focal_mean(radius=1.5, kernelType="square", units="pixels")
        .rename("VH_corrected")
    )


def run_rice_analysis(
    geometry,
    start_date,
    end_date,
    transplant_end,
    peak_start,
    peak_end,
    trans_min_threshold,
    peak_max_threshold,
    peak_trans_diff_threshold,
    seasonal_min_threshold,
    seasonal_max_threshold,
    scale,
):
    sen1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterDate(start_date, end_date)
        .filterBounds(geometry)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
        .select("VH")
    )

    clipped_sen1 = sen1.map(lambda img: img.clip(geometry))
    corrected_sen1 = clipped_sen1.map(speckle_correction)

    # Growth-stage periods
    trans = clipped_sen1.filterDate(start_date, transplant_end)
    peak = clipped_sen1.filterDate(peak_start, peak_end)

    # Transplant period
    transboxer = trans.map(speckle_correction)
    transmax = transboxer.max().clip(geometry)
    transmin = transboxer.min().clip(geometry)
    transavg = transboxer.mean().clip(geometry)
    transstd = transboxer.reduce(ee.Reducer.stdDev()).clip(geometry)

    # Peak period
    peakboxer = peak.map(speckle_correction)
    peakmax = peakboxer.max().clip(geometry)
    peakmin = peakboxer.min().clip(geometry)
    peakavg = peakboxer.mean().clip(geometry)
    peakstd = peakboxer.reduce(ee.Reducer.stdDev()).clip(geometry)

    # Peak maximum - transplant minimum
    peaktransmaxmin_diff = peakmax.subtract(transmin).clip(geometry)

    # Whole season
    seasonal_mean = corrected_sen1.mean().clip(geometry)
    seasonal_std = corrected_sen1.reduce(ee.Reducer.stdDev()).clip(geometry)

    # Rice conditions
    con1 = transmin.lte(trans_min_threshold)
    con2 = peakmax.gte(peak_max_threshold)
    con3 = peaktransmaxmin_diff.gte(peak_trans_diff_threshold)
    con4 = seasonal_mean.gte(seasonal_min_threshold).And(
        seasonal_mean.lte(seasonal_max_threshold)
    )

    # Final rice mask
    rice1 = con1.And(con2).And(con3).And(con4).selfMask().clip(geometry)

    # Pixel count
    pixel_count = rice1.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=geometry,
        scale=scale,
        maxPixels=1e13,
    )

    pixel_info = pixel_count.getInfo()
    rice_pixels = float(list(pixel_info.values())[0]) if pixel_info else float("nan")

    # Area calculation
    # 1 pixel = scale x scale square metres
    rice_area_m2 = rice_pixels * scale * scale
    rice_area_ha = rice_area_m2 / 10000
    rice_area_km2 = rice_area_m2 / 1_000_000

    return {
        "collection": sen1,
        "corrected": corrected_sen1,
        "rice": rice1,
        "transmin": transmin,
        "transmax": transmax,
        "transavg": transavg,
        "transstd": transstd,
        "peakmin": peakmin,
        "peakmax": peakmax,
        "peakavg": peakavg,
        "peakstd": peakstd,
        "peaktransmaxmin_diff": peaktransmaxmin_diff,
        "seasonal_mean": seasonal_mean,
        "seasonal_std": seasonal_std,
        "rice_pixels": rice_pixels,
        "rice_area_m2": rice_area_m2,
        "rice_area_ha": rice_area_ha,
        "rice_area_km2": rice_area_km2,
    }


# ============================================================
# Map helper - adds an EE image as a tile layer on a folium map
# ============================================================
def add_ee_layer(folium_map, ee_image, vis_params, name):
    map_id_dict = ee.Image(ee_image).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict["tile_fetcher"].url_format,
        attr="Google Earth Engine",
        name=name,
        overlay=True,
        control=True,
    ).add_to(folium_map)


# ============================================================
# Session state defaults
# ============================================================
if "manual_epsg" not in st.session_state:
    st.session_state["manual_epsg"] = ""
if "result" not in st.session_state:
    st.session_state["result"] = None
if "roi_gdf" not in st.session_state:
    st.session_state["roi_gdf"] = None


def _on_roi_upload():
    """Auto-fill the EPSG field (silently) when an uploaded file has no CRS."""
    uploaded = st.session_state.get("roi_file")
    if uploaded is None:
        return

    try:
        file_path, _ext = resolve_roi_filepath(uploaded)
        gdf = gpd.read_file(file_path)
    except Exception:
        return

    if gdf.crs is None:
        epsg, _basis = guess_epsg_from_bbox(gdf.total_bounds)
        st.session_state["manual_epsg"] = str(epsg)
    else:
        st.session_state["manual_epsg"] = ""


# ============================================================
# Header / GEE status
# ============================================================
st.title("\U0001F33E Automated Rice Monitoring")

gee_ready = False
try:
    initialize_gee()
    gee_ready = True
except Exception as e:
    st.error(f"GEE initialization failed: {e}")

tab_analysis, tab_results, tab_about = st.tabs(
    ["Rice Analysis", "Results", "About"]
)

# ============================================================
# ANALYSIS TAB
# ============================================================
with tab_analysis:

    col1, col2, col3 = st.columns(3)

    # ---------------- Study area ----------------
    with col1:
        st.subheader("Study Area")

        roi_source = st.radio(
            "Study area source",
            options=["upload", "asset"],
            format_func=lambda v: {
                "upload": "Upload shapefile/GeoJSON",
                "asset": "Earth Engine asset",
            }[v],
        )

        ee_asset = EE_ASSET_DEFAULT

        if roi_source == "upload":
            st.file_uploader(
                "Upload study-area file",
                type=["zip", "shp", "shx", "dbf", "prj", "geojson", "json", "gpkg"],
                key="roi_file",
                on_change=_on_roi_upload,
            )
            st.caption(
                "For a shapefile, upload a ZIP containing .shp, .shx, .dbf and .prj."
            )
            st.text_input(
                "EPSG code if CRS can't be detected (optional)",
                key="manual_epsg",
                placeholder="e.g. 4326",
            )
            st.caption(
                "Only needed if you see a \"no CRS\" error "
                "(usually a shapefile ZIP missing its .prj file)."
            )
        else:
            ee_asset = st.text_input(
                "Earth Engine FeatureCollection asset",
                value=EE_ASSET_DEFAULT,
                placeholder="projects/ee-your-project/assets/table12",
            )

    # ---------------- Season & growth stage ----------------
    with col2:
        st.subheader("Season and Growth Stage")

        start_date = st.date_input("Season start", value=date(2023, 7, 1))
        end_date = st.date_input("Season end", value=date(2023, 12, 31))
        transplant_end = st.date_input(
            "Transplant period end", value=date(2023, 10, 15)
        )
        peak_start = st.date_input("Peak period start", value=date(2023, 10, 15))
        peak_end = st.date_input("Peak period end", value=date(2023, 11, 30))

    # ---------------- Rice classification thresholds ----------------
    with col3:
        st.subheader("Rice Classification")

        trans_min = st.number_input("Transplant minimum VH \u2264", value=-21.0, step=0.1)
        peak_max = st.number_input("Peak maximum VH \u2265", value=-20.0, step=0.1)
        peak_trans_diff = st.number_input(
            "Peak - transplant VH \u2265", value=3.0, step=0.1
        )
        season_min = st.number_input("Seasonal mean VH \u2265", value=-27.0, step=0.1)
        season_max = st.number_input("Seasonal mean VH \u2264", value=-15.0, step=0.1)
        scale = st.selectbox("Pixel scale", options=[10, 20, 30], index=0)

        run_clicked = st.button(
            "\u25B6 Run Rice Analysis", type="primary", use_container_width=True
        )

    st.divider()

    # ---------------- Run analysis ----------------
    if run_clicked:
        if not gee_ready:
            st.error("Earth Engine is not ready yet.")
        elif start_date >= end_date:
            st.error("Season start must be earlier than season end.")
        elif not (transplant_end > start_date):
            st.error("Invalid transplant period.")
        elif not (peak_start < peak_end):
            st.error("Invalid peak period.")
        else:
            try:
                with st.spinner("Running Sentinel-1 rice classification..."):

                    if roi_source == "asset":
                        if not ee_asset:
                            st.error("Please provide an Earth Engine asset ID.")
                            st.stop()
                        geometry = ee.FeatureCollection(ee_asset).geometry()
                        st.session_state["roi_gdf"] = None
                    else:
                        uploaded = st.session_state.get("roi_file")
                        if uploaded is None:
                            st.error("Please upload a study-area file.")
                            st.stop()
                        gdf = load_roi_gdf(uploaded, st.session_state["manual_epsg"])
                        st.session_state["roi_gdf"] = gdf
                        geometry = sf_to_ee_geometry(gdf)

                    ans = run_rice_analysis(
                        geometry=geometry,
                        start_date=str(start_date),
                        end_date=str(end_date),
                        transplant_end=str(transplant_end),
                        peak_start=str(peak_start),
                        peak_end=str(peak_end),
                        trans_min_threshold=trans_min,
                        peak_max_threshold=peak_max,
                        peak_trans_diff_threshold=peak_trans_diff,
                        seasonal_min_threshold=season_min,
                        seasonal_max_threshold=season_max,
                        scale=float(scale),
                    )

                    ans["params"] = {
                        "Season start": str(start_date),
                        "Season end": str(end_date),
                        "Transplant period end": str(transplant_end),
                        "Peak period start": str(peak_start),
                        "Peak period end": str(peak_end),
                        "Transplant minimum VH": trans_min,
                        "Peak maximum VH": peak_max,
                        "Peak-transplant difference": peak_trans_diff,
                        "Seasonal minimum VH": season_min,
                        "Seasonal maximum VH": season_max,
                        "Pixel scale": f"{scale} m",
                        "Rice pixels": ans["rice_pixels"],
                        "Rice area (m2)": ans["rice_area_m2"],
                        "Rice area (ha)": ans["rice_area_ha"],
                        "Rice area (km2)": ans["rice_area_km2"],
                    }

                    st.session_state["result"] = ans

                st.success("Analysis complete.")

            except Exception as e:
                st.error(f"Analysis failed: {e}")

    # ---------------- Value boxes + map ----------------
    result = st.session_state["result"]

    if result is not None:
        m1, m2, m3 = st.columns(3)
        m1.metric("Rice Area (hectares)", f"{result['rice_area_ha']:,.2f}")
        m2.metric("Rice Pixels", f"{result['rice_pixels']:,.0f}")
        n_images = result["collection"].size().getInfo()
        m3.metric("Sentinel-1 Images", f"{n_images:,}")

        st.subheader("Rice Classification Map")

        gdf = st.session_state.get("roi_gdf")
        if gdf is not None and not gdf.empty:
            bounds = gdf.total_bounds  # minx, miny, maxx, maxy
            center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        else:
            center = [23.8, 90.4]

        fmap = folium.Map(location=center, zoom_start=8, tiles="CartoDB positron")

        add_ee_layer(
            fmap,
            result["rice"],
            {"min": -50, "max": 10, "palette": ["red"]},
            "Rice",
        )

        if gdf is not None and not gdf.empty:
            folium.GeoJson(
                data=json.loads(gdf.to_json()),
                name="Study Area",
                style_function=lambda _f: {
                    "fillOpacity": 0,
                    "color": "#222222",
                    "weight": 2,
                },
            ).add_to(fmap)
            fmap.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

        folium.LayerControl(collapsed=False).add_to(fmap)
        st_folium(fmap, height=650, use_container_width=True)
    else:
        st.info("Run the analysis to see results here.")


# ============================================================
# RESULTS TAB
# ============================================================
with tab_results:
    st.subheader("Rice Analysis Results")

    result = st.session_state["result"]

    if result is None:
        st.info("Run the analysis on the Rice Analysis tab first.")
    else:
        df = pd.DataFrame(
            {
                "Parameter": list(result["params"].keys()),
                "Value": list(result["params"].values()),
            }
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name=f"rice_analysis_{date.today()}.csv",
            mime="text/csv",
        )

# ============================================================
# ABOUT TAB
# ============================================================
with tab_about:
    st.subheader("Automated Rice Monitoring using Sentinel-1")

    st.write(
        "This app implements the rice classification logic from the "
        "supplied Google Earth Engine JavaScript, ported to Python/Streamlit."
    )

    st.markdown("#### Sentinel-1 filtering")
    st.markdown(
        """
- COPERNICUS/S1_GRD
- VV and VH availability
- IW instrument mode
- Ascending orbit
- VH band
"""
    )

    st.markdown("#### Rice conditions")
    st.markdown(
        """
1. Transplant minimum VH \u2264 -21 dB
2. Peak maximum VH \u2265 -20 dB
3. Peak minus transplant maximum/minimum difference \u2265 3 dB
4. Seasonal mean VH between -27 and -15 dB
"""
    )
