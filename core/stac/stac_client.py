# -*- coding: utf-8 -*-
"""
core/stac/stac_client.py
Element 84 Earth Search STAC API Client for:
1. Sentinel-2A Level-2A (sentinel-2-c1-l2a, platform: sentinel-2a)
2. Landsat-8 OLI Collection 2 Level-2 Surface Reflectance (landsat-c2-l2, platform: landsat-8)
3. Landsat-9 OLI-2 Collection 2 Level-2 Surface Reflectance (landsat-c2-l2, platform: landsat-9)

Sentinel-2B and Sentinel-2C are strictly excluded and rejected.
Endpoint: https://earth-search.aws.element84.com/v1
"""

import json
import time
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Any, Tuple, Set


EARTH_SEARCH_ENDPOINT = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION_ID = "sentinel-2-c1-l2a"
LANDSAT_COLLECTION_ID = "landsat-c2-l2"
COLLECTION_ID = S2_COLLECTION_ID  # Preserved default for backward compatibility

# Supported satellite platforms (Sentinel-2B and Sentinel-2C strictly excluded)
ALLOWED_PLATFORMS: Set[str] = {"sentinel-2a", "landsat-8", "landsat-9"}
DISALLOWED_PLATFORMS: Set[str] = {"sentinel-2b", "sentinel-2c"}

# Standard 12 Reflectance Bands mapping for Sentinel-2A
BAND_ASSET_MAP = {
    "B01": {"asset_key": "coastal", "common_name": "coastal", "desc": "Coastal Aerosol", "res": 60},
    "B02": {"asset_key": "blue", "common_name": "blue", "desc": "Blue", "res": 10},
    "B03": {"asset_key": "green", "common_name": "green", "desc": "Green", "res": 10},
    "B04": {"asset_key": "red", "common_name": "red", "desc": "Red", "res": 10},
    "B05": {"asset_key": "rededge1", "common_name": "rededge1", "desc": "Red Edge 1", "res": 20},
    "B06": {"asset_key": "rededge2", "common_name": "rededge2", "desc": "Red Edge 2", "res": 20},
    "B07": {"asset_key": "rededge3", "common_name": "rededge3", "desc": "Red Edge 3", "res": 20},
    "B08": {"asset_key": "nir", "common_name": "nir", "desc": "NIR (Broad)", "res": 10},
    "B8A": {"asset_key": "nir08", "common_name": "nir08", "desc": "Narrow NIR", "res": 20},
    "B09": {"asset_key": "nir09", "common_name": "nir09", "desc": "Water Vapour", "res": 60},
    "B11": {"asset_key": "swir16", "common_name": "swir16", "desc": "SWIR 1", "res": 20},
    "B12": {"asset_key": "swir22", "common_name": "swir22", "desc": "SWIR 2", "res": 20},
}

AUX_ASSET_MAP = {
    "SCL": {"asset_key": "scl", "desc": "Scene Classification Layer", "res": 20},
    "AOT": {"asset_key": "aot", "desc": "Aerosol Optical Thickness", "res": 20},
    "WVP": {"asset_key": "wvp", "desc": "Water Vapour Content", "res": 20},
    "visual": {"asset_key": "visual", "desc": "True Color Image (TCI)", "res": 10},
    "preview": {"asset_key": "preview", "desc": "Preview Image", "res": 343},
    "thumbnail": {"asset_key": "thumbnail", "desc": "Thumbnail Image", "res": 343},
}

ORDERED_BAND_LIST = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"
]

# Standard 7 Reflectance Bands mapping for Landsat-8 and Landsat-9
LANDSAT_BAND_ASSET_MAP = {
    "SR_B1": {"asset_key": "coastal", "common_name": "coastal", "desc": "Coastal Aerosol", "res": 30},
    "SR_B2": {"asset_key": "blue", "common_name": "blue", "desc": "Blue", "res": 30},
    "SR_B3": {"asset_key": "green", "common_name": "green", "desc": "Green", "res": 30},
    "SR_B4": {"asset_key": "red", "common_name": "red", "desc": "Red", "res": 30},
    "SR_B5": {"asset_key": "nir08", "common_name": "nir08", "desc": "NIR", "res": 30},
    "SR_B6": {"asset_key": "swir16", "common_name": "swir16", "desc": "SWIR 1", "res": 30},
    "SR_B7": {"asset_key": "swir22", "common_name": "swir22", "desc": "SWIR 2", "res": 30},
}

LANDSAT_AUX_ASSET_MAP = {
    "QA_PIXEL": {"asset_key": "qa_pixel", "desc": "Pixel Quality Assessment", "res": 30},
    "QA_RADSAT": {"asset_key": "qa_radsat", "desc": "Radiometric Saturation QA", "res": 30},
    "SR_QA_AEROSOL": {"asset_key": "qa_aerosol", "desc": "Surface Reflectance Aerosol QA", "res": 30},
    "thumbnail": {"asset_key": "thumbnail", "desc": "Thumbnail Image", "res": 300},
    "reduced_resolution_browse": {"asset_key": "reduced_resolution_browse", "desc": "Browse Image", "res": 300},
}

LANDSAT_ORDERED_BAND_LIST = [
    "SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"
]

_LANDSAT_SAS_TOKEN_CACHE: Dict[str, Any] = {"token": None, "expires_at": 0.0}


def get_landsat_sas_token(timeout: int = 10) -> Optional[str]:
    """
    Fetch and cache a free, public SAS token for the Landsat Collection 2 Level-2 container
    from Microsoft Planetary Computer's open signing API.
    Does NOT require any user credentials or AWS account.
    """
    now = time.time()
    if _LANDSAT_SAS_TOKEN_CACHE["token"] and _LANDSAT_SAS_TOKEN_CACHE["expires_at"] > now + 300:
        return _LANDSAT_SAS_TOKEN_CACHE["token"]

    url = "https://planetarycomputer.microsoft.com/api/sas/v1/token/landsat-c2-l2"
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Bathymetrix-AI-QGIS-Plugin/2.0",
                "Accept": "application/json"
            }
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            token = data.get("token", "")
            if token:
                _LANDSAT_SAS_TOKEN_CACHE["token"] = token
                _LANDSAT_SAS_TOKEN_CACHE["expires_at"] = now + 3000
                return token
    except Exception:
        pass
    return None


def resolve_landsat_public_https_url(s3_or_https_url: str) -> str:
    """
    Translate AWS Requester-Pays 's3://usgs-landsat/...' URIs into open public HTTPS
    Cloud-Optimized GeoTIFF (COG) URLs using the open Microsoft Planetary Computer mirror and SAS token.
    Fulfills Requirement 11: 'Prefer STAC HTTPS assets that can be accessed without credentials.
    Do not require AWS credentials by default.'
    """
    if not s3_or_https_url or not isinstance(s3_or_https_url, str):
        return s3_or_https_url

    if "s3://usgs-landsat/" in s3_or_https_url:
        token = get_landsat_sas_token()
        if token:
            if "s3://usgs-landsat/collection02/" in s3_or_https_url:
                rel = s3_or_https_url.split("s3://usgs-landsat/collection02/")[-1]
            else:
                rel = s3_or_https_url.split("s3://usgs-landsat/")[-1]
            return f"https://landsateuwest.blob.core.windows.net/landsat-c2/{rel}?{token}"

    return s3_or_https_url


class STACSearchError(Exception):
    """Custom exception for STAC search errors."""
    pass


def _http_post_json(url: str, payload: dict, timeout: int = 30, max_retries: int = 4, backoff_factor: float = 1.5) -> dict:
    """Send an HTTP POST request with exponential backoff on retries."""
    data_bytes = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/geo+json, application/json",
        "User-Agent": "Bathymetrix-AI-QGIS-Plugin/2.0",
    }
    
    last_err = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status_code = resp.getcode()
                if status_code == 200:
                    raw_data = resp.read()
                    return json.loads(raw_data.decode("utf-8"))
                else:
                    raise STACSearchError(f"HTTP request failed with status code {status_code}")
        except urllib.error.HTTPError as e:
            last_err = e
            # Retry on rate limiting (429) or server errors (5xx)
            if e.code in [429, 500, 502, 503, 504]:
                wait_time = backoff_factor ** attempt
                time.sleep(wait_time)
                continue
            else:
                err_body = ""
                try:
                    err_body = e.read().decode("utf-8")
                except Exception:
                    pass
                raise STACSearchError(f"STAC API HTTP {e.code}: {e.reason}. {err_body}")
        except urllib.error.URLError as e:
            last_err = e
            wait_time = backoff_factor ** attempt
            time.sleep(wait_time)
            continue
        except Exception as e:
            last_err = e
            raise STACSearchError(f"Network error querying STAC API: {e}")

    raise STACSearchError(f"Failed to query STAC endpoint after {max_retries} attempts: {last_err}")


def calculate_polygon_coverage(scene_geometry: dict, aoi_geometry: dict) -> float:
    """
    Calculate the percentage of AOI covered by the scene footprint.
    Uses Shapely if available, otherwise approximates via bounding box overlap.
    """
    if not aoi_geometry or not scene_geometry:
        return 100.0

    try:
        from shapely.geometry import shape
        aoi_poly = shape(aoi_geometry)
        scene_poly = shape(scene_geometry)
        if aoi_poly.is_empty or aoi_poly.area <= 0:
            return 100.0
        inter = aoi_poly.intersection(scene_poly)
        coverage = (inter.area / aoi_poly.area) * 100.0
        return max(0.0, min(100.0, float(coverage)))
    except Exception:
        # Fallback to simple bbox intersection approximation
        try:
            def geom_bbox(g):
                coords = []
                def extract(c):
                    if isinstance(c[0], (int, float)):
                        coords.append(c)
                    else:
                        for sub in c:
                            extract(sub)
                extract(g['coordinates'])
                lons = [p[0] for p in coords]
                lats = [p[1] for p in coords]
                return min(lons), min(lats), max(lons), max(lats)
            
            s_minx, s_miny, s_maxx, s_maxy = geom_bbox(scene_geometry)
            a_minx, a_miny, a_maxx, a_maxy = geom_bbox(aoi_geometry)
            
            ix_min = max(s_minx, a_minx)
            iy_min = max(s_miny, a_miny)
            ix_max = min(s_maxx, a_maxx)
            iy_max = min(s_maxy, a_maxy)
            
            if ix_min < ix_max and iy_min < iy_max:
                inter_area = (ix_max - ix_min) * (iy_max - iy_min)
                aoi_area = (a_maxx - a_minx) * (a_maxy - a_miny)
                if aoi_area > 0:
                    return max(0.0, min(100.0, (inter_area / aoi_area) * 100.0))
            return 0.0
        except Exception:
            return 100.0


def resolve_asset_for_band(item_assets: dict, band_id: str, platform: Optional[str] = None) -> Optional[dict]:
    """
    Resolve asset metadata (href, raster:bands, scale, offset) for a given band ID.
    Supports both Sentinel-2A (e.g. 'B02', 'B8A', 'SCL') and Landsat-8/9 (e.g. 'SR_B1', 'SR_B2', 'QA_PIXEL').
    Checks:
    1. Direct match with BAND_ASSET_MAP / LANDSAT_BAND_ASSET_MAP / AUX maps
    2. Direct match with band_id as key in item_assets
    3. Match via 'eo:bands' (name, common_name)
    4. Suffix of href (e.g. '_SR_B1.TIF', 'B02.tif')
    """
    # 1. Check Sentinel-2 maps
    if band_id in BAND_ASSET_MAP:
        key = BAND_ASSET_MAP[band_id]["asset_key"]
        if key in item_assets:
            return item_assets[key]
    if band_id in AUX_ASSET_MAP:
        key = AUX_ASSET_MAP[band_id]["asset_key"]
        if key in item_assets:
            return item_assets[key]

    # 2. Check Landsat maps
    if band_id in LANDSAT_BAND_ASSET_MAP:
        key = LANDSAT_BAND_ASSET_MAP[band_id]["asset_key"]
        if key in item_assets:
            return item_assets[key]
    if band_id in LANDSAT_AUX_ASSET_MAP:
        key = LANDSAT_AUX_ASSET_MAP[band_id]["asset_key"]
        if key in item_assets:
            return item_assets[key]

    # 3. Direct key match
    if band_id in item_assets:
        return item_assets[band_id]

    # Also check case-insensitive key
    for k, asset in item_assets.items():
        if k.lower() == band_id.lower():
            return asset

    # 4. Search through eo:bands
    b_clean = band_id.upper().replace("SR_", "").replace("B0", "B")
    for k, asset in item_assets.items():
        if "eo:bands" in asset and isinstance(asset["eo:bands"], list):
            for eb in asset["eo:bands"]:
                name = str(eb.get("name", "")).upper()
                c_name = str(eb.get("common_name", "")).lower()
                if (
                    name == band_id.upper()
                    or name == b_clean
                    or name == f"OLI_{b_clean}"
                    or name == f"OLI_{band_id.upper()}"
                    or c_name == band_id.lower()
                ):
                    return asset

    # 5. Search through href filename suffix
    target_suffixes = [f"{band_id}.tif".lower(), f"_{band_id}.tif".lower()]
    if band_id.startswith("SR_"):
        b_num = band_id.replace("SR_", "")
        target_suffixes.extend([f"_{b_num}.tif".lower(), f"sr_{b_num}.tif".lower()])
    for k, asset in item_assets.items():
        href = asset.get("href", "").lower()
        for suf in target_suffixes:
            if href.endswith(suf):
                return asset

    return None


class STACClient:
    """Client for Element 84 Earth Search STAC API supporting Sentinel-2A, Landsat-8, and Landsat-9."""

    def __init__(self, endpoint: str = EARTH_SEARCH_ENDPOINT, collection: Optional[str] = None):
        self.endpoint = endpoint.rstrip("/")
        self.search_url = f"{self.endpoint}/search"
        self.default_collection = collection

    def _normalize_platform_filter(self, platform_input: Optional[str]) -> Tuple[List[str], List[str]]:
        """
        Normalize and validate requested satellite platform.
        Strictly forbids Sentinel-2B and Sentinel-2C.
        
        Returns:
            (target_collections, allowed_platform_filters)
        """
        if not platform_input or str(platform_input).strip().lower() in ["all", "all platforms", "any", "none"]:
            # Default to all 3 supported platforms
            return [S2_COLLECTION_ID, LANDSAT_COLLECTION_ID], ["sentinel-2a", "landsat-8", "landsat-9"]

        p = str(platform_input).strip().lower().replace("_", "-")

        # Strict exclusion check
        if "sentinel-2b" in p or "s2b" in p or "sentinel-2c" in p or "s2c" in p:
            raise STACSearchError(
                f"Satellite '{platform_input}' is not supported. Sentinel-2B and Sentinel-2C have been "
                "completely removed. Only Sentinel-2A, Landsat-8, and Landsat-9 are allowed."
            )

        if p in ["sentinel-2a", "s2a", "sentinel-2", "s2"]:
            return [S2_COLLECTION_ID], ["sentinel-2a"]
        elif p in ["landsat-8", "l8", "landsat8"]:
            return [LANDSAT_COLLECTION_ID], ["landsat-8"]
        elif p in ["landsat-9", "l9", "landsat9"]:
            return [LANDSAT_COLLECTION_ID], ["landsat-9"]
        elif p in ["landsat", "landsat-8/9", "landsat-8+9", "landsat_both", "landsat-8 & landsat-9"]:
            return [LANDSAT_COLLECTION_ID], ["landsat-8", "landsat-9"]
        else:
            raise STACSearchError(
                f"Unrecognized satellite platform filter: '{platform_input}'. "
                "Supported options are 'Sentinel-2A', 'Landsat-8', 'Landsat-9', or 'Landsat-8 & Landsat-9'."
            )

    def search_scenes(
        self,
        aoi_geometry: Optional[dict] = None,
        bbox: Optional[List[float]] = None,
        start_date: str = "2023-01-01",
        end_date: str = "2023-12-31",
        max_cloud_cover: float = 10.0,
        platform: Optional[str] = None,
        min_aoi_coverage: float = 0.0,
        max_scenes: int = 50,
        feedback: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for Sentinel-2A, Landsat-8, or Landsat-9 items matching criteria.
        Sentinel-2B and Sentinel-2C are strictly excluded.
        
        Parameters:
            aoi_geometry: GeoJSON geometry dict in EPSG:4326.
            bbox: [min_lon, min_lat, max_lon, max_lat] in EPSG:4326.
            start_date: YYYY-MM-DD or ISO-8601 string.
            end_date: YYYY-MM-DD or ISO-8601 string.
            max_cloud_cover: Maximum scene cloud cover percentage (0-100).
            platform: 'Sentinel-2A', 'Landsat-8', 'Landsat-9', or 'Landsat-8 & Landsat-9'.
            min_aoi_coverage: Minimum percentage of AOI covered by scene (0-100).
            max_scenes: Maximum number of scenes to return.
            feedback: Optional progress/feedback reporter (pushInfo, isCanceled).
            
        Returns:
            List of validated scene dictionaries with metadata and asset links.
        """
        # Validate and resolve collections and platforms
        collections, allowed_platforms = self._normalize_platform_filter(platform)

        # Format datetime range
        start_iso = f"{start_date}T00:00:00Z" if len(start_date) == 10 else start_date
        end_iso = f"{end_date}T23:59:59Z" if len(end_date) == 10 else end_date
        datetime_str = f"{start_iso}/{end_iso}"

        if not aoi_geometry and not bbox:
            raise STACSearchError("Either aoi_geometry or bbox must be specified for STAC search.")

        all_scenes = []

        for col_id in collections:
            if len(all_scenes) >= max_scenes:
                break

            if feedback and hasattr(feedback, "isCanceled") and feedback.isCanceled():
                break

            # Filter platforms specific to this collection
            if col_id == S2_COLLECTION_ID:
                col_platforms = [p for p in allowed_platforms if p == "sentinel-2a"]
            else:
                col_platforms = [p for p in allowed_platforms if p in ["landsat-8", "landsat-9"]]

            if not col_platforms:
                continue

            # Build query filter
            query_filter: Dict[str, Any] = {
                "eo:cloud_cover": {"lte": float(max_cloud_cover)}
            }
            if len(col_platforms) == 1:
                query_filter["platform"] = {"eq": col_platforms[0]}
            else:
                query_filter["platform"] = {"in": col_platforms}

            page_limit = min(50, max_scenes - len(all_scenes))
            payload: Dict[str, Any] = {
                "collections": [col_id],
                "datetime": datetime_str,
                "query": query_filter,
                "limit": page_limit,
                "sortby": [{"field": "properties.datetime", "direction": "desc"}]
            }

            if aoi_geometry:
                payload["intersects"] = aoi_geometry
            elif bbox:
                payload["bbox"] = bbox

            if feedback:
                feedback.pushInfo(f"Querying Earth Search STAC ({col_id} | platforms: {col_platforms})...")

            next_body = payload
            total_matched = None

            while len(all_scenes) < max_scenes:
                if feedback and hasattr(feedback, "isCanceled") and feedback.isCanceled():
                    if feedback:
                        feedback.pushInfo("Search canceled by user.")
                    break

                response = _http_post_json(self.search_url, next_body)
                
                if total_matched is None:
                    context = response.get("context", {})
                    total_matched = context.get("matched", 0)
                    if feedback:
                        feedback.pushInfo(f"Found {total_matched} matching scenes in {col_id}.")

                features = response.get("features", [])
                if not features:
                    break

                for feat in features:
                    if len(all_scenes) >= max_scenes:
                        break

                    feat_id = str(feat.get("id", ""))
                    props = feat.get("properties", {})
                    feat_geom = feat.get("geometry", {})
                    plat_raw = str(props.get("platform", "")).lower()

                    # STRICT VALIDATION: Ensure platform is strictly Sentinel-2A, Landsat-8, or Landsat-9
                    # Reject Sentinel-2B or Sentinel-2C under any circumstances
                    if plat_raw in DISALLOWED_PLATFORMS or "s2b" in feat_id.lower() or "s2c" in feat_id.lower():
                        continue
                    if plat_raw not in ALLOWED_PLATFORMS:
                        continue

                    # Calculate AOI coverage percentage
                    coverage_pct = 100.0
                    if aoi_geometry:
                        coverage_pct = calculate_polygon_coverage(feat_geom, aoi_geometry)
                    
                    if coverage_pct < min_aoi_coverage:
                        continue

                    assets = feat.get("assets", {})
                    cloud_pct = float(props.get("eo:cloud_cover", 0.0))
                    acq_date = props.get("datetime", "")

                    # SENSOR-SPECIFIC PARSING & VALIDATION
                    if plat_raw == "sentinel-2a":
                        satellite_name = "Sentinel-2A"
                        sensor_name = "MSI"
                        
                        # Tile
                        mgrs_tile = props.get("mgrs:grid_square") or ""
                        utm_zone = props.get("mgrs:utm_zone") or ""
                        lat_band = props.get("mgrs:latitude_band") or ""
                        tile_str = f"{utm_zone}{lat_band}{mgrs_tile}".strip()
                        if not tile_str and "s2:tile_id" in props:
                            tile_str = props["s2:tile_id"].split("_")[-2] if "_" in props["s2:tile_id"] else props["s2:tile_id"]

                        baseline = str(props.get("s2:processing_baseline", "N/A"))
                        boa_applied = props.get("earthsearch:boa_offset_applied", None)

                        # Resolve Sentinel-2 assets
                        resolved_assets = {}
                        for band_id in list(BAND_ASSET_MAP.keys()) + list(AUX_ASSET_MAP.keys()):
                            ast_meta = resolve_asset_for_band(assets, band_id, platform="sentinel-2a")
                            if ast_meta:
                                raster_bands = ast_meta.get("raster:bands", [])
                                rb0 = raster_bands[0] if raster_bands else {}
                                resolved_assets[band_id] = {
                                    "href": ast_meta.get("href"),
                                    "type": ast_meta.get("type", "image/tiff; application=geotiff; profile=cloud-optimized"),
                                    "scale": rb0.get("scale", 0.0001),
                                    "offset": rb0.get("offset", 0.0),
                                    "nodata": rb0.get("nodata", 0),
                                    "data_type": rb0.get("data_type", "uint16"),
                                    "spatial_resolution": rb0.get("spatial_resolution", BAND_ASSET_MAP.get(band_id, {}).get("res", 10))
                                }

                        # Validate that essential optical surface reflectance bands exist
                        if not all(b in resolved_assets for b in ["B02", "B03", "B04", "B08"]):
                            continue

                        # Preview URL
                        preview_url = ""
                        for p_key in ["preview", "thumbnail", "visual"]:
                            if p_key in resolved_assets and resolved_assets[p_key]["href"]:
                                preview_url = resolved_assets[p_key]["href"]
                                break

                    else:  # landsat-8 or landsat-9
                        satellite_name = "Landsat-8" if plat_raw == "landsat-8" else "Landsat-9"
                        sensor_name = "OLI" if plat_raw == "landsat-8" else "OLI-2"

                        # WRS Path / Row
                        wrs_path = props.get("landsat:wrs_path")
                        wrs_row = props.get("landsat:wrs_row")
                        if wrs_path is not None and wrs_row is not None:
                            tile_str = f"P{int(wrs_path):03d}R{int(wrs_row):03d}"
                        else:
                            tile_str = str(props.get("landsat:scene_id", ""))[:14]

                        baseline = "Collection 2 Level-2"
                        boa_applied = False

                        # Resolve Landsat assets
                        resolved_assets = {}
                        for band_id in list(LANDSAT_BAND_ASSET_MAP.keys()) + list(LANDSAT_AUX_ASSET_MAP.keys()):
                            ast_meta = resolve_asset_for_band(assets, band_id, platform=plat_raw)
                            if ast_meta:
                                raster_bands = ast_meta.get("raster:bands", [])
                                rb0 = raster_bands[0] if raster_bands else {}
                                raw_href = ast_meta.get("href", "")
                                # Fulfill Requirement 11: Prefer open public HTTPS assets over requester-pays S3
                                public_href = resolve_landsat_public_https_url(raw_href)
                                resolved_assets[band_id] = {
                                    "href": public_href,
                                    "raw_s3_href": raw_href,
                                    "type": ast_meta.get("type", "image/tiff; application=geotiff; profile=cloud-optimized"),
                                    "scale": rb0.get("scale", 0.0000275),
                                    "offset": rb0.get("offset", -0.2),
                                    "nodata": rb0.get("nodata", 0),
                                    "data_type": rb0.get("data_type", "uint16"),
                                    "spatial_resolution": rb0.get("spatial_resolution", 30.0)
                                }

                        # Validate that required Landsat optical Surface Reflectance bands exist
                        if not all(b in resolved_assets for b in ["SR_B2", "SR_B3", "SR_B4", "SR_B5"]):
                            continue

                        # Preview URL for Landsat: Earth Search public HTTPS thumbnail
                        preview_url = f"{self.endpoint}/collections/{col_id}/items/{feat_id}/thumbnail"

                    scene_info = {
                        "id": feat_id,
                        "satellite": satellite_name,
                        "platform": plat_raw,
                        "sensor": sensor_name,
                        "collection": col_id,
                        "datetime": acq_date,
                        "date": acq_date.split("T")[0] if "T" in acq_date else acq_date,
                        "cloud_cover": round(cloud_pct, 2),
                        "mgrs_tile": tile_str,
                        "tile": tile_str,
                        "processing_baseline": baseline,
                        "boa_offset_applied": boa_applied,
                        "aoi_coverage_pct": round(coverage_pct, 1),
                        "geometry": feat_geom,
                        "bbox": feat.get("bbox", []),
                        "preview_url": preview_url,
                        "assets": resolved_assets,
                        "raw_properties": props
                    }
                    all_scenes.append(scene_info)

                # Pagination
                links = response.get("links", [])
                next_link = None
                for link in links:
                    if link.get("rel") == "next":
                        next_link = link
                        break

                if next_link and "body" in next_link:
                    next_body = next_link["body"]
                else:
                    break

        if feedback:
            feedback.pushInfo(f"Retrieved {len(all_scenes)} validated scenes (Sentinel-2A / Landsat-8 / Landsat-9).")

        return all_scenes
