# -*- coding: utf-8 -*-
"""
core/stac/grid.py
Spatial Grid Management, Reprojection, and Polygon Masking for Satellite Processing.
Ensures common target grid (CRS, pixel size, origin, Affine transform) across all scenes and multi-tile mosaics.
Supports Sentinel-2A (10m/20m) and Landsat-8/9 (30m).
"""

import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask
from typing import Tuple, Dict, Any, Optional
import pyproj
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform


def get_utm_epsg(lon: float, lat: float) -> int:
    """Calculate the UTM EPSG code for a given WGS84 coordinate."""
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + zone
    else:
        return 32700 + zone


def build_target_grid(
    aoi_geojson: dict,
    target_crs: Optional[str] = None,
    pixel_size: float = 10.0
) -> Dict[str, Any]:
    """
    Construct a unified raster target grid for an AOI geometry.
    
    Parameters:
        aoi_geojson: GeoJSON geometry dictionary (in EPSG:4326).
        target_crs: Optional CRS string (e.g. 'EPSG:32635'). If None, calculates optimal UTM.
        pixel_size: Grid cell resolution in CRS units (e.g. 10.0 or 30.0 meters).
        
    Returns:
        Dict with keys:
          - crs: rasterio.crs.CRS
          - transform: Affine
          - width: int
          - height: int
          - aoi_mask: 2D boolean numpy array (True inside AOI polygon, False outside)
          - bounds: (minx, miny, maxx, maxy)
          - aoi_projected_geom: shapely geometry in target_crs
          - pixel_size: float
    """
    aoi_shapely = shape(aoi_geojson)
    centroid = aoi_shapely.centroid

    if target_crs is None:
        epsg = get_utm_epsg(centroid.x, centroid.y)
        dst_crs = rasterio.crs.CRS.from_epsg(epsg)
    else:
        dst_crs = rasterio.crs.CRS.from_user_input(target_crs)

    src_crs = rasterio.crs.CRS.from_epsg(4326)
    
    # Reproject AOI geometry to target CRS
    project_fn = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform
    aoi_proj = shapely_transform(project_fn, aoi_shapely)

    minx, miny, maxx, maxy = aoi_proj.bounds

    # Snap bounds to integer multiples of pixel_size for clean alignment
    minx_snap = np.floor(minx / pixel_size) * pixel_size
    maxy_snap = np.ceil(maxy / pixel_size) * pixel_size
    maxx_snap = np.ceil(maxx / pixel_size) * pixel_size
    miny_snap = np.floor(miny / pixel_size) * pixel_size

    width = max(1, int(np.round((maxx_snap - minx_snap) / pixel_size)))
    height = max(1, int(np.round((maxy_snap - miny_snap) / pixel_size)))

    grid_transform = from_origin(minx_snap, maxy_snap, pixel_size, pixel_size)

    # Generate exact Polygon Mask (True inside polygon, False outside)
    poly_mask = geometry_mask(
        [aoi_proj],
        out_shape=(height, width),
        transform=grid_transform,
        invert=True
    )

    return {
        "crs": dst_crs,
        "transform": grid_transform,
        "width": width,
        "height": height,
        "aoi_mask": poly_mask,
        "bounds": (minx_snap, miny_snap, maxx_snap, maxy_snap),
        "aoi_projected_geom": aoi_proj,
        "pixel_size": pixel_size
    }


def read_raster_into_grid(
    url_or_path: str,
    target_crs: rasterio.crs.CRS,
    target_transform: rasterio.Affine,
    target_shape: Tuple[int, int],
    is_classification: bool = False,
    src_nodata: float = 0.0,
    target_nodata: float = 0.0,
    window_bounds: Optional[Tuple[float, float, float, float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read remote COG / local raster and reproject directly into target grid.
    Supports both HTTPS and S3 (/vsis3/) protocols with access issue detection.
    
    Parameters:
        url_or_path: Remote HTTPS URL, S3 URI, or local file path.
        target_crs: Destination CRS.
        target_transform: Destination Affine transform.
        target_shape: (height, width) tuple.
        is_classification: If True, uses nearest neighbor resampling (for SCL/QA_PIXEL/QA_RADSAT).
                           If False, uses bilinear resampling (for continuous reflectance).
        src_nodata: Expected source nodata.
        target_nodata: Target nodata.
        window_bounds: Optional (minx, miny, maxx, maxy) in target_crs to optimize reading.
        
    Returns:
        (data_array, valid_mask)
    """
    resample_alg = Resampling.nearest if is_classification else Resampling.bilinear
    # Use uint16 for categorical/QA to prevent truncating 16-bit Landsat QA_PIXEL / QA_RADSAT
    dtype = np.uint16 if is_classification else np.float32

    destination = np.full(target_shape, target_nodata, dtype=dtype)

    # Resolve S3 vs HTTP path
    read_path = url_or_path
    if url_or_path.startswith("s3://usgs-landsat/"):
        try:
            from .stac_client import resolve_landsat_public_https_url
            resolved_url = resolve_landsat_public_https_url(url_or_path)
            if resolved_url.startswith("http"):
                read_path = f"/vsicurl/{resolved_url}"
            else:
                read_path = url_or_path.replace("s3://", "/vsis3/")
        except Exception:
            read_path = url_or_path.replace("s3://", "/vsis3/")
    elif url_or_path.startswith("s3://"):
        read_path = url_or_path.replace("s3://", "/vsis3/")
    elif (url_or_path.startswith("http://") or url_or_path.startswith("https://")) and not url_or_path.startswith("/vsicurl/"):
        read_path = f"/vsicurl/{url_or_path}"

    env_kwargs = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF",
        "VSI_CACHE": True,
        "GDAL_HTTP_TIMEOUT": "30",
        "GDAL_HTTP_MAX_RETRY": "4",
        "GDAL_HTTP_RETRY_DELAY": "2"
    }

    # Pass requester-pays and AWS settings if environment configured
    if "AWS_REQUEST_PAYER" in os.environ:
        env_kwargs["AWS_REQUEST_PAYER"] = os.environ["AWS_REQUEST_PAYER"]
    else:
        env_kwargs["AWS_REQUEST_PAYER"] = "requester"

    if "AWS_REGION" in os.environ:
        env_kwargs["AWS_REGION"] = os.environ["AWS_REGION"]
    else:
        env_kwargs["AWS_REGION"] = "us-west-2"

    try:
        with rasterio.Env(**env_kwargs):
            with rasterio.open(read_path) as src:
                actual_src_nodata = src.nodata if src.nodata is not None else src_nodata
                
                reproject(
                    source=rasterio.band(src, 1),
                    destination=destination,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=actual_src_nodata,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    dst_nodata=target_nodata,
                    resampling=resample_alg
                )
    except Exception as e:
        err_str = str(e)
        if any(w in err_str for w in ["403", "Requester Pays", "AccessDenied", "Authentication", "credentials", "Anonymous users"]):
            raise RuntimeError(
                f"Asset access issue on '{url_or_path}': The asset is in a Requester-Pays bucket ('usgs-landsat') "
                "or requires AWS authentication. Please ensure AWS credentials or requester-pays settings are configured. "
                f"Original error: {err_str}"
            )
        raise RuntimeError(f"Failed to read and reproject raster from '{url_or_path}': {err_str}")

    if is_classification:
        valid_mask = (destination != target_nodata)
    else:
        valid_mask = (destination != target_nodata) & (destination > 0) & np.isfinite(destination)

    return destination, valid_mask
