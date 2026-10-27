# -*- coding: utf-8 -*-
"""
core/stac package
Satellite Imagery Downloader, Reflectance Normalizer, and Temporal Median Compositor.
Supports Sentinel-2A, Landsat-8, and Landsat-9 via Element 84 Earth Search STAC.
Sentinel-2B and Sentinel-2C are strictly excluded.
"""

from .stac_client import (
    STACClient,
    STACSearchError,
    BAND_ASSET_MAP,
    AUX_ASSET_MAP,
    ORDERED_BAND_LIST,
    LANDSAT_BAND_ASSET_MAP,
    LANDSAT_AUX_ASSET_MAP,
    LANDSAT_ORDERED_BAND_LIST,
    ALLOWED_PLATFORMS,
    DISALLOWED_PLATFORMS,
    EARTH_SEARCH_ENDPOINT,
    S2_COLLECTION_ID,
    LANDSAT_COLLECTION_ID,
    COLLECTION_ID,
    calculate_polygon_coverage,
    resolve_asset_for_band,
    get_landsat_sas_token,
    resolve_landsat_public_https_url
)
from .reflectance import (
    normalize_surface_reflectance,
    normalize_landsat_surface_reflectance,
    normalize_sensor_reflectance,
    resolve_reflectance_transform,
    parse_processing_baseline
)
from .water_mask import (
    create_water_mask,
    decode_landsat_qa_pixel,
    decode_landsat_qa_radsat,
    decode_landsat_qa_aerosol,
    compute_spectral_index,
    compute_otsu_threshold
)
from .grid import (
    build_target_grid,
    read_raster_into_grid,
    get_utm_epsg
)
from .processor import (
    process_single_scene,
    process_temporal_median_composite,
    save_multiband_geotiff,
    save_singleband_raster,
    ProcessorError
)

__all__ = [
    "STACClient",
    "STACSearchError",
    "BAND_ASSET_MAP",
    "AUX_ASSET_MAP",
    "ORDERED_BAND_LIST",
    "LANDSAT_BAND_ASSET_MAP",
    "LANDSAT_AUX_ASSET_MAP",
    "LANDSAT_ORDERED_BAND_LIST",
    "ALLOWED_PLATFORMS",
    "DISALLOWED_PLATFORMS",
    "EARTH_SEARCH_ENDPOINT",
    "S2_COLLECTION_ID",
    "LANDSAT_COLLECTION_ID",
    "COLLECTION_ID",
    "calculate_polygon_coverage",
    "resolve_asset_for_band",
    "normalize_surface_reflectance",
    "normalize_landsat_surface_reflectance",
    "normalize_sensor_reflectance",
    "resolve_reflectance_transform",
    "parse_processing_baseline",
    "create_water_mask",
    "decode_landsat_qa_pixel",
    "decode_landsat_qa_radsat",
    "decode_landsat_qa_aerosol",
    "compute_spectral_index",
    "compute_otsu_threshold",
    "build_target_grid",
    "read_raster_into_grid",
    "get_utm_epsg",
    "process_single_scene",
    "process_temporal_median_composite",
    "save_multiband_geotiff",
    "save_singleband_raster",
    "ProcessorError"
]
