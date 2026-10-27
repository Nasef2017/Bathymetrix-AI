# -*- coding: utf-8 -*-
"""
core/stac/processor.py
Main Processing Engine for Satellite Imagery Downloader (Earth Search STAC).
Supports:
1. Sentinel-2A Level-2A (MSI Surface Reflectance)
2. Landsat-8 OLI Collection 2 Level-2 Surface Reflectance
3. Landsat-9 OLI-2 Collection 2 Level-2 Surface Reflectance

Sentinel-2B and Sentinel-2C are strictly excluded and rejected.
Handles Single Scene Processing and Temporal Median Composite Processing with
windowed COG reads, sensor-specific reflectance normalization, QA bitmask decoding,
exact polygon masking, and Float32 GeoTIFF export.
"""

import os
import json
import time
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import rasterio

from .stac_client import (
    BAND_ASSET_MAP,
    AUX_ASSET_MAP,
    ORDERED_BAND_LIST,
    LANDSAT_BAND_ASSET_MAP,
    LANDSAT_AUX_ASSET_MAP,
    LANDSAT_ORDERED_BAND_LIST,
    ALLOWED_PLATFORMS,
    DISALLOWED_PLATFORMS,
    STACClient
)
from .reflectance import (
    normalize_surface_reflectance,
    normalize_landsat_surface_reflectance,
    normalize_sensor_reflectance
)
from .water_mask import (
    create_water_mask,
    decode_landsat_qa_pixel,
    decode_landsat_qa_radsat,
    decode_landsat_qa_aerosol,
    compute_spectral_index
)
from .grid import build_target_grid, read_raster_into_grid


class ProcessorError(Exception):
    """Custom exception for processing errors."""
    pass


def save_multiband_geotiff(
    output_path: str,
    band_arrays: List[np.ndarray],
    band_names: List[str],
    grid: dict,
    nodata_val: float = -9999.0,
    compression: str = "DEFLATE",
    metadata_extra: Optional[dict] = None
) -> str:
    """Save a multiband Float32 GeoTIFF with sensor-aware band descriptions and metadata."""
    num_bands = len(band_arrays)
    height, width = grid["height"], grid["width"]

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": nodata_val,
        "width": width,
        "height": height,
        "count": num_bands,
        "crs": grid["crs"],
        "transform": grid["transform"],
        "compress": compression.lower(),
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "interleave": "band"
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        for idx, (arr, b_id) in enumerate(zip(band_arrays, band_names), start=1):
            dst.write(arr.astype(np.float32), idx)
            # Check Sentinel-2 and Landsat asset mappings
            b_info = BAND_ASSET_MAP.get(b_id) or LANDSAT_BAND_ASSET_MAP.get(b_id, {})
            b_desc = f"{b_id} - {b_info.get('desc', b_id)} ({b_info.get('res', '')}m)"
            dst.set_band_description(idx, b_desc)

        tags = {
            "DATA_TYPE": "Surface Reflectance (Float32)",
            "PROCESSED_BY": "Bathymetrix-AI Satellite STAC Downloader",
            "UNITS": "Surface Reflectance [0.0 - 1.0]",
            "NODATA_VALUE": str(nodata_val),
            "SCALING_APPLIED": "True (Surface Reflectance, already scaled)",
            "DATE_PROCESSED": datetime.now().isoformat()
        }
        if metadata_extra:
            for k, v in metadata_extra.items():
                if isinstance(v, (str, int, float, bool)):
                    tags[str(k).upper()] = str(v)
        dst.update_tags(**tags)

    return output_path


def save_singleband_raster(
    output_path: str,
    data_array: np.ndarray,
    grid: dict,
    dtype: str = "uint8",
    nodata_val: Optional[float] = 255,
    description: str = "",
    compression: str = "DEFLATE"
) -> str:
    """Save a single-band categorical or quality raster (e.g. Water Mask, Valid Count)."""
    height, width = grid["height"], grid["width"]

    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "nodata": nodata_val,
        "width": width,
        "height": height,
        "count": 1,
        "crs": grid["crs"],
        "transform": grid["transform"],
        "compress": compression.lower(),
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data_array.astype(dtype), 1)
        if description:
            dst.set_band_description(1, description)

    return output_path


def process_single_scene(
    scene_meta: dict,
    aoi_geojson: dict,
    output_dir: str,
    bands_to_process: Optional[List[str]] = None,
    target_crs: Optional[str] = None,
    pixel_size: Optional[float] = None,
    water_only: bool = False,
    water_method: str = "hybrid",
    water_mode: str = "permissive",
    water_threshold: Optional[float] = None,
    save_water_mask_raster: bool = True,
    file_prefix: Optional[str] = None,
    feedback: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Process a single Sentinel-2A, Landsat-8, or Landsat-9 scene.
    Sentinel-2B and Sentinel-2C are strictly rejected.
    
    Workflow:
      1. Retrieve STAC asset metadata & identify sensor
      2. Build unified target spatial grid (auto UTM)
      3. Read COG assets via HTTP range requests
      4. Apply sensor-specific reflectance normalization
      5. Decode QA bands (SCL for S2A, QA_PIXEL/QA_RADSAT for Landsat)
      6. Apply water masking (Full ROI vs Water-Only)
      7. Clip strictly to AOI polygon
      8. Export Float32 GeoTIFF, water mask, and provenance metadata
    """
    scene_id = scene_meta["id"]
    plat = str(scene_meta.get("platform", "sentinel-2a")).lower().strip()

    # STRICT PLATFORM VALIDATION
    if plat in DISALLOWED_PLATFORMS or "s2b" in scene_id.lower() or "s2c" in scene_id.lower():
        raise ProcessorError(
            f"Scene {scene_id} belongs to removed satellite platform '{plat}'. "
            "Sentinel-2B and Sentinel-2C are strictly excluded from download and processing."
        )

    if plat == "sentinel-2a":
        is_landsat = False
        satellite_name = "Sentinel-2A"
        sensor_name = "MSI"
        collection_name = scene_meta.get("collection", "sentinel-2-c1-l2a")
        default_bands = ORDERED_BAND_LIST
        default_res = 10.0
        default_prefix = "S2A"
    elif plat in ["landsat-8", "landsat-9"]:
        is_landsat = True
        satellite_name = "Landsat-8" if plat == "landsat-8" else "Landsat-9"
        sensor_name = "OLI" if plat == "landsat-8" else "OLI-2"
        collection_name = scene_meta.get("collection", "landsat-c2-l2")
        default_bands = LANDSAT_ORDERED_BAND_LIST
        default_res = 30.0
        default_prefix = "L8" if plat == "landsat-8" else "L9"
    else:
        raise ProcessorError(
            f"Unsupported satellite platform: '{plat}'. Only Sentinel-2A, Landsat-8, and Landsat-9 are supported."
        )

    eff_pixel_size = float(pixel_size) if pixel_size is not None else default_res
    prefix = file_prefix if file_prefix is not None else default_prefix
    target_bands = bands_to_process if bands_to_process is not None else default_bands
    scene_date = scene_meta["date"].replace("-", "")

    if feedback:
        feedback.pushInfo(f"=== Starting Single Scene Processing: {scene_id} ({satellite_name}) ===")
        feedback.pushInfo(f"Acquisition Date: {scene_meta['date']} | Cloud Cover: {scene_meta['cloud_cover']}% | Res: {eff_pixel_size}m")

    # 1. Build Target Grid
    if feedback:
        feedback.pushInfo("Building unified target spatial grid...")
    grid = build_target_grid(aoi_geojson, target_crs=target_crs, pixel_size=eff_pixel_size)
    height, width = grid["height"], grid["width"]
    aoi_poly_mask = grid["aoi_mask"]

    if feedback:
        feedback.pushInfo(f"Target Grid: {width}x{height} pixels | CRS: {grid['crs']} | Pixel Size: {eff_pixel_size}m")

    # 2. Read QA layers
    scl_arr = None
    qa_pixel_arr = None
    qa_radsat_arr = None

    if not is_landsat:
        # Sentinel-2: Read SCL if available
        if "SCL" in scene_meta["assets"]:
            if feedback:
                feedback.pushInfo("Reading Scene Classification Layer (SCL)...")
            scl_url = scene_meta["assets"]["SCL"]["href"]
            try:
                scl_arr, _ = read_raster_into_grid(
                    scl_url,
                    target_crs=grid["crs"],
                    target_transform=grid["transform"],
                    target_shape=(height, width),
                    is_classification=True,
                    src_nodata=0,
                    target_nodata=0
                )
            except Exception as err:
                if feedback:
                    feedback.pushWarning(f"Could not load SCL: {err}")
    else:
        # Landsat: Read QA_PIXEL and QA_RADSAT if available
        if "QA_PIXEL" in scene_meta["assets"]:
            if feedback:
                feedback.pushInfo("Reading Landsat QA_PIXEL bitmask...")
            qa_url = scene_meta["assets"]["QA_PIXEL"]["href"]
            try:
                qa_pixel_arr, _ = read_raster_into_grid(
                    qa_url,
                    target_crs=grid["crs"],
                    target_transform=grid["transform"],
                    target_shape=(height, width),
                    is_classification=True,
                    src_nodata=1,
                    target_nodata=1
                )
            except Exception as err:
                if feedback:
                    feedback.pushWarning(f"Could not load QA_PIXEL: {err}")

        if "QA_RADSAT" in scene_meta["assets"]:
            if feedback:
                feedback.pushInfo("Reading Landsat QA_RADSAT bitmask...")
            rad_url = scene_meta["assets"]["QA_RADSAT"]["href"]
            try:
                qa_radsat_arr, _ = read_raster_into_grid(
                    rad_url,
                    target_crs=grid["crs"],
                    target_transform=grid["transform"],
                    target_shape=(height, width),
                    is_classification=True,
                    src_nodata=0,
                    target_nodata=0
                )
            except Exception as err:
                if feedback:
                    feedback.pushWarning(f"Could not load QA_RADSAT: {err}")

    # 3. Read and normalize each spectral band
    band_refl_arrays = []
    normalized_band_names = []
    provenance_log = {}

    total_bands = len(target_bands)
    for i, b_id in enumerate(target_bands):
        if feedback and hasattr(feedback, "isCanceled") and feedback.isCanceled():
            raise ProcessorError("Processing canceled by user.")

        if b_id not in scene_meta["assets"]:
            # Check if mandatory band missing
            mandatory_bands = ["B02", "B03", "B04", "B08"] if not is_landsat else ["SR_B2", "SR_B3", "SR_B4", "SR_B5"]
            if b_id in mandatory_bands:
                raise ProcessorError(f"Mandatory Surface Reflectance band '{b_id}' not found in scene {scene_id} assets.")
            if feedback:
                feedback.pushWarning(f"Band {b_id} not found in scene assets; skipping.")
            continue

        b_meta = scene_meta["assets"][b_id]
        b_url = b_meta["href"]

        if feedback:
            feedback.pushInfo(f"[{i+1}/{total_bands}] Reading & normalizing {satellite_name} band {b_id}...")
            if hasattr(feedback, "setProgress"):
                feedback.setProgress(int((i + 1) / (total_bands + 3) * 100))

        # Read raw DN
        raw_dn, raw_valid = read_raster_into_grid(
            b_url,
            target_crs=grid["crs"],
            target_transform=grid["transform"],
            target_shape=(height, width),
            is_classification=False,
            src_nodata=b_meta.get("nodata", 0),
            target_nodata=0
        )

        # Apply sensor-specific surface reflectance normalization
        if is_landsat:
            refl_arr, val_m, b_prov = normalize_landsat_surface_reflectance(
                raw_dn,
                scale=b_meta.get("scale", 0.0000275),
                offset=b_meta.get("offset", -0.2),
                raw_nodata=b_meta.get("nodata", 0),
                output_nodata=-9999.0
            )
        else:
            refl_arr, val_m, b_prov = normalize_surface_reflectance(
                raw_dn,
                scale=b_meta.get("scale", 0.0001),
                offset=b_meta.get("offset", None),
                raw_nodata=b_meta.get("nodata", 0),
                processing_baseline=scene_meta.get("processing_baseline"),
                boa_offset_applied=scene_meta.get("boa_offset_applied"),
                output_nodata=-9999.0
            )

        # Apply True AOI Polygon Mask: pixels outside AOI are strictly set to NoData
        refl_arr[~aoi_poly_mask] = -9999.0

        band_refl_arrays.append(refl_arr)
        normalized_band_names.append(b_id)
        provenance_log[b_id] = b_prov

    if not band_refl_arrays:
        raise ProcessorError(f"No spectral bands could be loaded for scene {scene_id}")

    # 4. Optional Water Masking
    water_bool_mask = None
    water_stats = {}
    water_uint8_mask = None

    if water_only or save_water_mask_raster:
        if feedback:
            feedback.pushInfo(f"Computing Water Mask using method: {water_method} (mode: {water_mode})...")

        def get_band(b_name):
            if b_name in normalized_band_names:
                idx = normalized_band_names.index(b_name)
                return band_refl_arrays[idx]
            return None

        # Resolve spectral bands for water indices:
        # Sentinel-2A: Green=B03, NIR=B08, SWIR1=B11
        # Landsat-8/9: Green=SR_B3, NIR=SR_B5, SWIR1=SR_B6
        if is_landsat:
            green = get_band("SR_B3")
            nir = get_band("SR_B5")
            swir1 = get_band("SR_B6")
        else:
            green = get_band("B03")
            nir = get_band("B08")
            swir1 = get_band("B11")

        water_bool_mask, water_uint8_mask, water_stats = create_water_mask(
            scl=scl_arr,
            qa_pixel=qa_pixel_arr,
            qa_radsat=qa_radsat_arr,
            green_refl=green,
            nir_refl=nir,
            swir1_refl=swir1,
            valid_mask=aoi_poly_mask,
            method=water_method,
            mode=water_mode,
            threshold=water_threshold,
            use_otsu=(water_threshold is None)
        )

        water_uint8_mask[~aoi_poly_mask] = 255

        if feedback:
            feedback.pushInfo(f"Water Mask Stats: {water_stats['water_percentage']}% Water | {water_stats['invalid_percentage']}% Invalid/Clouds")

        if water_only:
            if feedback:
                feedback.pushInfo("Applying Water-Only Mask: non-water land pixels set to NoData (-9999.0)...")
            for arr in band_refl_arrays:
                arr[~water_bool_mask | ~aoi_poly_mask] = -9999.0

    # 5. Export GeoTIFF Outputs
    os.makedirs(output_dir, exist_ok=True)
    mode_str = "WaterOnly" if water_only else "BOA_Float32"
    refl_filename = f"{prefix}_{scene_date}_{scene_id}_{mode_str}.tif"
    refl_path = os.path.join(output_dir, refl_filename)

    if feedback:
        feedback.pushInfo(f"Exporting Multiband Float32 GeoTIFF to: {refl_path}")

    # Metadata tags for raster
    scale_val = 0.0000275 if is_landsat else 0.0001
    offset_val = -0.2 if is_landsat else (-0.1 if scene_meta.get("processing_baseline") not in [None, "N/A"] else 0.0)

    save_multiband_geotiff(
        refl_path,
        band_arrays=band_refl_arrays,
        band_names=normalized_band_names,
        grid=grid,
        nodata_val=-9999.0,
        metadata_extra={
            "SCENE_ID": scene_id,
            "SATELLITE": satellite_name,
            "SENSOR": sensor_name,
            "PLATFORM": plat,
            "STAC_COLLECTION": collection_name,
            "ACQUISITION_DATE": scene_meta["date"],
            "TILE": scene_meta.get("mgrs_tile", ""),
            "PROCESSING_LEVEL": "Collection 2 Level-2" if is_landsat else "Level-2A BOA",
            "WATER_ONLY_APPLIED": str(water_only),
            "WATER_METHOD": water_method,
            "SCALE_FACTOR": str(scale_val),
            "OFFSET": str(offset_val),
            "PIXEL_SIZE": str(eff_pixel_size)
        }
    )

    outputs = {
        "surface_reflectance": refl_path,
        "scene_id": scene_id,
        "satellite": satellite_name,
        "platform": plat,
        "grid": grid,
        "bands": normalized_band_names,
        "water_stats": water_stats
    }

    # Save Water Mask raster if requested
    if save_water_mask_raster and water_uint8_mask is not None:
        mask_filename = f"{prefix}_{scene_date}_{scene_id}_WaterMask.tif"
        mask_path = os.path.join(output_dir, mask_filename)
        save_singleband_raster(
            mask_path,
            water_uint8_mask,
            grid=grid,
            dtype="uint8",
            nodata_val=255,
            description="Water Mask: 1=Water, 0=Land, 255=NoData"
        )
        outputs["water_mask"] = mask_path

    # Save Provenance JSON Metadata (Section 12 compliance)
    json_path = os.path.join(output_dir, f"{prefix}_{scene_date}_{scene_id}_Processing_Metadata.json")
    meta_doc = {
        "satellite": satellite_name,
        "platform": plat,
        "sensor": sensor_name,
        "stac_collection": collection_name,
        "processing_mode": "Single Scene",
        "scene_id": scene_id,
        "acquisition_date": scene_meta["datetime"],
        "cloud_cover_percent": scene_meta["cloud_cover"],
        "processing_level": "Collection 2 Level-2" if is_landsat else f"Processing Baseline {scene_meta.get('processing_baseline', 'N/A')}",
        "tile": scene_meta.get("mgrs_tile", ""),
        "crs": str(grid["crs"]),
        "spatial_resolution_meters": eff_pixel_size,
        "dimensions": [width, height],
        "input_data_type": "uint16",
        "output_data_type": "float32",
        "scale_factor": scale_val,
        "offset": offset_val,
        "scaling_applied_by": "Bathymetrix-AI STAC Reader (Surface Reflectance, already scaled)",
        "bands": normalized_band_names,
        "source_assets": {b: scene_meta["assets"][b]["href"] for b in normalized_band_names if b in scene_meta["assets"]},
        "water_only": water_only,
        "water_mask_statistics": water_stats,
        "band_normalization_provenance": provenance_log,
        "generation_timestamp": datetime.now().isoformat(),
        "export_path": refl_path
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta_doc, f, indent=2)
    outputs["metadata_json"] = json_path

    if feedback:
        feedback.pushInfo(f"=== Single Scene Processing Completed Successfully: {refl_path} ===")

    return outputs


def process_temporal_median_composite(
    scenes_meta: List[dict],
    aoi_geojson: dict,
    output_dir: str,
    bands_to_process: Optional[List[str]] = None,
    target_crs: Optional[str] = None,
    pixel_size: Optional[float] = None,
    water_only: bool = False,
    water_temporal_policy: str = "consensus",
    water_occurrence_threshold: float = 50.0,
    min_valid_observations: int = 1,
    save_valid_count_raster: bool = True,
    save_temp_std_raster: bool = True,
    save_water_occurrence_raster: bool = True,
    chunk_size: int = 512,
    file_prefix: Optional[str] = None,
    feedback: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Process Temporal Median Composite across multiple scenes.
    
    Supports:
    - Sentinel-2A composite (MSI Level-2A)
    - Landsat-8 composite (OLI)
    - Landsat-9 composite (OLI-2)
    - Landsat-8 + Landsat-9 combined composite
    
    Strict constraints:
    - Sentinel-2B and Sentinel-2C are strictly excluded and rejected.
    - Sentinel-2A and Landsat-8/9 cannot be combined in the same composite automatically.
    """
    if not scenes_meta:
        raise ProcessorError("No scenes provided for temporal median composite.")

    # 1. Validate scenes and detect sensor compatibility
    platforms_present = set()
    for s in scenes_meta:
        plat = str(s.get("platform", "")).lower().strip()
        s_id = str(s.get("id", ""))
        if plat in DISALLOWED_PLATFORMS or "s2b" in s_id.lower() or "s2c" in s_id.lower():
            raise ProcessorError(
                f"Scene {s_id} belongs to removed satellite platform '{plat}'. "
                "Sentinel-2B and Sentinel-2C cannot enter Median Composite processing."
            )
        if plat not in ALLOWED_PLATFORMS:
            raise ProcessorError(f"Scene {s_id} platform '{plat}' is not supported. Supported: Sentinel-2A, Landsat-8, Landsat-9.")
        platforms_present.add(plat)

    # Check for invalid cross-sensor mixing (Sentinel-2A + Landsat)
    has_s2a = "sentinel-2a" in platforms_present
    has_landsat = ("landsat-8" in platforms_present) or ("landsat-9" in platforms_present)

    if has_s2a and has_landsat:
        raise ProcessorError(
            "Sentinel-2A and Landsat-8/9 cannot be combined in the same median composite automatically. "
            "Cross-sensor harmonization is outside the scope of this update. "
            "Please composite Sentinel-2A and Landsat scenes separately."
        )

    # Sensor-specific setup
    if has_landsat:
        is_landsat = True
        default_bands = LANDSAT_ORDERED_BAND_LIST
        default_res = 30.0
        has_l8 = "landsat-8" in platforms_present
        has_l9 = "landsat-9" in platforms_present
        if has_l8 and has_l9:
            satellite_label = "Landsat-8 & Landsat-9"
            sensor_label = "OLI/OLI-2"
            default_prefix = "Landsat_Composite"
        elif has_l8:
            satellite_label = "Landsat-8"
            sensor_label = "OLI"
            default_prefix = "L8_Composite"
        else:
            satellite_label = "Landsat-9"
            sensor_label = "OLI-2"
            default_prefix = "L9_Composite"
        collection_label = "landsat-c2-l2"
    else:
        is_landsat = False
        satellite_label = "Sentinel-2A"
        sensor_label = "MSI"
        default_bands = ORDERED_BAND_LIST
        default_res = 10.0
        default_prefix = "S2A_Composite"
        collection_label = "sentinel-2-c1-l2a"

    eff_pixel_size = float(pixel_size) if pixel_size is not None else default_res
    prefix = file_prefix if file_prefix is not None else default_prefix
    target_bands = bands_to_process if bands_to_process is not None else default_bands

    # Sort scenes chronologically
    sorted_scenes = sorted(scenes_meta, key=lambda s: s.get("datetime", ""))
    date_start = sorted_scenes[0]["date"].replace("-", "")
    date_end = sorted_scenes[-1]["date"].replace("-", "")

    if feedback:
        feedback.pushInfo(f"=== Starting Temporal Median Composite ({len(sorted_scenes)} scenes) ===")
        feedback.pushInfo(f"Satellite: {satellite_label} | Range: {sorted_scenes[0]['date']} to {sorted_scenes[-1]['date']} | Res: {eff_pixel_size}m")
        feedback.pushInfo(f"Water-Only: {water_only} (Policy: {water_temporal_policy}, Threshold: {water_occurrence_threshold}%)")

    # 2. Build Target Grid
    grid = build_target_grid(aoi_geojson, target_crs=target_crs, pixel_size=eff_pixel_size)
    height, width = grid["height"], grid["width"]
    aoi_poly_mask = grid["aoi_mask"]

    if feedback:
        feedback.pushInfo(f"Target Grid: {width}x{height} pixels | CRS: {grid['crs']} | Chunk Size: {chunk_size}x{chunk_size}")

    num_bands = len(target_bands)
    composite_bands = [np.full((height, width), -9999.0, dtype=np.float32) for _ in range(num_bands)]
    valid_count_raster = np.zeros((height, width), dtype=np.int16)
    temp_std_raster = np.full((height, width), -9999.0, dtype=np.float32)
    water_occurrence_raster = np.full((height, width), -9999.0, dtype=np.float32)
    final_water_mask = np.zeros((height, width), dtype=np.uint8)

    # 3. Chunk-Based Temporal Processing
    num_chunks_y = int(np.ceil(height / chunk_size))
    num_chunks_x = int(np.ceil(width / chunk_size))
    total_chunks = num_chunks_y * num_chunks_x
    chunk_idx = 0

    for cy in range(num_chunks_y):
        y0 = cy * chunk_size
        y1 = min(height, (cy + 1) * chunk_size)
        ch_h = y1 - y0

        for cx in range(num_chunks_x):
            x0 = cx * chunk_size
            x1 = min(width, (cx + 1) * chunk_size)
            ch_w = x1 - x0

            chunk_idx += 1
            if feedback and hasattr(feedback, "isCanceled") and feedback.isCanceled():
                raise ProcessorError("Temporal Composite processing canceled by user.")

            chunk_aoi = aoi_poly_mask[y0:y1, x0:x1]
            if not np.any(chunk_aoi):
                continue

            if feedback:
                feedback.pushInfo(f"Processing chunk [{chunk_idx}/{total_chunks}] ({ch_w}x{ch_h} px)...")
                if hasattr(feedback, "setProgress"):
                    feedback.setProgress(int((chunk_idx / total_chunks) * 100))

            chunk_transform = grid["transform"] * rasterio.Affine.translation(x0, y0)

            band_stacks = {b_id: [] for b_id in target_bands}
            water_stacks = []
            valid_observation_counts = np.zeros((ch_h, ch_w), dtype=np.int16)

            # Process each scene for this chunk
            for s_meta in sorted_scenes:
                cloud_invalid = np.zeros((ch_h, ch_w), dtype=bool)
                scene_is_water = None
                scl_chunk = None
                qa_chunk = None
                radsat_chunk = None

                if not is_landsat:
                    # Sentinel-2 SCL
                    if "SCL" in s_meta["assets"]:
                        try:
                            scl_chunk, _ = read_raster_into_grid(
                                s_meta["assets"]["SCL"]["href"],
                                target_crs=grid["crs"],
                                target_transform=chunk_transform,
                                target_shape=(ch_h, ch_w),
                                is_classification=True,
                                src_nodata=0,
                                target_nodata=0
                            )
                            for c in [0, 1, 3, 8, 9, 10, 11]:
                                cloud_invalid |= (scl_chunk == c)
                            scene_is_water = (scl_chunk == 6)
                        except Exception:
                            scl_chunk = None
                else:
                    # Landsat QA_PIXEL & QA_RADSAT
                    if "QA_PIXEL" in s_meta["assets"]:
                        try:
                            qa_chunk, _ = read_raster_into_grid(
                                s_meta["assets"]["QA_PIXEL"]["href"],
                                target_crs=grid["crs"],
                                target_transform=chunk_transform,
                                target_shape=(ch_h, ch_w),
                                is_classification=True,
                                src_nodata=1,
                                target_nodata=1
                            )
                            qa_dec = decode_landsat_qa_pixel(qa_chunk)
                            cloud_invalid |= qa_dec["cloud_invalid_mask"]
                            scene_is_water = qa_dec["water"]
                        except Exception:
                            qa_chunk = None

                    if "QA_RADSAT" in s_meta["assets"]:
                        try:
                            radsat_chunk, _ = read_raster_into_grid(
                                s_meta["assets"]["QA_RADSAT"]["href"],
                                target_crs=grid["crs"],
                                target_transform=chunk_transform,
                                target_shape=(ch_h, ch_w),
                                is_classification=True,
                                src_nodata=0,
                                target_nodata=0
                            )
                            cloud_invalid |= decode_landsat_qa_radsat(radsat_chunk)
                        except Exception:
                            radsat_chunk = None

                # Read and normalize spectral bands for chunk
                scene_bands_valid = True
                chunk_band_refls = {}

                for b_id in target_bands:
                    if b_id not in s_meta["assets"]:
                        scene_bands_valid = False
                        break
                    
                    b_meta = s_meta["assets"][b_id]
                    try:
                        raw_chunk, _ = read_raster_into_grid(
                            b_meta["href"],
                            target_crs=grid["crs"],
                            target_transform=chunk_transform,
                            target_shape=(ch_h, ch_w),
                            is_classification=False,
                            src_nodata=b_meta.get("nodata", 0),
                            target_nodata=0
                        )
                    except Exception:
                        scene_bands_valid = False
                        break

                    if is_landsat:
                        refl_chunk, v_mask, _ = normalize_landsat_surface_reflectance(
                            raw_chunk,
                            scale=b_meta.get("scale", 0.0000275),
                            offset=b_meta.get("offset", -0.2),
                            raw_nodata=b_meta.get("nodata", 0),
                            output_nodata=np.nan
                        )
                    else:
                        refl_chunk, v_mask, _ = normalize_surface_reflectance(
                            raw_chunk,
                            scale=b_meta.get("scale", 0.0001),
                            offset=b_meta.get("offset", None),
                            raw_nodata=b_meta.get("nodata", 0),
                            processing_baseline=s_meta.get("processing_baseline"),
                            boa_offset_applied=s_meta.get("boa_offset_applied"),
                            output_nodata=np.nan
                        )

                    bad_px = (~v_mask) | cloud_invalid | (~chunk_aoi)
                    refl_chunk[bad_px] = np.nan
                    chunk_band_refls[b_id] = refl_chunk

                if not scene_bands_valid or not chunk_band_refls:
                    continue

                # Water masking for scene observation
                if is_landsat:
                    green_k, nir_k, swir_k = "SR_B3", "SR_B5", "SR_B6"
                else:
                    green_k, nir_k, swir_k = "B03", "B08", "B11"

                if green_k in chunk_band_refls and nir_k in chunk_band_refls:
                    green_ch = chunk_band_refls[green_k]
                    nir_ch = chunk_band_refls[nir_k]
                    swir_ch = chunk_band_refls.get(swir_k, None)
                    w_mask, _, _ = create_water_mask(
                        scl=scl_chunk,
                        qa_pixel=qa_chunk,
                        qa_radsat=radsat_chunk,
                        green_refl=green_ch,
                        nir_refl=nir_ch,
                        swir1_refl=swir_ch,
                        valid_mask=chunk_aoi & ~cloud_invalid,
                        method="hybrid",
                        mode="permissive"
                    )
                    water_stacks.append(w_mask)
                elif scene_is_water is not None:
                    water_stacks.append(scene_is_water & chunk_aoi & ~cloud_invalid)

                for b_id in target_bands:
                    band_stacks[b_id].append(chunk_band_refls[b_id])

            # Compute median for this chunk
            if not band_stacks[target_bands[0]]:
                continue

            ref_stack = np.array(band_stacks[target_bands[0]], dtype=np.float32)
            valid_counts = np.sum(np.isfinite(ref_stack), axis=0)
            valid_observation_counts[chunk_aoi] = valid_counts[chunk_aoi]
            valid_count_raster[y0:y1, x0:x1] = valid_observation_counts

            # Water Occurrence %
            if water_stacks:
                water_stack_arr = np.array(water_stacks, dtype=bool)
                water_count = np.sum(water_stack_arr, axis=0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    occ_pct = np.where(valid_counts > 0, (water_count / valid_counts) * 100.0, -9999.0)
                water_occurrence_raster[y0:y1, x0:x1] = np.where(chunk_aoi, occ_pct, -9999.0)

                if water_temporal_policy == "consensus":
                    is_water_chunk = (occ_pct >= 90.0)
                elif water_temporal_policy == "occurrence_threshold":
                    is_water_chunk = (occ_pct >= float(water_occurrence_threshold))
                else:  # hybrid
                    is_water_chunk = (occ_pct >= max(20.0, float(water_occurrence_threshold) * 0.6))
            else:
                is_water_chunk = np.ones((ch_h, ch_w), dtype=bool)

            for b_idx, b_id in enumerate(target_bands):
                stack_3d = np.array(band_stacks[b_id], dtype=np.float32)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    with np.errstate(all="ignore"):
                        med_2d = np.nanmedian(stack_3d, axis=0)
                        if b_idx == 0:
                            std_2d = np.nanstd(stack_3d, axis=0)
                            std_2d[~chunk_aoi | (valid_counts < 2)] = -9999.0
                            temp_std_raster[y0:y1, x0:x1] = std_2d

                insufficient = (valid_counts < min_valid_observations)
                med_2d[insufficient | ~chunk_aoi] = -9999.0

                if water_only:
                    med_2d[~is_water_chunk | ~chunk_aoi] = -9999.0

                composite_bands[b_idx][y0:y1, x0:x1] = med_2d

            final_water_mask[y0:y1, x0:x1] = np.where(is_water_chunk & chunk_aoi, 1, 0)
            final_water_mask[y0:y1, x0:x1][~chunk_aoi] = 255

    # 4. Export Composite GeoTIFF
    os.makedirs(output_dir, exist_ok=True)
    mode_tag = "Median_WaterOnly" if water_only else "Median_BOA_Float32"
    comp_filename = f"{prefix}_{date_start}_{date_end}_{mode_tag}.tif"
    comp_path = os.path.join(output_dir, comp_filename)

    if feedback:
        feedback.pushInfo(f"Exporting Multiband Float32 Composite GeoTIFF to: {comp_path}")

    scale_val = 0.0000275 if is_landsat else 0.0001
    offset_val = -0.2 if is_landsat else -0.1

    save_multiband_geotiff(
        comp_path,
        band_arrays=composite_bands,
        band_names=target_bands,
        grid=grid,
        nodata_val=-9999.0,
        metadata_extra={
            "COMPOSITE_TYPE": "Temporal Pixel-wise Median Surface Reflectance",
            "SATELLITE": satellite_label,
            "SENSOR": sensor_label,
            "DATE_START": sorted_scenes[0]["date"],
            "DATE_END": sorted_scenes[-1]["date"],
            "NUM_SCENES": len(sorted_scenes),
            "WATER_ONLY_APPLIED": str(water_only),
            "WATER_POLICY": water_temporal_policy,
            "WATER_OCCURRENCE_THRESHOLD": str(water_occurrence_threshold),
            "SCALE_FACTOR": str(scale_val),
            "OFFSET": str(offset_val),
            "PIXEL_SIZE": str(eff_pixel_size)
        }
    )

    outputs = {
        "composite_reflectance": comp_path,
        "satellite": satellite_label,
        "date_start": sorted_scenes[0]["date"],
        "date_end": sorted_scenes[-1]["date"],
        "num_scenes": len(sorted_scenes),
        "bands": target_bands
    }

    # Save Quality Rasters
    if save_valid_count_raster:
        vc_path = os.path.join(output_dir, f"{prefix}_{date_start}_{date_end}_ValidCount.tif")
        save_singleband_raster(
            vc_path,
            valid_count_raster,
            grid=grid,
            dtype="int16",
            nodata_val=-1,
            description="Valid Observations Count per Pixel"
        )
        outputs["valid_count_raster"] = vc_path

    if save_temp_std_raster:
        std_path = os.path.join(output_dir, f"{prefix}_{date_start}_{date_end}_TemporalStd.tif")
        save_singleband_raster(
            std_path,
            temp_std_raster,
            grid=grid,
            dtype="float32",
            nodata_val=-9999.0,
            description="Temporal Standard Deviation"
        )
        outputs["temporal_std_raster"] = std_path

    if save_water_occurrence_raster:
        occ_path = os.path.join(output_dir, f"{prefix}_{date_start}_{date_end}_WaterOccurrence.tif")
        save_singleband_raster(
            occ_path,
            water_occurrence_raster,
            grid=grid,
            dtype="float32",
            nodata_val=-9999.0,
            description="Water Occurrence Percentage (0 - 100%)"
        )
        outputs["water_occurrence_raster"] = occ_path

    # Save Composite Water Mask
    wm_path = os.path.join(output_dir, f"{prefix}_{date_start}_{date_end}_WaterMask.tif")
    save_singleband_raster(
        wm_path,
        final_water_mask,
        grid=grid,
        dtype="uint8",
        nodata_val=255,
        description="Composite Water Mask: 1=Water, 0=Land, 255=NoData"
    )
    outputs["water_mask"] = wm_path

    # Save Processing Metadata JSON
    json_path = os.path.join(output_dir, f"{prefix}_{date_start}_{date_end}_Processing_Metadata.json")
    meta_doc = {
        "product": f"{satellite_label} Temporal Median Composite",
        "satellite": satellite_label,
        "sensor": sensor_label,
        "stac_collection": collection_label,
        "date_range": [sorted_scenes[0]["date"], sorted_scenes[-1]["date"]],
        "total_source_scenes": len(sorted_scenes),
        "source_scene_ids": [s["id"] for s in sorted_scenes],
        "source_scene_dates": [s["date"] for s in sorted_scenes],
        "source_cloud_covers": [s["cloud_cover"] for s in sorted_scenes],
        "bands": target_bands,
        "spatial_resolution_meters": eff_pixel_size,
        "crs": str(grid["crs"]),
        "dimensions": [width, height],
        "input_data_type": "uint16",
        "output_data_type": "float32",
        "scale_factor": scale_val,
        "offset": offset_val,
        "scaling_applied_by": "Bathymetrix-AI STAC Reader (Surface Reflectance, already scaled)",
        "water_only": water_only,
        "water_policy": water_temporal_policy,
        "water_occurrence_threshold": water_occurrence_threshold,
        "min_valid_observations": min_valid_observations,
        "generation_timestamp": datetime.now().isoformat(),
        "export_path": comp_path
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta_doc, f, indent=2)
    outputs["metadata_json"] = json_path

    if feedback:
        feedback.pushInfo(f"=== Temporal Median Composite Completed Successfully: {comp_path} ===")

    return outputs
