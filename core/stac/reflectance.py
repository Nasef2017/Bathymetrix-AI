# -*- coding: utf-8 -*-
"""
core/stac/reflectance.py
Robust Bottom-of-Atmosphere (BOA) Surface Reflectance Normalization for:
1. Sentinel-2A Level-2A (MSI): Scale 0.0001, Baseline checks (PB >= 04.00), NoData protection.
2. Landsat-8 and Landsat-9 Collection 2 Level-2 (OLI/OLI-2): Scale 0.0000275, Offset -0.2, Fill 0 -> -9999.0.

Prevents double-scaling, strictly protects NoData/Fill pixels, preserves valid small negative
reflectances, never clips blindly to [0,1], and outputs physical Float32 reflectance.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional


def parse_processing_baseline(baseline_str: Optional[str]) -> Optional[float]:
    """Parse baseline string such as '04.00', '05.13', or '02.04' into a float."""
    if not baseline_str or str(baseline_str).strip() in ["N/A", "None", ""]:
        return None
    try:
        clean = str(baseline_str).strip().replace("V", "").replace("v", "")
        parts = clean.split(".")
        if len(parts) >= 2:
            return float(f"{int(parts[0])}.{parts[1]}")
        return float(parts[0])
    except Exception:
        return None


def resolve_reflectance_transform(
    scale_from_metadata: Optional[float] = 0.0001,
    offset_from_metadata: Optional[float] = None,
    processing_baseline_str: Optional[str] = None,
    boa_offset_applied: Optional[bool] = None,
) -> Tuple[float, float, str]:
    """
    Determine the exact physical scale and offset for converting Sentinel-2 quantized DN to Float32 Surface Reflectance.
    
    Returns:
        (scale, offset, rationale)
    """
    scale = float(scale_from_metadata) if scale_from_metadata is not None else 0.0001
    if scale <= 0:
        scale = 0.0001

    pb = parse_processing_baseline(processing_baseline_str)
    
    # Case 1: Earth Search explicitly states boa_offset_applied == True
    if boa_offset_applied is True:
        return scale, 0.0, "boa_offset_applied is True; DN offset already compensated."

    # Case 2: Metadata explicitly provides offset
    meta_offset = float(offset_from_metadata) if offset_from_metadata is not None else None

    # Case 3: Processing Baseline check
    if pb is not None:
        if pb >= 4.0:
            # Baseline 04.00+ introduced the -1000 DN (-0.1 reflectance) radiometry shift
            offset = meta_offset if (meta_offset is not None and meta_offset != 0) else -0.1
            return scale, offset, f"Baseline {pb:.2f} >= 04.00: applied BOA offset {offset}."
        else:
            # Pre-04.00 baselines do not have the radiometry shift
            return scale, 0.0, f"Baseline {pb:.2f} < 04.00: no BOA radiometry offset needed (offset=0)."

    # Case 4: Baseline unknown, rely on metadata offset if present
    if meta_offset is not None and abs(meta_offset) > 1e-6:
        return scale, meta_offset, f"Metadata specifies offset={meta_offset}."

    # Default fallback for recent Sentinel-2 Collection 1 products
    return scale, -0.1, "Default fallback for Sentinel-2 Collection 1 (PB >= 04.00, offset=-0.1)."


def normalize_surface_reflectance(
    dn_array: np.ndarray,
    scale: Optional[float] = 0.0001,
    offset: Optional[float] = None,
    raw_nodata: float = 0.0,
    processing_baseline: Optional[str] = None,
    boa_offset_applied: Optional[bool] = None,
    output_nodata: float = -9999.0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Convert raw Sentinel-2 quantized DN array to physical Float32 Surface Reflectance.
    Preserved exactly as original implementation.
    """
    # 1. Check if array is already normalized surface reflectance (float with small values)
    if np.issubdtype(dn_array.dtype, np.floating):
        finite_vals = dn_array[np.isfinite(dn_array) & (dn_array != output_nodata)]
        if len(finite_vals) > 0:
            p95 = np.percentile(finite_vals, 95)
            # If 95% of data is <= 2.5 and min is >= -0.5, it's already normalized float reflectance!
            if p95 <= 2.5 and np.min(finite_vals) >= -0.5:
                valid_m = np.isfinite(dn_array) & (dn_array != output_nodata)
                refl = dn_array.astype(np.float32)
                refl[~valid_m] = output_nodata
                return refl, valid_m, {
                    "sensor": "Sentinel-2A MSI",
                    "applied_scale": 1.0,
                    "applied_offset": 0.0,
                    "rationale": "Data already detected as normalized Float32 surface reflectance (bypassed conversion).",
                    "valid_pixel_pct": round(float(np.mean(valid_m)) * 100.0, 2)
                }

    # 2. Identify NoData / Off-swath pixels
    nodata_mask = (dn_array == raw_nodata) | (dn_array <= 0) | np.isnan(dn_array)
    valid_mask = ~nodata_mask

    # 3. Determine scale and offset
    eff_scale, eff_offset, rationale = resolve_reflectance_transform(
        scale_from_metadata=scale,
        offset_from_metadata=offset,
        processing_baseline_str=processing_baseline,
        boa_offset_applied=boa_offset_applied
    )

    # 4. Perform Float32 conversion only on valid pixels
    out_refl = np.full(dn_array.shape, output_nodata, dtype=np.float32)
    
    if np.any(valid_mask):
        valid_dns = dn_array[valid_mask].astype(np.float32)
        calc_refl = valid_dns * eff_scale + eff_offset

        # Heuristic sanity check:
        # If applying eff_offset (-0.1) caused over 70% of valid pixels to be negative,
        # then the DNs did NOT contain the +1000 baseline shift (pre-04.00 or already compensated).
        # Fall back to offset = 0.0!
        if eff_offset < -0.01:
            neg_fraction = np.mean(calc_refl < -0.02)
            if neg_fraction > 0.70:
                eff_offset = 0.0
                calc_refl = valid_dns * eff_scale
                rationale += " [Fallback: >70% negative reflectances detected, adjusted offset to 0.0 to prevent underflow]"

        out_refl[valid_mask] = calc_refl

    valid_pct = round(float(np.mean(valid_mask)) * 100.0, 2)
    
    provenance = {
        "sensor": "Sentinel-2A MSI",
        "applied_scale": eff_scale,
        "applied_offset": eff_offset,
        "rationale": rationale,
        "valid_pixel_pct": valid_pct,
        "processing_baseline": processing_baseline,
        "boa_offset_applied": boa_offset_applied
    }

    return out_refl, valid_mask, provenance


def normalize_landsat_surface_reflectance(
    dn_array: np.ndarray,
    scale: Optional[float] = 0.0000275,
    offset: Optional[float] = -0.2,
    raw_nodata: float = 0.0,
    output_nodata: float = -9999.0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Convert raw Landsat Collection 2 Level-2 quantized UInt16 DN array to physical Float32 Surface Reflectance.
    
    Formula:
        Surface Reflectance = DN * 0.0000275 - 0.2
        
    Strict Requirements & Safeguards:
    1. Inspect metadata for scale (0.0000275) and offset (-0.2).
    2. Determine whether scaling has already been applied (prevent double scaling).
    3. Apply scale and offset exactly once.
    4. Convert valid reflectance values to Float32.
    5. Mask Fill/NoData (DN == 0) BEFORE scaling. Never convert Fill/NoData into valid reflectance (-0.2).
    6. Preserve physically plausible negative reflectances where relevant (e.g. Rayleigh over-correction).
    7. Never clip all reflectance values automatically to [0, 1].
    8. Never treat UInt16-to-Float32 conversion alone as reflectance scaling.
    9. Do not apply this Landsat formula to Sentinel-2A.
    
    Parameters:
        dn_array: Input 2D numpy array (UInt16 or Float32).
        scale: Scale factor (default: 0.0000275).
        offset: Offset value (default: -0.2).
        raw_nodata: Source Fill/NoData value (default: 0.0).
        output_nodata: Destination NoData value (default: -9999.0).
        
    Returns:
        (reflectance_float32, valid_mask, provenance_dict)
    """
    eff_scale = float(scale) if (scale is not None and scale > 0) else 0.0000275
    eff_offset = float(offset) if offset is not None else -0.2

    # 1. Check if array is already normalized surface reflectance (Float32 with values in plausible reflectance range)
    if np.issubdtype(dn_array.dtype, np.floating):
        finite_vals = dn_array[np.isfinite(dn_array) & (dn_array != output_nodata)]
        if len(finite_vals) > 0:
            p95 = np.percentile(finite_vals, 95)
            # If 95% of data is <= 2.5 and min is >= -0.5, it's already physical surface reflectance!
            if p95 <= 2.5 and np.min(finite_vals) >= -0.5:
                valid_m = np.isfinite(dn_array) & (dn_array != output_nodata)
                refl = dn_array.astype(np.float32)
                refl[~valid_m] = output_nodata
                return refl, valid_m, {
                    "sensor": "Landsat-8/9 OLI Collection 2 Level-2",
                    "applied_scale": 1.0,
                    "applied_offset": 0.0,
                    "rationale": "Data already detected as normalized Float32 surface reflectance (bypassed conversion).",
                    "valid_pixel_pct": round(float(np.mean(valid_m)) * 100.0, 2)
                }

    # 2. Strict NoData / Fill Guard: Mask Fill/NoData (DN == 0 or <= 0 or NaN) BEFORE scaling
    # In USGS Landsat Collection 2 Level-2 Surface Reflectance: 0 = Fill / NoData
    nodata_mask = (dn_array == raw_nodata) | (dn_array <= 0) | np.isnan(dn_array)
    valid_mask = ~nodata_mask

    # 3. Apply scale and offset exactly once to valid pixels only
    out_refl = np.full(dn_array.shape, output_nodata, dtype=np.float32)

    if np.any(valid_mask):
        valid_dns = dn_array[valid_mask].astype(np.float32)
        # Reflectance = DN * 0.0000275 - 0.2
        calc_refl = valid_dns * eff_scale + eff_offset
        out_refl[valid_mask] = calc_refl

    valid_pct = round(float(np.mean(valid_mask)) * 100.0, 2)

    provenance = {
        "sensor": "Landsat-8/9 OLI Collection 2 Level-2",
        "applied_scale": eff_scale,
        "applied_offset": eff_offset,
        "raw_fill_value": raw_nodata,
        "output_nodata": output_nodata,
        "formula": "Surface_Reflectance = DN * scale + offset",
        "rationale": f"USGS Landsat Collection 2 Level-2 Surface Reflectance scaling (scale={eff_scale}, offset={eff_offset}).",
        "valid_pixel_pct": valid_pct
    }

    return out_refl, valid_mask, provenance


def normalize_sensor_reflectance(
    dn_array: np.ndarray,
    platform: str = "sentinel-2a",
    scale: Optional[float] = None,
    offset: Optional[float] = None,
    raw_nodata: float = 0.0,
    processing_baseline: Optional[str] = None,
    boa_offset_applied: Optional[bool] = None,
    output_nodata: float = -9999.0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Unified dispatcher for sensor-specific surface reflectance normalization.
    Routes to normalize_surface_reflectance for Sentinel-2A and normalize_landsat_surface_reflectance for Landsat.
    """
    plat_lower = str(platform).lower().strip()
    if plat_lower in ["landsat-8", "landsat-9", "landsat"]:
        s = scale if scale is not None else 0.0000275
        o = offset if offset is not None else -0.2
        return normalize_landsat_surface_reflectance(
            dn_array=dn_array,
            scale=s,
            offset=o,
            raw_nodata=raw_nodata,
            output_nodata=output_nodata
        )
    else:
        # Sentinel-2A
        s = scale if scale is not None else 0.0001
        return normalize_surface_reflectance(
            dn_array=dn_array,
            scale=s,
            offset=offset,
            raw_nodata=raw_nodata,
            processing_baseline=processing_baseline,
            boa_offset_applied=boa_offset_applied,
            output_nodata=output_nodata
        )
