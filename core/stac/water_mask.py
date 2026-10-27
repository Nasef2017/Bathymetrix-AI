# -*- coding: utf-8 -*-
"""
core/stac/water_mask.py
Water Detection and Masking Engine for:
1. Sentinel-2A Level-2A (SCL Scene Classification Layer)
2. Landsat-8 and Landsat-9 Collection 2 Level-2 (QA_PIXEL, QA_RADSAT, SR_QA_AEROSOL)

Supports:
- SCL Water Extraction with Cloud/Shadow/Invalid Exclusion
- Landsat QA_PIXEL Bitmask Decoding (Fill, Clouds, Dilated Clouds, Cirrus, Cloud Shadows, Snow, Water)
- Landsat QA_RADSAT Radiometric Saturation Decoding
- Landsat SR_QA_AEROSOL Aerosol Level Decoding
- Spectral Water Indices (NDWI, MNDWI) with Manual or Otsu Thresholding
- Smart Hybrid Water Masking (Strict vs Permissive Coastal)
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional


# SCL Classification Values (Sentinel-2):
# 0: NoData, 1: Saturated/Defective, 2: Dark Area, 3: Cloud Shadows, 4: Veg, 5: Not Veg,
# 6: Water, 7: Unclassified, 8: Cloud Med, 9: Cloud High, 10: Thin Cirrus, 11: Snow/Ice
SCL_WATER = 6
SCL_INVALID_OR_CLOUD = {0, 1, 3, 8, 9, 10, 11}
SCL_CAUTION = {2, 7}


def decode_landsat_qa_pixel(qa_array: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Decode USGS Landsat Collection 2 QA_PIXEL 16-bit integer bitmask.
    
    Bit Definitions:
      Bit 0: Fill (1 = Fill / NoData)
      Bit 1: Dilated Cloud (1 = Dilated Cloud)
      Bit 2: Cirrus (1 = High confidence cirrus)
      Bit 3: Cloud (1 = Cloud)
      Bit 4: Cloud Shadow (1 = Cloud Shadow)
      Bit 5: Snow/Ice (1 = Snow / Ice)
      Bit 6: Clear (1 = Clear sky, 0 = Cloud/Shadow/Dilated)
      Bit 7: Water (1 = Water, 0 = Land)
      Bits 8-9: Cloud Confidence (00: None, 01: Low, 10: Medium, 11: High)
      Bits 10-11: Cloud Shadow Confidence (00: None, 01: Low, 10: Reserved, 11: High)
      Bits 12-13: Snow/Ice Confidence (00: None, 01: Low, 10: Reserved, 11: High)
      Bits 14-15: Cirrus Confidence (00: None, 01: Low, 10: Reserved, 11: High)
      
    Returns:
        Dictionary of boolean masks including 'water' and 'cloud_invalid_mask'.
    """
    qa = qa_array.astype(np.uint16)
    fill = (qa & (1 << 0)) > 0
    dilated_cloud = (qa & (1 << 1)) > 0
    cirrus = (qa & (1 << 2)) > 0
    cloud = (qa & (1 << 3)) > 0
    cloud_shadow = (qa & (1 << 4)) > 0
    snow = (qa & (1 << 5)) > 0
    clear = (qa & (1 << 6)) > 0
    water = (qa & (1 << 7)) > 0

    cloud_conf = (qa >> 8) & 0b11
    shadow_conf = (qa >> 10) & 0b11
    snow_conf = (qa >> 12) & 0b11
    cirrus_conf = (qa >> 14) & 0b11

    # Medium and High confidence flags
    med_high_cloud = (cloud_conf >= 2)
    high_shadow = (shadow_conf >= 2)
    med_high_cirrus = (cirrus_conf >= 2)
    high_snow = (snow_conf >= 2)

    # Combined cloud/shadow/invalid mask (never reject valid water)
    cloud_invalid = fill | dilated_cloud | cirrus | cloud | cloud_shadow | snow | med_high_cloud | high_shadow | med_high_cirrus | high_snow

    return {
        "fill": fill,
        "dilated_cloud": dilated_cloud,
        "cirrus": cirrus,
        "cloud": cloud,
        "cloud_shadow": cloud_shadow,
        "snow": snow,
        "clear": clear,
        "water": water,
        "cloud_conf": cloud_conf,
        "shadow_conf": shadow_conf,
        "cirrus_conf": cirrus_conf,
        "cloud_invalid_mask": cloud_invalid
    }


def decode_landsat_qa_radsat(radsat_array: np.ndarray) -> np.ndarray:
    """
    Decode Landsat QA_RADSAT bitmask for radiometric saturation.
    Bits 0-6 represent saturation for optical bands 1 to 7.
    Returns boolean mask: True where radiometric saturation is present on optical bands.
    """
    rad = radsat_array.astype(np.uint16)
    optical_saturation = (rad & 0x7F) > 0
    return optical_saturation


def decode_landsat_qa_aerosol(aerosol_array: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Decode Landsat SR_QA_AEROSOL 8-bit integer bitmask.
    Bit 0: Fill
    Bit 1: Valid aerosol retrieval
    Bits 6-7: Aerosol level (00: Climatology, 01: Low, 10: Medium, 11: High)
    """
    aer = aerosol_array.astype(np.uint8)
    fill = (aer & (1 << 0)) > 0
    valid_retrieval = (aer & (1 << 1)) > 0
    level = (aer >> 6) & 0b11
    high_aerosol = (level == 3)
    return {
        "fill": fill,
        "valid_retrieval": valid_retrieval,
        "aerosol_level": level,
        "high_aerosol": high_aerosol
    }


def compute_otsu_threshold(values: np.ndarray, num_bins: int = 256) -> float:
    """Compute Otsu threshold on a 1D array of index values."""
    if len(values) < 10:
        return 0.0

    # Clip to valid index bounds
    clipped = np.clip(values, -1.0, 1.0)
    hist, bin_edges = np.histogram(clipped, bins=num_bins, range=(-1.0, 1.0))
    hist = hist.astype(float)
    total = hist.sum()
    if total <= 0:
        return 0.0

    current_max = 0.0
    threshold = 0.0
    sum_total = np.dot(np.arange(num_bins), hist)
    sum_b = 0.0
    w_b = 0.0

    for i in range(num_bins):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break

        sum_b += i * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f

        between_var = w_b * w_f * ((m_b - m_f) ** 2)
        if between_var > current_max:
            current_max = between_var
            threshold = (bin_edges[i] + bin_edges[i + 1]) / 2.0

    return float(threshold)


def compute_spectral_index(
    band_numerator: np.ndarray,
    band_denominator: np.ndarray,
    valid_mask: np.ndarray
) -> np.ndarray:
    """Compute normalized difference spectral index: (Num - Denom) / (Num + Denom)."""
    index = np.full(band_numerator.shape, np.nan, dtype=np.float32)
    denom = band_numerator + band_denominator
    safe_mask = valid_mask & np.isfinite(denom) & (np.abs(denom) > 1e-6)
    index[safe_mask] = (band_numerator[safe_mask] - band_denominator[safe_mask]) / denom[safe_mask]
    return index


def create_water_mask(
    scl: Optional[np.ndarray] = None,
    qa_pixel: Optional[np.ndarray] = None,
    qa_radsat: Optional[np.ndarray] = None,
    green_refl: Optional[np.ndarray] = None,
    nir_refl: Optional[np.ndarray] = None,
    swir1_refl: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    method: str = "hybrid",  # 'scl', 'qa', 'ndwi', 'mndwi', 'hybrid'
    mode: str = "permissive",  # 'strict' or 'permissive' (coastal friendly)
    threshold: Optional[float] = None,  # If None, Otsu is used when applicable
    use_otsu: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate high-accuracy water mask for Sentinel-2A or Landsat-8/9.
    
    Parameters:
        scl: Sentinel-2 Scene Classification Layer (UInt8).
        qa_pixel: Landsat-8/9 QA_PIXEL bitmask (UInt16).
        qa_radsat: Landsat-8/9 QA_RADSAT bitmask (UInt16).
        green_refl: Green band reflectance (S2: B03, Landsat: SR_B3).
        nir_refl: NIR band reflectance (S2: B08, Landsat: SR_B5).
        swir1_refl: SWIR 1 band reflectance (S2: B11, Landsat: SR_B6).
        valid_mask: Boolean array of valid AOI pixels.
        method: Masking method ('hybrid', 'scl', 'ndwi', 'mndwi').
        mode: Strictness mode ('permissive' or 'strict').
        threshold: Manual threshold, or None for Otsu.
        use_otsu: If True and threshold is None, compute Otsu threshold.
        
    Returns:
        (water_boolean_mask, water_uint8_raster, statistics_dict)
        where water_uint8_raster:
          1 = Confirmed Water
          0 = Non-Water (Land)
          255 = Invalid / Cloud / NoData
    """
    # Establish overall shape
    shape = None
    for arr in [scl, qa_pixel, qa_radsat, green_refl, nir_refl, swir1_refl, valid_mask]:
        if arr is not None:
            shape = arr.shape
            break
            
    if shape is None:
        raise ValueError("At least one raster array must be provided to create_water_mask.")

    if valid_mask is None:
        valid_mask = np.ones(shape, dtype=bool)
    else:
        valid_mask = valid_mask.copy()

    cloud_invalid_mask = np.zeros(shape, dtype=bool)
    aux_water_mask = np.zeros(shape, dtype=bool)
    aux_source_name = "None"

    # 1. Decode QA from Sentinel-2 SCL if available
    if scl is not None:
        aux_source_name = "SCL"
        for c in SCL_INVALID_OR_CLOUD:
            cloud_invalid_mask |= (scl == c)
        aux_water_mask = (scl == SCL_WATER)
        valid_mask &= (scl != 0)

    # 2. Decode QA from Landsat QA_PIXEL / QA_RADSAT if available
    elif qa_pixel is not None:
        aux_source_name = "QA_PIXEL"
        qa_info = decode_landsat_qa_pixel(qa_pixel)
        cloud_invalid_mask = qa_info["cloud_invalid_mask"]
        aux_water_mask = qa_info["water"]
        valid_mask &= (~qa_info["fill"])

        # Include radiometric saturation if QA_RADSAT provided
        if qa_radsat is not None:
            sat_mask = decode_landsat_qa_radsat(qa_radsat)
            cloud_invalid_mask |= sat_mask

    # 3. Spectral Water Indices (NDWI & MNDWI)
    spectral_water_mask = np.zeros(shape, dtype=bool)
    computed_threshold = 0.0
    index_name = "None"
    index_values = None

    method_clean = method.lower().strip()

    if method_clean in ["ndwi", "mndwi", "hybrid"]:
        if method_clean == "mndwi" and green_refl is not None and swir1_refl is not None:
            index_name = "MNDWI"
            index_values = compute_spectral_index(green_refl, swir1_refl, valid_mask & ~cloud_invalid_mask)
        elif green_refl is not None and nir_refl is not None:
            index_name = "NDWI"
            index_values = compute_spectral_index(green_refl, nir_refl, valid_mask & ~cloud_invalid_mask)

        if index_values is not None:
            valid_idx_vals = index_values[np.isfinite(index_values)]
            if use_otsu and (threshold is None):
                computed_threshold = compute_otsu_threshold(valid_idx_vals)
            else:
                computed_threshold = float(threshold) if threshold is not None else 0.0

            spectral_water_mask = (index_values > computed_threshold)

    # 4. Decision Logic based on Selected Method
    if method_clean in ["scl", "qa"]:
        if scl is None and qa_pixel is None:
            raise ValueError(f"SCL or QA_PIXEL array required for method='{method}'")
        water_mask = aux_water_mask & valid_mask & ~cloud_invalid_mask

    elif method_clean in ["ndwi", "mndwi"]:
        if index_values is None:
            raise ValueError(f"Required reflectance bands not provided for {method_clean}")
        water_mask = spectral_water_mask & valid_mask & ~cloud_invalid_mask

    else:  # hybrid
        has_aux = (scl is not None or qa_pixel is not None)
        if has_aux and index_values is not None:
            if mode.lower() == "strict":
                # Strict: Must be classified as water by SCL/QA AND pass spectral threshold, no cloud
                water_mask = aux_water_mask & spectral_water_mask & valid_mask & ~cloud_invalid_mask
            else:
                # Permissive (Coastal Friendly):
                # Accepts (SCL/QA water) OR (Spectral water AND not high probability cloud/shadow).
                # This protects turbid coastal water and shallow reefs where QA may misclassify.
                green_valid = (green_refl > 0.005) if green_refl is not None else True
                water_mask = valid_mask & ~cloud_invalid_mask & (aux_water_mask | (spectral_water_mask & green_valid))
        elif has_aux:
            water_mask = aux_water_mask & valid_mask & ~cloud_invalid_mask
        elif index_values is not None:
            water_mask = spectral_water_mask & valid_mask
        else:
            water_mask = valid_mask

    # 5. Create Categorical UInt8 Water Mask
    # 1 = Water, 0 = Land, 255 = Invalid/Cloud/NoData
    water_uint8 = np.zeros(shape, dtype=np.uint8)
    water_uint8[water_mask] = 1
    water_uint8[~valid_mask | cloud_invalid_mask] = 255

    # 6. Statistics
    total_pixels = int(np.prod(shape))
    valid_pixels = int(np.sum(valid_mask & ~cloud_invalid_mask))
    water_pixels = int(np.sum(water_mask))
    land_pixels = max(0, valid_pixels - water_pixels)
    invalid_pixels = total_pixels - valid_pixels

    water_pct = round((water_pixels / total_pixels) * 100.0, 2) if total_pixels > 0 else 0.0
    invalid_pct = round((invalid_pixels / total_pixels) * 100.0, 2) if total_pixels > 0 else 0.0

    stats = {
        "method": method,
        "mode": mode,
        "aux_qa_source": aux_source_name,
        "index_used": index_name,
        "threshold": round(computed_threshold, 4),
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "water_pixels": water_pixels,
        "land_pixels": land_pixels,
        "invalid_pixels": invalid_pixels,
        "water_percentage": water_pct,
        "invalid_percentage": invalid_pct
    }

    return water_mask, water_uint8, stats
