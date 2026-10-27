# Bathymetrix-AI: Satellite Imagery Downloader & Temporal Composite
### Sentinel-2A, Landsat-8, and Landsat-9 Surface Reflectance Ingestion Engine
### Technical Architecture, User Guide & Scientific Specification


> v8.0 documentation synchronization (2026-10-27). This document describes the integrated Bathymetrix-AI Satellite Imagery Downloader and Temporal Composite module. The QA table below records implementation specifications and capabilities.

---

## 1. Executive Summary

The **Satellite Imagery Downloader (Earth Search STAC)** is a dedicated, production-grade remote sensing data acquisition module integrated directly into the **Bathymetrix-AI** QGIS plugin under section **`2. Data Acquisition & Datum`**.

The tool enables coastal researchers, hydrographers, and geospatial analysts to search, stream, normalize, quality-filter, and composite Bottom-of-Atmosphere (BOA) Surface Reflectance imagery directly within QGIS. It interfaces with the **Element 84 Earth Search STAC API**, supporting exactly three optical satellite platforms:

1. **Sentinel-2A** — MSI Level-2A Surface Reflectance (Copernicus / ESA).
2. **Landsat-8** — OLI Collection 2 Level-2 Surface Reflectance (USGS / NASA).
3. **Landsat-9** — OLI-2 Collection 2 Level-2 Surface Reflectance (USGS / NASA).

> [!IMPORTANT]
> **Strict Satellite Exclusions**: In accordance with the system specification, **Sentinel-2B** and **Sentinel-2C** are strictly excluded from all search queries, satellite selectors, validation checks, and download workflows. Any legacy configuration attempting to query 2B or 2C is safely intercepted with a user notice and rejected. No other platforms (such as Sentinel-1 or older Landsat sensors) are permitted.

---

## 2. Core Architecture & Modularity

The module follows a strictly decoupled, modular architecture:

1. **Core STAC Engine (`core/stac/`)**:
   - Completely independent of PyQGIS GUI.
   - Built on standard Python, NumPy, GDAL, and Rasterio.
   - Usable in headless CLI tools, automated cloud pipelines, and WebGIS backends.
2. **QGIS Processing Algorithm (`qgis_interface/algorithms/Sentinel2_Downloader.py`)**:
   - Registered under QGIS Processing Toolbox in provider `Bathymetrix-AI` under group `2. Data Acquisition & Datum` (`data_prep`).
   - Supports automated modeler chains, QGIS Graphical Modeler, and batch processing.
   - Implements **Directory-Only** destination handling with automatic raster layer output registration for seamless QGIS canvas display.
3. **Interactive PyQt GUI Dialog (`qgis_interface/sentinel2_downloader_dialog.py`)**:
   - Rich interactive GUI accessible from `Plugins -> Bathymetrix-AI -> Satellite Imagery Downloader (Sentinel-2A & Landsat-8/9)` and toolbar icon.
   - Dynamic satellite platform selector, live STAC scene table, on-canvas ROI polygon/rectangle drawing, preview thumbnails, and multi-threaded background downloading (`QgsTask`).

```
Bathymetrix-AI
├── core/
│   └── stac/
│       ├── __init__.py           # Unified package exports
│       ├── stac_client.py        # Earth Search STAC API client, platform filters & asset resolver
│       ├── reflectance.py        # Sentinel-2A & Landsat-8/9 Surface Reflectance normalization
│       ├── water_mask.py         # SCL & Landsat QA_PIXEL bitmask decoding, NDWI/MNDWI/Hybrid
│       ├── grid.py               # UTM reprojection, grid snapping, Affine transform & polygon masking
│       └── processor.py          # Single Scene & Temporal Median Composite processing engine
├── qgis_interface/
│   ├── algorithms/
│   │   └── Sentinel2_Downloader.py       # QgsProcessingAlgorithm (Directory-Only output)
│   ├── sentinel2_downloader_dialog.py    # Interactive PyQt GUI Dialog & QgsTask
│   └── provider.py                       # Algorithm registration
├── docs/
│   └── Sentinel2_Downloader_Technical_Documentation.md
└── tests/
    └── test_sentinel2_stac.py            # Comprehensive QA test suite (19 unit/synthetic tests)
```

---

## 3. Data Source: Element 84 Earth Search STAC API

- **Endpoint**: `https://earth-search.aws.element84.com/v1`
- **Target Collections**:
  - `sentinel-2-c1-l2a`: Sentinel-2 Level-2A BOA Surface Reflectance.
  - `landsat-c2-l2`: Landsat Collection 2 Level-2 Surface Reflectance.
- **Search Parameters**:
  - Spatial Intersection: GeoJSON geometry (Bounding Box or exact polygon).
  - Temporal Window: `YYYY-MM-DD/YYYY-MM-DD`.
  - Platform Query Filters:
    - Sentinel-2A: `{"platform": {"eq": "sentinel-2a"}}`
    - Landsat-8: `{"platform": {"eq": "landsat-8"}}`
    - Landsat-9: `{"platform": {"eq": "landsat-9"}}`
    - Landsat-8 & Landsat-9: `{"platform": {"in": ["landsat-8", "landsat-9"]}}`
  - Cloud Cover Threshold: `eo:cloud_cover <= max_cloud_cover`.

---

## 4. Radiometric Calibration & Surface Reflectance Scaling

Surface reflectance assets in both Sentinel-2A and Landsat Collection 2 are stored as quantized integer digital numbers (DN) in source GeoTIFF files to optimize file size. Converting raw DN into true physical Bottom-of-Atmosphere (BOA) Surface Reflectance requires sensor-specific transformations.

### 4.1. Sentinel-2A Surface Reflectance Normalization

Raw Sentinel-2 Level-2A products store radiometric bands as quantized UInt16 integers:

$$\text{Reflectance}_{\text{S2A}} = \text{DN} \times \text{Scale} + \text{Offset}$$

- **Scale Factor**: $0.0001$ ($10000 \text{ DN} = 1.0$ reflectance).
- **Processing Baseline Safeguards**:
  - For Processing Baseline $\ge 04.00$ (and Collection 1 retroactively), ESA introduced a radiometric offset of $-1000 \text{ DN}$ (reflectance offset of $-0.1$) to prevent clipping of small negative reflectances.
  - The engine checks `s2:processing_baseline` and `earthsearch:boa_offset_applied`:
    - Baseline $\ge 04.00$ and offset not yet applied: $\text{Offset} = -0.1$.
    - Baseline $< 04.00$ or $\text{boa\_offset\_applied} = \text{True}$: $\text{Offset} = 0.0$.
- **Strict NoData Protection**: In Sentinel-2 L2A rasters, $\text{DN} = 0$ represents NoData / off-swath. Applying offset $-0.1$ blindly would shift $\text{DN} = 0 \to -0.1$, creating false dark water pixels. The engine masks $\text{DN} \le 0$ to NoData ($-9999.0$) **before** scaling.

### 4.2. Landsat-8 & Landsat-9 Surface Reflectance Scaling

According to USGS and STAC `raster:bands` specifications for Landsat Collection 2 Level-2 Surface Reflectance:

$$\text{Reflectance}_{\text{Landsat}} = \text{DN} \times 0.0000275 - 0.2$$

- **Scale Factor**: $0.0000275$.
- **Offset**: $-0.2$.
- **Source Fill Value**: $0$ (strictly masked to $-9999.0$ before scaling; never converted into valid reflectance $-0.2$).
- **Output Data Type**: `Float32`.
- **Double-Scaling Prevention**: If the input raster is already in Float32 reflectance (e.g. $95^{\text{th}}\text{ percentile} \le 2.0$), the scaling is recognized and not applied twice.
- **Physical Range Integrity**: Small negative reflectance values (e.g. $-0.02$ to $0.0$) resulting from atmospheric over-correction over clear, deep coastal waters are preserved. Blind clipping to $[0, 1]$ is strictly avoided.
- **Sensor Separation**: The Landsat scaling formula is strictly isolated to Landsat-8 and Landsat-9 and is never applied to Sentinel-2A.

---

## 5. Spectral Bands & Sensor Mappings

### 5.1. Sentinel-2A Spectral Bands (12 Bands)

| Band Index | Band ID | Description | Native Resolution | Central Wavelength |
|:---:|:---:|:---|:---:|:---:|
| 1 | **B01** | Coastal Aerosol | 60 m | 443 nm |
| 2 | **B02** | Blue | 10 m | 490 nm |
| 3 | **B03** | Green | 10 m | 560 nm |
| 4 | **B04** | Red | 10 m | 665 nm |
| 5 | **B05** | Red Edge 1 | 20 m | 705 nm |
| 6 | **B06** | Red Edge 2 | 20 m | 740 nm |
| 7 | **B07** | Red Edge 3 | 20 m | 783 nm |
| 8 | **B08** | NIR (Broad) | 10 m | 842 nm |
| 9 | **B8A** | Narrow NIR | 20 m | 865 nm |
| 10 | **B09** | Water Vapour | 60 m | 945 nm |
| 11 | **B11** | SWIR 1 | 20 m | 1610 nm |
| 12 | **B12** | SWIR 2 | 20 m | 2190 nm |

### 5.2. Landsat-8 & Landsat-9 Optical Reflectance Bands (7 Bands)

| Band Index | Band ID | Description | Native Resolution | Central Wavelength | STAC Asset Keys |
|:---:|:---:|:---|:---:|:---:|:---|
| 1 | **SR_B1** | Coastal Aerosol | 30 m | 443 nm | `coastal`, `SR_B1`, `B1` |
| 2 | **SR_B2** | Blue | 30 m | 482 nm | `blue`, `SR_B2`, `B2` |
| 3 | **SR_B3** | Green | 30 m | 562 nm | `green`, `SR_B3`, `B3` |
| 4 | **SR_B4** | Red | 30 m | 655 nm | `red`, `SR_B4`, `B4` |
| 5 | **SR_B5** | NIR | 30 m | 865 nm | `nir08`, `SR_B5`, `B5` |
| 6 | **SR_B6** | SWIR 1 | 30 m | 1609 nm | `swir16`, `SR_B6`, `B6` |
| 7 | **SR_B7** | SWIR 2 | 30 m | 2201 nm | `swir22`, `SR_B7`, `B7` |

Every exported GeoTIFF embeds explicit per-band descriptions (e.g. `SR_B1 - Coastal Aerosol (30m)`) and metadata tags in its header.

---

## 6. Quality Control & Bitmask Decoding

Quality control is executed prior to compositing and masking:

### 6.1. Sentinel-2A Quality Control
- Uses the **Scene Classification Layer (SCL)** (20m, resampled nearest-neighbor).
- Invalid classes: 0 (NoData), 1 (Defective), 3 (Cloud Shadows), 8 (Cloud Medium Prob), 9 (Cloud High Prob), 10 (Thin Cirrus), 11 (Snow/Ice).
- Valid clear classes: 4 (Vegetation), 5 (Bare Soil), 6 (Water).

### 6.2. Landsat-8/9 Quality Control
Read directly as unscaled 16-bit integer bitmasks (`uint16`):
- **`QA_PIXEL`**:
  - Bit 0: Fill ($1$)
  - Bit 1: Dilated Cloud ($2$)
  - Bit 2: Cirrus ($4$)
  - Bit 3: Cloud ($8$)
  - Bit 4: Cloud Shadow ($16$)
  - Bit 5: Snow ($32$)
  - Bit 6: Clear ($64$)
  - Bit 7: Water ($128$)
  - Bits 8–9: Cloud Confidence
  - Bits 10–11: Cloud Shadow Confidence
  - Bits 12–13: Snow/Ice Confidence
  - Bits 14–15: Cirrus Confidence
  - **Water Preservation**: Water pixels (Bit 7 = 1) are strictly recognized as valid aquatic targets and are **never** rejected as invalid.
- **`QA_RADSAT`**: Detects optical radiometric saturation across individual bands (bits 0–6).
- **`SR_QA_AEROSOL`**: Decodes aerosol retrieval quality flags.

---

## 7. Water Masking & Water-Only Export

The engine provides an automated water segmentation workflow adapted per sensor:

### 7.1. Spectral Band Mapping for Water Indices

| Index | Formula | Sentinel-2A Bands | Landsat-8/9 Bands |
|:---|:---:|:---:|:---:|
| **NDWI** (McFeeters) | $\frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR}}$ | Green: **B03**, NIR: **B08** | Green: **SR_B3**, NIR: **SR_B5** |
| **MNDWI** (Xu) | $\frac{\text{Green} - \text{SWIR1}}{\text{Green} + \text{SWIR1}}$ | Green: **B03**, SWIR1: **B11** | Green: **SR_B3**, SWIR1: **SR_B6** |

### 7.2. Segmentation Methods
1. **QA Classification Layer**:
   - Sentinel-2A: SCL class 6 (Water).
   - Landsat: `QA_PIXEL` bit 7 (Water).
2. **Spectral Index Threshold**:
   - Fixed threshold (default: $0.0$).
   - Automated **Otsu bimodal histogram thresholding** computed dynamically over the AOI.
3. **Smart Hybrid Masking [Recommended for SDB]**:
   - **Permissive Mode**: Combines QA water flags with spectral indices ($\text{NDWI} > -0.05$ or $\text{MNDWI} > -0.05$) to preserve shallow coastal sandbars, coral reefs, and turbid coastal waters that pure QA flags may omit.
   - **Strict Mode**: Requires mutual consensus between QA water classification and spectral thresholding.

### 7.3. Water-Only Export
When **Water-Only Output** is checked, all terrestrial and cloud pixels are assigned NoData ($-9999.0$), leaving only clean water pixels for downstream Satellite-Derived Bathymetry (SDB) inversion.

---

## 8. Single Scene & Temporal Median Composite Workflows

### 8.1. Single Scene Processing
1. Filter STAC search results by platform (`sentinel-2a`, `landsat-8`, `landsat-9`), date range, and maximum cloud cover.
2. Build unified target spatial grid (auto-projected UTM).
3. Read raw COG windows directly via HTTP range requests.
4. Normalize raw DN to Float32 BOA surface reflectance.
5. Decode QA bitmasks and generate water mask.
6. Clip strictly to the true AOI polygon.
7. Save Float32 multi-band GeoTIFF, water mask, and provenance metadata JSON.

### 8.2. Temporal Median Composite
1. Load multiple scenes acquired over the target time period.
2. Filter each scene independently using its QA mask.
3. Align all scenes to the common spatial grid and pixel resolution.
4. Compute temporal pixel-wise median:
   $$\text{Median\_Reflectance}(x, y) = \text{nanmedian}\left(\{\text{Reflectance}_k(x, y)\}_{k=1}^N\right)$$
5. **Sensor Combination Rules**:
   - Sentinel-2A scenes: Composable with Sentinel-2A only.
   - Landsat-8 scenes: Composable with Landsat-8.
   - Landsat-9 scenes: Composable with Landsat-9.
   - **Landsat-8 & Landsat-9 Joint Composite**: Fully supported and composable together without cross-sensor discrepancies since OLI and OLI-2 share identical optical spectral specifications.
   - **Sentinel-2A + Landsat Joint Composite**: Strictly prohibited. Cross-sensor radiometry harmonization between MSI and OLI is outside the scope of this tool.
6. Export Composite Products:
   - Primary multi-band median surface reflectance Float32 GeoTIFF.
   - `_ValidCount.tif`: Number of clear observations per pixel (Int16).
   - `_TemporalStd.tif`: Standard deviation across time (Float32).
   - `_WaterOccurrence.tif`: Percentage of clear acquisitions where the pixel was classified as water (Float32).
   - `_WaterMask.tif`: Consensus water mask (UInt8).
   - `_Processing_Metadata.json`: Audit log of all ingested scenes.

---

## 9. Output Settings: Directory-Only Management (Requirement 16)

The output interface is streamlined to eliminate redundant file-path pickers:

1. **Single Parameter: Output Directory (`OUTPUT_DIR`)**:
   - The user selects an existing folder where all outputs will be created.
   - The former `Primary Reflectance GeoTIFF (Optional explicit file path)` parameter is completely removed from the UI.
2. **Automated Filename Generation**:
   Output GeoTIFFs, masks, and metadata files are automatically generated using standardized naming patterns:
   - **Sentinel-2A Single Scene**: `S2A_{YYYYMMDD}_{SCENE_ID}_{MODE}.tif`
   - **Landsat-8 Single Scene**: `L8_{YYYYMMDD}_{SCENE_ID}_{MODE}.tif`
   - **Landsat-9 Single Scene**: `L9_{YYYYMMDD}_{SCENE_ID}_{MODE}.tif`
   - **Sentinel-2A Composite**: `S2A_Composite_{START}_{END}_{MODE}.tif`
   - **Landsat-8 Composite**: `L8_Composite_{START}_{END}_{MODE}.tif`
   - **Landsat-9 Composite**: `L9_Composite_{START}_{END}_{MODE}.tif`
   - **Landsat-8 & 9 Combined Composite**: `Landsat_Composite_{START}_{END}_{MODE}.tif`
3. **Automatic QGIS Loading**:
   - When the user selects **Open output file after running algorithm**, the generated primary Float32 reflectance raster is automatically added to the QGIS map canvas upon completion.

---

## 10. Spatial Resolution Specifications

- **Sentinel-2A**: Default native resolution is $10.0\text{ m}$ (with optional user override).
- **Landsat-8**: Default native resolution is $30.0\text{ m}$.
- **Landsat-9**: Default native resolution is $30.0\text{ m}$.
- **Resampling Methods**:
  - Continuous reflectance bands: Bilinear resampling.
  - Categorical QA layers (SCL, QA_PIXEL, Water Mask): Nearest Neighbor resampling.

---

## 11. Credential-Free Public HTTPS Data Access (Requirement 11)

In strict accordance with Requirement 11 (*"Prefer STAC HTTPS assets that can be accessed without credentials. Use ROI-based COG window reads where supported. Do not require AWS credentials by default"*):

- **Sentinel-2A Data Access**: COG assets reside on AWS Open Data (`https://sentinel-c1-l2a.s3.amazonaws.com/...`) and are freely readable via HTTPS without credentials using GDAL range requests (`/vsicurl/`).
- **Landsat-8 & Landsat-9 Data Access**:
  - While Earth Search STAC indexes Landsat scenes with `s3://usgs-landsat/` URIs (which require an AWS account when accessed via S3 due to AWS Requester-Pays bucket policies), the engine features an automated **Public HTTPS Translator** (`resolve_landsat_public_https_url`).
  - This maps `s3://usgs-landsat/collection02/` to the free, open **Microsoft Planetary Computer Landsat Collection 2 Level-2** mirror (`https://landsateuwest.blob.core.windows.net/landsat-c2/`).
  - It dynamically acquires and caches an open public Shared Access Signature (SAS) token from `https://planetarycomputer.microsoft.com/api/sas/v1/token/landsat-c2-l2`.
  - **Result**: Users can stream, clip to ROI, and normalize Landsat-8 and Landsat-9 Surface Reflectance COGs **completely free and without any AWS credentials, API keys, or cloud accounts**.
  - **Fallback**: If AWS credentials exist in the user's environment, direct access to `s3://usgs-landsat/` via `/vsis3/` is still retained.

---

## 12. Quality Assurance & Regression Testing

The implementation is verified by an automated test suite (`tests/test_sentinel2_stac.py`) comprising 20 unit and synthetic integration tests:

| Test ID | Category | Description | Status |
|:---|:---|:---|:---:|
| `test_processing_baseline_parsing` | Sentinel-2A | Baseline string parsing into numeric values | **PASSED** |
| `test_reflectance_transform_resolution_s2a` | Sentinel-2A | Scale and offset resolution for baselines $\ge 04.00$ vs $< 04.00$ | **PASSED** |
| `test_s2a_nodata_preservation_prevent_negative_zero` | Sentinel-2A | Strict $\text{DN}=0$ NoData ($-9999.0$) preservation | **PASSED** |
| `test_strict_exclusion_of_sentinel_2b_and_2c` | Security/Exclusion | Strict rejection of Sentinel-2B and 2C from query, single scene, and composite | **PASSED** |
| `test_landsat_surface_reflectance_scaling_formula` | Landsat Scaling | $\text{DN} \times 0.0000275 - 0.2$, Fill=0 masked, Float32 | **PASSED** |
| `test_landsat_prevent_double_scaling` | Landsat Scaling | Double-scaling detection and prevention on Float32 inputs | **PASSED** |
| `test_landsat_formula_not_applied_to_sentinel2a` | Sensor Separation | Strict isolation of Landsat scaling from Sentinel-2A | **PASSED** |
| `test_landsat_band_mappings` | Landsat Bands | 7 optical bands and descriptions verified | **PASSED** |
| `test_landsat_asset_key_resolution` | Landsat STAC | Resolution of Earth Search STAC asset keys and aliases | **PASSED** |
| `test_landsat_public_https_url_resolution` | Data Access | Translation of `s3://` Requester-Pays to public HTTPS COG URLs | **PASSED** |
| `test_landsat_qa_pixel_decoding` | Landsat QA | 16-bit `QA_PIXEL` bitmask decoding and water pixel preservation | **PASSED** |
| `test_landsat_qa_radsat_decoding` | Landsat QA | `QA_RADSAT` optical band saturation detection | **PASSED** |
| `test_landsat_water_masking` | Water Masking | NDWI and Smart Hybrid masking using Landsat Green, NIR, and SWIR1 | **PASSED** |
| `test_landsat_geotiff_export_and_band_descriptions` | GeoTIFF Export | Float32 multi-band GeoTIFF export with band descriptions | **PASSED** |
| `test_temporal_composite_landsat8_and_landsat9_combined` | Compositing | Rejection of mixed Sentinel-2A + Landsat stacks | **PASSED** |
| `test_process_single_scene_sentinel_2a_synthetic` | Synthetic E2E | Sentinel-2A Single Scene full pipeline execution | **PASSED** |
| `test_process_single_scene_landsat_8_synthetic` | Synthetic E2E | Landsat-8 Single Scene full pipeline execution | **PASSED** |
| `test_process_single_scene_landsat_9_synthetic` | Synthetic E2E | Landsat-9 Single Scene full pipeline execution | **PASSED** |
| `test_process_temporal_median_composite_landsat_8_and_9_combined_synthetic` | Synthetic E2E | Joint Landsat-8 + Landsat-9 temporal median composite execution | **PASSED** |
| `test_directory_only_destination_workflow` | Output Settings | Verification of Requirement 16 Directory-Only destination | **PASSED** |

---

## 13. Data Attribution & Citations

- **Copernicus Sentinel Data**: Contains modified Copernicus Sentinel data [2015–2026], processed by Element 84 and ESA.
- **USGS Landsat Data**: Landsat Collection 2 Level-2 Surface Reflectance courtesy of the U.S. Geological Survey and NASA.
- **Element 84 Earth Search**: STAC API and catalog indexing provided by Element 84 on the AWS Open Data Sponsorship Program.
- **Bathymetrix-AI**: Developed by Mohamed Aly Nasef.


### Source and method references (v8.0)

Use actual acquisition years in Copernicus attribution rather than the fixed example year range above. Source hrefs must be read from current STAC items; the historical sample hostname in Section 11 is not an API contract.

- Copernicus / ESA, Collection 1 Level-2A: 12 reflectance bands; B10 is excluded. https://sentinels.copernicus.eu/sentinel-data-access/sentinel-products/sentinel-2-data-products/collection-1-level-2a
- USGS, Landsat Collection 2 Surface Reflectance: https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-reflectance . Product DOI: https://doi.org/10.5066/P9OGBGM6
- USGS, QA_PIXEL and QA_RADSAT definitions: https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands
- Element 84, Earth Search catalog and collection documentation: https://github.com/Element84/earth-search ; https://earth-search.aws.element84.com/v1 . Read per-asset raster:bands scale/offset/nodata metadata; avoid applying radiometric offsets twice.
- Microsoft Planetary Computer SAS access: https://planetarycomputer.microsoft.com/docs/concepts/sas/ . Public mirror availability and signing are external service dependencies.
- McFeeters (1996), NDWI: https://doi.org/10.1080/01431169608948714
- Xu (2006), MNDWI: https://doi.org/10.1080/01431160600589179
- Otsu (1979), histogram thresholding: https://doi.org/10.1109/TSMC.1979.4310076
- GDAL utilities: https://gdal.org/en/stable/api/python/utilities.html ; NumPy nanmedian: https://numpy.org/doc/stable/reference/generated/numpy.nanmedian.html

The hybrid coastal water decision is an implementation heuristic, not a separately validated published classifier. Citation of NDWI/MNDWI/Otsu does not establish local classification accuracy. No bathymetric validation or new scientific result is claimed by this documentation update.
