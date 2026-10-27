# -*- coding: utf-8 -*-
"""
qgis_interface/algorithms/Sentinel2_Downloader.py
QGIS Processing Algorithm for Satellite Imagery Downloader (Earth Search STAC).
Supports exactly 3 satellites:
1. Sentinel-2A (MSI Level-2A BOA Surface Reflectance)
2. Landsat-8 (OLI Collection 2 Level-2 Surface Reflectance)
3. Landsat-9 (OLI-2 Collection 2 Level-2 Surface Reflectance)

Sentinel-2B and Sentinel-2C are strictly excluded and rejected.
Belongs to Group: 2. Data Acquisition & Datum
"""

import os
import json
import tempfile
from datetime import datetime

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterExtent,
    QgsProcessingParameterEnum,
    QgsProcessingParameterString,
    QgsProcessingParameterNumber,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFolderDestination,
    QgsProcessingOutputRasterLayer,
    QgsProcessingOutputFolder,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsRasterLayer,
    QgsProject,
    QgsProcessingException
)

try:
    from ...core.stac.stac_client import (
        STACClient,
        STACSearchError,
        ORDERED_BAND_LIST,
        LANDSAT_ORDERED_BAND_LIST,
        ALLOWED_PLATFORMS,
        DISALLOWED_PLATFORMS
    )
    from ...core.stac.processor import (
        process_single_scene,
        process_temporal_median_composite,
        ProcessorError
    )
except (ImportError, ValueError):
    try:
        from Bathymetrix_AI.core.stac.stac_client import (
            STACClient,
            STACSearchError,
            ORDERED_BAND_LIST,
            LANDSAT_ORDERED_BAND_LIST,
            ALLOWED_PLATFORMS,
            DISALLOWED_PLATFORMS
        )
        from Bathymetrix_AI.core.stac.processor import (
            process_single_scene,
            process_temporal_median_composite,
            ProcessorError
        )
    except (ImportError, ValueError):
        from core.stac.stac_client import (
            STACClient,
            STACSearchError,
            ORDERED_BAND_LIST,
            LANDSAT_ORDERED_BAND_LIST,
            ALLOWED_PLATFORMS,
            DISALLOWED_PLATFORMS
        )
        from core.stac.processor import (
            process_single_scene,
            process_temporal_median_composite,
            ProcessorError
        )


class Sentinel2DownloaderAlgorithm(QgsProcessingAlgorithm):
    # Parameter Keys
    AOI_POLYGON = "AOI_POLYGON"
    AOI_EXTENT = "AOI_EXTENT"
    MODE = "MODE"
    START_DATE = "START_DATE"
    END_DATE = "END_DATE"
    MAX_CLOUD = "MAX_CLOUD"
    PLATFORM = "PLATFORM"
    MIN_COVERAGE = "MIN_COVERAGE"
    MAX_SCENES = "MAX_SCENES"
    TARGET_RES = "TARGET_RES"
    WATER_ONLY = "WATER_ONLY"
    WATER_METHOD = "WATER_METHOD"
    WATER_MODE = "WATER_MODE"
    WATER_OCCURRENCE_THRESHOLD = "WATER_OCCURRENCE_THRESHOLD"
    SAVE_QUALITY_RASTERS = "SAVE_QUALITY_RASTERS"
    OUTPUT_DIR = "OUTPUT_DIR"
    LOAD_LAYERS = "LOAD_LAYERS"
    OUTPUT_RASTER = "OUTPUT_RASTER"

    def tr(self, string):
        return QCoreApplication.translate("Sentinel2DownloaderAlgorithm", string)

    def createInstance(self):
        return Sentinel2DownloaderAlgorithm()

    def name(self):
        # Keep internal name for backwards compatibility with existing models and scripts
        return "sentinel2_imagery_downloader"

    def displayName(self):
        return self.tr("2.0 Satellite Imagery Downloader (Sentinel-2A & Landsat-8/9)")

    def group(self):
        return self.tr("2. Data Acquisition & Datum")

    def groupId(self):
        return "data_prep"

    def shortHelpString(self):
        return """
        <div style="font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.5; color: #2C3E50;">
            <h2 style="margin-bottom: 5px; color: #1B4F72;">🛰️ Satellite Surface Reflectance Downloader & Compositor</h2>
            <p style="margin-top: 0; margin-bottom: 15px; font-size: 13px;">
                Direct access to <b>Element 84 Earth Search STAC API</b> for <b>Sentinel-2A</b>, <b>Landsat-8</b>, and <b>Landsat-9</b> surface reflectance imagery.
                Download and composite calibrated BOA reflectance directly inside QGIS with <b>zero credentials or cloud accounts</b> required.
            </p>

            <h3 style="color: #0E6251; margin-bottom: 5px; border-bottom: 2px solid #0E6251; padding-bottom: 3px;">🛰️ Supported Satellites (Sentinel-2B/2C Excluded)</h3>
            <ul style="font-size: 12px; margin-top: 5px; padding-left: 20px;">
                <li><b>Sentinel-2A:</b> MSI Level-2A BOA Surface Reflectance (10m/20m/60m, 12 bands B01-B12).</li>
                <li><b>Landsat-8:</b> OLI Collection 2 Level-2 Surface Reflectance (30m, 7 bands SR_B1 to SR_B7).</li>
                <li><b>Landsat-9:</b> OLI-2 Collection 2 Level-2 Surface Reflectance (30m, 7 bands SR_B1 to SR_B7).</li>
                <li><b>Landsat-8 & Landsat-9 (Combined):</b> Supports joint multi-sensor median compositing.</li>
            </ul>

            <h3 style="color: #0E6251; margin-bottom: 5px; border-bottom: 2px solid #0E6251; padding-bottom: 3px;">✨ Operational Modes</h3>
            <ul style="font-size: 12px; margin-top: 5px; padding-left: 20px;">
                <li><b>Single Scene:</b> Downloads and normalizes the best single cloud-free scene for your AOI.</li>
                <li><b>Temporal Median Composite:</b> Calculates pixel-wise <code>nanmedian</code> surface reflectance across all available scenes within your date window.</li>
                <li><b>Water-Only Masking:</b> Excludes land pixels to NoData (-9999.0) using SCL, QA_PIXEL, NDWI/MNDWI, or smart hybrid classification while preserving georeferencing.</li>
            </ul>

            <h3 style="color: #7D6608; margin-top: 15px; margin-bottom: 5px; border-bottom: 2px solid #7D6608; padding-bottom: 3px;">🔬 Spectral Precision & Radiometry</h3>
            <ul style="font-size: 12px; margin-top: 5px; padding-left: 20px;">
                <li><b>Sentinel-2A:</b> Normalized Float32 BOA Reflectance with Processing Baseline >= 04.00 radiometry handling.</li>
                <li><b>Landsat-8/9:</b> USGS Collection 2 Level-2 scaling: <code>Reflectance = DN * 0.0000275 - 0.2</code> applied exactly once; Fill=0 masked strictly to NoData (-9999.0).</li>
                <li><b>QA Decoding:</b> Full bitmask evaluation of clouds, dilated clouds, cirrus, shadows, snow, and radiometric saturation.</li>
            </ul>

            <p style="margin-top: 15px; background: #E8F8F5; border-left: 4px solid #1ABC9C; padding: 8px 12px; font-size: 12px; color: #0E6251;">
                <b>🚀 Directory-Only Output:</b> All generated files (GeoTIFFs, masks, metadata, logs) are saved inside the selected Output Directory with automatic, structured filenames.
            </p>
            <br><b>Author / Developer:</b> Mohamed Aly Nasef
        </div>
        """

    def helpString(self):
        return self.shortHelpString()

    def initAlgorithm(self, config=None):
        # 1. Spatial AOI
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.AOI_POLYGON,
                self.tr("Area of Interest Polygon (Vector Layer)"),
                [QgsProcessing.TypeVectorPolygon],
                optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterExtent(
                self.AOI_EXTENT,
                self.tr("Area of Interest Extent (Alternative if Polygon layer not used)"),
                optional=True
            )
        )

        # 2. Operational Mode
        modes = [
            "1. Single Scene (Best Available Scene)",
            "2. Temporal Median Composite (Multi-Scene Stack)"
        ]
        self.addParameter(
            QgsProcessingParameterEnum(
                self.MODE,
                self.tr("Operational Mode"),
                options=modes,
                defaultValue=0
            )
        )

        # 3. Temporal Filters
        self.addParameter(
            QgsProcessingParameterString(
                self.START_DATE,
                self.tr("Start Date (YYYY-MM-DD)"),
                defaultValue="2023-06-01"
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.END_DATE,
                self.tr("End Date (YYYY-MM-DD)"),
                defaultValue="2023-06-30"
            )
        )

        # 4. Search Criteria
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_CLOUD,
                self.tr("Maximum Scene Cloud Cover (%)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=10.0
            )
        )
        
        # Satellite Selection: Exactly Sentinel-2A, Landsat-8, Landsat-9 (Sentinel-2B and 2C removed)
        platforms = [
            "Sentinel-2A",
            "Landsat-8",
            "Landsat-9",
            "Landsat-8 & Landsat-9 (Combined)"
        ]
        self.addParameter(
            QgsProcessingParameterEnum(
                self.PLATFORM,
                self.tr("Satellite Platform Filter"),
                options=platforms,
                defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_COVERAGE,
                self.tr("Minimum AOI Spatial Coverage (%)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=50.0
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_SCENES,
                self.tr("Maximum Number of Scenes to Query"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=20
            )
        )

        # 5. Spatial & Spectral Output Resolution
        resolutions = [
            "Optimal Sensor Resolution (10m for S2A, 30m for Landsat)",
            "10 meters",
            "20 meters",
            "30 meters"
        ]
        self.addParameter(
            QgsProcessingParameterEnum(
                self.TARGET_RES,
                self.tr("Target Spatial Resolution"),
                options=resolutions,
                defaultValue=0
            )
        )

        # 6. Water Masking Configuration
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.WATER_ONLY,
                self.tr("Water-Only Output (Mask Land Pixels to NoData)"),
                defaultValue=False
            )
        )
        water_methods = [
            "Smart Hybrid (SCL/QA + NDWI/MNDWI + Cloud Mask)",
            "SCL / QA_PIXEL Water Classification Only",
            "NDWI Spectral Mask (Green - NIR)",
            "MNDWI Spectral Mask (Green - SWIR1)"
        ]
        self.addParameter(
            QgsProcessingParameterEnum(
                self.WATER_METHOD,
                self.tr("Water Masking Method"),
                options=water_methods,
                defaultValue=0
            )
        )
        water_modes = [
            "Permissive (Coastal & Turbid Reef Friendly)",
            "Strict (High-Consensus Only)"
        ]
        self.addParameter(
            QgsProcessingParameterEnum(
                self.WATER_MODE,
                self.tr("Water Mask Strictness / Sensitivity"),
                options=water_modes,
                defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.WATER_OCCURRENCE_THRESHOLD,
                self.tr("Temporal Water Occurrence Threshold (%) [Composite Only]"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=50.0
            )
        )

        # 7. Quality Rasters & Output Destination (Section 16: Directory Only)
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.SAVE_QUALITY_RASTERS,
                self.tr("Export Ancillary Quality Rasters (Valid Count, Temp Std, Water Occurrence)"),
                defaultValue=True
            )
        )
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_DIR,
                self.tr("Output Directory")
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.LOAD_LAYERS,
                self.tr("Automatically Load Output Rasters into QGIS Map Canvas"),
                defaultValue=True
            )
        )

        # QGIS Processing Output declarations (for modeler and automatic loading without UI file path selection)
        self.addOutput(
            QgsProcessingOutputRasterLayer(
                self.OUTPUT_RASTER,
                self.tr("Primary Reflectance GeoTIFF")
            )
        )
        self.addOutput(
            QgsProcessingOutputFolder(
                self.OUTPUT_DIR,
                self.tr("Output Directory")
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        # Setup logging
        log_lines = []
        original_pushInfo = feedback.pushInfo

        def log(msg):
            log_lines.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
            original_pushInfo(msg)

        feedback.pushInfo = log

        log("Initializing Satellite Earth Search STAC Downloader (Sentinel-2A / Landsat-8 / Landsat-9)...")

        # 1. Resolve Output Directory (Requirement 16)
        out_dir = self.parameterAsString(parameters, self.OUTPUT_DIR, context)
        if not out_dir or out_dir == "TEMPORARY_OUTPUT":
            out_dir = os.path.join(tempfile.gettempdir(), "bathymetrix_satellite_downloads")
        os.makedirs(out_dir, exist_ok=True)
        log(f"Output directory resolved: {out_dir}")

        # 2. Resolve AOI Geometry in EPSG:4326
        aoi_source = self.parameterAsSource(parameters, self.AOI_POLYGON, context)
        aoi_extent = self.parameterAsExtent(parameters, self.AOI_EXTENT, context)

        wgs84_crs = QgsCoordinateReferenceSystem("EPSG:4326")
        aoi_geojson = None

        if aoi_source:
            log(f"Extracting AOI from Polygon layer: {aoi_source.sourceName()} ({aoi_source.featureCount()} features)")
            transform_to_wgs84 = QgsCoordinateTransform(aoi_source.sourceCrs(), wgs84_crs, context.project())
            
            geoms = []
            for feat in aoi_source.getFeatures():
                geom = feat.geometry()
                if geom and not geom.isEmpty():
                    geom_wgs = QgsGeometry(geom)
                    geom_wgs.transform(transform_to_wgs84)
                    geoms.append(geom_wgs)

            if not geoms:
                raise QgsProcessingException("The provided vector polygon layer contains no valid features.")

            if len(geoms) == 1:
                combined_geom = geoms[0]
            else:
                combined_geom = QgsGeometry.unaryUnion(geoms)

            aoi_geojson = json.loads(combined_geom.asJson())
            log(f"Combined polygon AOI extracted successfully (WGS84 centroid: {combined_geom.centroid().asPoint().x():.4f}, {combined_geom.centroid().asPoint().y():.4f})")

        elif aoi_extent and not aoi_extent.isNull():
            extent_crs = self.parameterAsExtentCrs(parameters, self.AOI_EXTENT, context)
            log(f"Extracting AOI from Extent in CRS: {extent_crs.authid()}")
            transform_to_wgs84 = QgsCoordinateTransform(extent_crs, wgs84_crs, context.project())
            extent_wgs = transform_to_wgs84.transformBoundingBox(aoi_extent)
            
            aoi_geojson = {
                "type": "Polygon",
                "coordinates": [[
                    [extent_wgs.xMinimum(), extent_wgs.yMinimum()],
                    [extent_wgs.xMaximum(), extent_wgs.yMinimum()],
                    [extent_wgs.xMaximum(), extent_wgs.yMaximum()],
                    [extent_wgs.xMinimum(), extent_wgs.yMaximum()],
                    [extent_wgs.xMinimum(), extent_wgs.yMinimum()]
                ]]
            }
            log(f"Bounding box AOI extracted: [{extent_wgs.xMinimum():.4f}, {extent_wgs.yMinimum():.4f}, {extent_wgs.xMaximum():.4f}, {extent_wgs.yMaximum():.4f}]")
        else:
            raise QgsProcessingException("Please provide either a Polygon Vector Layer or an Extent for AOI definition.")

        # 3. Read search parameters
        mode_idx = self.parameterAsEnum(parameters, self.MODE, context)
        start_date = self.parameterAsString(parameters, self.START_DATE, context).strip()
        end_date = self.parameterAsString(parameters, self.END_DATE, context).strip()
        max_cloud = self.parameterAsDouble(parameters, self.MAX_CLOUD, context)
        plat_idx = self.parameterAsEnum(parameters, self.PLATFORM, context)
        min_cov = self.parameterAsDouble(parameters, self.MIN_COVERAGE, context)
        max_scenes = self.parameterAsInt(parameters, self.MAX_SCENES, context)

        # Platform resolution with strict 2B/2C rejection and backward compatibility handling
        plat_map = {
            0: "sentinel-2a",
            1: "landsat-8",
            2: "landsat-9",
            3: "landsat"  # Combined Landsat-8 and Landsat-9
        }
        
        # Check if an older configuration provided an old index or string with 2B / 2C
        plat_raw_str = str(parameters.get(self.PLATFORM, "")).lower()
        if "2b" in plat_raw_str or "2c" in plat_raw_str:
            log("Notice: Older configuration requested Sentinel-2B or Sentinel-2C. Both satellites have been removed. Falling back safely to Sentinel-2A.")
            plat_filter = "sentinel-2a"
        else:
            plat_filter = plat_map.get(plat_idx, "sentinel-2a")

        # Target Resolution
        res_idx = self.parameterAsEnum(parameters, self.TARGET_RES, context)
        if res_idx == 0:
            target_res = 30.0 if "landsat" in plat_filter else 10.0
        elif res_idx == 1:
            target_res = 10.0
        elif res_idx == 2:
            target_res = 20.0
        elif res_idx == 3:
            target_res = 30.0
        else:
            target_res = 30.0 if "landsat" in plat_filter else 10.0

        water_only = self.parameterAsBoolean(parameters, self.WATER_ONLY, context)
        method_map = {0: "hybrid", 1: "scl", 2: "ndwi", 3: "mndwi"}
        water_method = method_map.get(self.parameterAsEnum(parameters, self.WATER_METHOD, context), "hybrid")
        water_mode = "permissive" if self.parameterAsEnum(parameters, self.WATER_MODE, context) == 0 else "strict"
        water_thresh = self.parameterAsDouble(parameters, self.WATER_OCCURRENCE_THRESHOLD, context)
        save_ancillary = self.parameterAsBoolean(parameters, self.SAVE_QUALITY_RASTERS, context)
        load_layers = self.parameterAsBoolean(parameters, self.LOAD_LAYERS, context)

        # 4. Execute STAC Search
        stac = STACClient()
        log(f"Executing STAC Search (Platform: {plat_filter}, Date: {start_date} to {end_date}, Max Cloud: {max_cloud}%)...")
        try:
            scenes = stac.search_scenes(
                aoi_geometry=aoi_geojson,
                start_date=start_date,
                end_date=end_date,
                max_cloud_cover=max_cloud,
                platform=plat_filter,
                min_aoi_coverage=min_cov,
                max_scenes=max_scenes,
                feedback=feedback
            )
        except STACSearchError as se:
            raise QgsProcessingException(f"STAC search failed: {se}")

        # Strict validation: verify every returned scene
        valid_scenes = []
        for s in scenes:
            p = str(s.get("platform", "")).lower()
            if p in DISALLOWED_PLATFORMS or "s2b" in s["id"].lower() or "s2c" in s["id"].lower():
                log(f"Warning: Discarded invalid scene {s['id']} (Sentinel-2B/2C not permitted).")
                continue
            if p not in ALLOWED_PLATFORMS:
                log(f"Warning: Discarded scene {s['id']} with unsupported platform '{p}'.")
                continue
            valid_scenes.append(s)

        if not valid_scenes:
            raise QgsProcessingException(
                f"No suitable scenes found for criteria (Satellite: {plat_filter}, Date: {start_date} to {end_date}, Max Cloud: {max_cloud}%). "
                "Try expanding the date range, increasing maximum cloud cover, or lowering the minimum coverage percentage."
            )

        log(f"STAC Search returned {len(valid_scenes)} validated scenes.")

        # 5. Process according to mode
        results = {}
        try:
            if mode_idx == 0:
                # Mode: Single Scene (Best scene by coverage and cloud cover)
                best_scene = sorted(valid_scenes, key=lambda s: (-s["aoi_coverage_pct"], s["cloud_cover"]))[0]
                log(f"Selected best scene: {best_scene['id']} ({best_scene.get('satellite', 'Satellite')}, Date: {best_scene['date']}, Cloud: {best_scene['cloud_cover']}%, Coverage: {best_scene['aoi_coverage_pct']}%)")

                results = process_single_scene(
                    scene_meta=best_scene,
                    aoi_geojson=aoi_geojson,
                    output_dir=out_dir,
                    pixel_size=target_res,
                    water_only=water_only,
                    water_method=water_method,
                    water_mode=water_mode,
                    save_water_mask_raster=save_ancillary,
                    feedback=feedback
                )
            else:
                # Mode: Temporal Median Composite
                log(f"Executing Temporal Median Composite across {len(valid_scenes)} scenes...")
                results = process_temporal_median_composite(
                    scenes_meta=valid_scenes,
                    aoi_geojson=aoi_geojson,
                    output_dir=out_dir,
                    pixel_size=target_res,
                    water_only=water_only,
                    water_temporal_policy="occurrence_threshold",
                    water_occurrence_threshold=water_thresh,
                    save_valid_count_raster=save_ancillary,
                    save_temp_std_raster=save_ancillary,
                    save_water_occurrence_raster=save_ancillary,
                    feedback=feedback
                )
        except ProcessorError as pe:
            raise QgsProcessingException(f"Processing error: {pe}")
        except Exception as e:
            raise QgsProcessingException(f"Unexpected error during processing: {e}")

        # 6. Save Process Execution Log inside Output Directory
        log_file_path = os.path.join(out_dir, f"Satellite_Processing_Log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        try:
            with open(log_file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(log_lines))
            log(f"Process execution log written to: {log_file_path}")
        except Exception as err:
            log(f"Warning: Could not save process log: {err}")

        # 7. Add primary output to QGIS Canvas if requested
        primary_raster = results.get("surface_reflectance") or results.get("composite_reflectance")
        if primary_raster and load_layers and QgsProject.instance():
            layer_name = os.path.splitext(os.path.basename(primary_raster))[0]
            rlayer = QgsRasterLayer(primary_raster, layer_name)
            if rlayer.isValid():
                QgsProject.instance().addMapLayer(rlayer)
                log(f"Added primary raster layer to QGIS canvas: {layer_name}")

        output_dict = {
            self.OUTPUT_DIR: out_dir,
            "PRIMARY_RASTER": primary_raster,
            "RESULTS_MAP": results
        }
        if primary_raster:
            output_dict[self.OUTPUT_RASTER] = primary_raster

        log("Satellite Imagery Downloader processing finished successfully!")
        return output_dict
