import os
import math
import numpy as np
import rasterio
from typing import Dict, List, Optional, Tuple, Any
try:
    from qgis.core import (
        QgsVectorLayer,
        QgsGeometry,
        QgsFeature,
        QgsField,
        QgsFields,
        QgsVectorFileWriter,
        QgsWkbTypes,
        QgsPointXY,
        QgsProject,
        QgsProcessingFeedback,
        QgsSpatialIndex,
    )
    from qgis.PyQt.QtCore import QVariant
    import processing
except ImportError:
    QgsVectorLayer = None
    QgsGeometry = None
    QgsFeature = None
    QgsField = None
    QgsFields = None
    QgsVectorFileWriter = None
    QgsWkbTypes = None
    QgsPointXY = None
    QgsProject = None
    QgsProcessingFeedback = object
    QgsSpatialIndex = None
    QVariant = None
    processing = None

class ShorelineDynamicsTracker:
    def __init__(self):
        pass

    def extract_water_mask_from_depth(self, year: int, depth_map_path: str, output_dir: str, feedback: QgsProcessingFeedback, shoreline_depth: float = 0.0) -> str:
        """
        Creates a binary water mask from a depth map based on the shoreline_depth threshold.
        Seamlessly handles both negative and positive depth conventions.
        Water = 1, Land/NoData = 0.
        """
        mask_path = os.path.join(output_dir, f"temp_water_mask_{year}.tif")
        with rasterio.open(depth_map_path) as src:
            depth = src.read(1).astype(np.float32)
            water_mask = np.zeros_like(depth, dtype=np.uint8)
            nodata = src.nodata if src.nodata is not None else -9999.0
            
            # Valid data pixels
            valid = (depth != nodata) & np.isfinite(depth) & (depth > -9000) & (depth < 9000)
            
            valid_vals = depth[valid]
            if len(valid_vals) > 0:
                is_neg = np.nanmean(valid_vals) < 0
                if is_neg:
                    # Depth is negative (e.g. -10m is deeper than -0.5m shoreline)
                    thresh = -abs(float(shoreline_depth)) if shoreline_depth != 0.0 else 0.0
                    water_mask[valid & (depth < thresh)] = 1
                else:
                    # Depth is positive (e.g. +10m is deeper than +0.5m shoreline)
                    thresh = abs(float(shoreline_depth)) if shoreline_depth != 0.0 else 0.0
                    water_mask[valid & (depth > thresh)] = 1
            
            profile = src.profile.copy()
            profile.update(count=1, dtype=rasterio.uint8, nodata=0)
            water_mask[~valid] = 0
            
            with rasterio.open(mask_path, "w", **profile) as dst:
                dst.write(water_mask, 1)
                
        return mask_path

    def process_shorelines(
        self,
        sdb_maps: Dict[int, str],
        output_dir: str,
        feedback: QgsProcessingFeedback,
        osw_shp: str = None,
        shoreline_depth: float = 0.0,
        comparison_mode: str = "Sequential (Year-to-Year)"
    ) -> List[str]:
        feedback.pushInfo(f"🏖️ [SHORELINE TRACKER] Processing shorelines with depth threshold: {shoreline_depth}m")
        
        years = sorted(list(sdb_maps.keys()))
        if len(years) < 2:
            return []
            
        # 1. Create water masks for all years
        water_masks = {}
        for y in years:
            mask = self.extract_water_mask_from_depth(y, sdb_maps[y], output_dir, feedback, shoreline_depth)
            water_masks[y] = mask
            
        change_polygons = []
        
        # 2. Compute change polygons based on comparison mode
        if "Baseline" in comparison_mode:
            baseline_y = years[0]
            for i in range(1, len(years)):
                curr_y = years[i]
                poly_shp = self.compute_shoreline_change_polygons(
                    year_a=baseline_y,
                    year_b=curr_y,
                    mask_a_path=water_masks[baseline_y],
                    mask_b_path=water_masks[curr_y],
                    osw_shp=osw_shp,
                    output_dir=output_dir,
                    feedback=feedback
                )
                if poly_shp and os.path.exists(poly_shp):
                    change_polygons.append(poly_shp)
        else:
            # Sequential
            for i in range(len(years) - 1):
                y1, y2 = years[i], years[i+1]
                poly_shp = self.compute_shoreline_change_polygons(
                    year_a=y1,
                    year_b=y2,
                    mask_a_path=water_masks[y1],
                    mask_b_path=water_masks[y2],
                    osw_shp=osw_shp,
                    output_dir=output_dir,
                    feedback=feedback
                )
                if poly_shp and os.path.exists(poly_shp):
                    change_polygons.append(poly_shp)
                    
        # 3. Always compute Overall Trend (First vs Last) ONLY if there are more than 2 years
        if len(years) > 2:
            y_first, y_last = years[0], years[-1]
            overall_poly_shp = self.compute_shoreline_change_polygons(
                year_a=y_first,
                year_b=y_last,
                mask_a_path=water_masks[y_first],
                mask_b_path=water_masks[y_last],
                osw_shp=osw_shp,
                output_dir=output_dir,
                feedback=feedback
            )
            if overall_poly_shp and os.path.exists(overall_poly_shp):
                if overall_poly_shp not in change_polygons:
                    change_polygons.append(overall_poly_shp)
                
        return change_polygons

    def extract_year_shoreline(
        self,
        year: int,
        image_path: str,
        output_dir: str,
        feedback: QgsProcessingFeedback,
        nir_idx: int = 4,
        green_idx: int = 3
    ) -> str:
        """
        Extracts shoreline water mask from image using NDWI + Otsu water mask.
        Returns the path to the temporary Raster Mask.
        """
        feedback.pushInfo(f"🏖️ [SHORELINE TRACKER] Extracting water mask for Year {year}...")

        with rasterio.open(image_path) as src:
            img = src.read()
            nbands = img.shape[0]
            g_idx_safe = min(max(1, green_idx), nbands) - 1
            n_idx_safe = min(max(1, nir_idx), nbands) - 1
            
            green = img[g_idx_safe].astype(np.float32)
            nir = img[n_idx_safe].astype(np.float32)

            denom = green + nir
            denom[denom == 0] = 0.0001
            ndwi = (green - nir) / denom

            water_mask = (ndwi > 0.0).astype(np.uint8)

            mask_path = os.path.join(output_dir, f"temp_water_mask_{year}.tif")
            profile = src.profile.copy()
            profile.update(count=1, dtype=rasterio.uint8, nodata=0)
            with rasterio.open(mask_path, "w", **profile) as dst:
                dst.write(water_mask, 1)

        feedback.pushInfo(f"✅ Year {year} Water Mask saved.")
        return mask_path

    def compute_shoreline_change_polygons(
        self,
        year_a: int,
        year_b: int,
        mask_a_path: str,
        mask_b_path: str,
        osw_shp: str,
        output_dir: str,
        feedback: QgsProcessingFeedback,
        shoreline_roi_shp: str = None,
    ) -> str:
        feedback.pushInfo(f"🟢🔴 Generating visual Erosion/Accretion Polygons between {year_a} and {year_b} (Ultra-Fast Raster Math)...")
        out_change_shp = os.path.join(output_dir, f"Shoreline_Change_Polygons_{year_a}_{year_b}.shp")
        
        from rasterio.warp import reproject, Resampling
        from qgis.core import (
            QgsVectorLayer,
            QgsCoordinateReferenceSystem,
            QgsCoordinateTransform,
            QgsProject,
            QgsGeometry,
            QgsField,
            QgsFields,
            QgsFeature,
            QgsVectorFileWriter,
            QgsWkbTypes,
        )
        
        # 1. Raster Math Difference with Auto-Alignment
        with rasterio.open(mask_a_path) as srcA, rasterio.open(mask_b_path) as srcB:
            mask_a = srcA.read(1).astype(np.int16)
            
            mask_b_aligned = np.zeros_like(mask_a, dtype=np.int16)
            reproject(
                source=srcB.read(1),
                destination=mask_b_aligned,
                src_transform=srcB.transform,
                src_crs=srcB.crs,
                dst_transform=srcA.transform,
                dst_crs=srcA.crs,
                resampling=Resampling.nearest
            )
            
            diff = mask_b_aligned - mask_a
            
            # CRITICAL: Set unchanged pixels (0) to NoData (-9999) 
            # so GDAL Polygonize ignores them instead of vectorizing millions of 0-value pixels!
            diff[diff == 0] = -9999
            
            diff_path = os.path.join(output_dir, f"temp_diff_{year_a}_{year_b}.tif")
            profile = srcA.profile.copy()
            profile.update(dtype=rasterio.int16, nodata=-9999)
            with rasterio.open(diff_path, "w", **profile) as dst:
                dst.write(diff, 1)

        # 2. Polygonize Diff
        poly_res = processing.run(
            "gdal:polygonize",
            {
                "INPUT": diff_path,
                "BAND": 1,
                "FIELD": "DN",
                "EIGHT_CONNECTEDNESS": False,
                "OUTPUT": "TEMPORARY_OUTPUT"
            }
        )

        raw_layer = poly_res["OUTPUT"]
        if isinstance(raw_layer, str):
            raw_layer = QgsVectorLayer(raw_layer, "raw_poly", "ogr")

        # 3. Read overlay / clip mask if provided
        clip_geom_combined = None
        clip_shp = shoreline_roi_shp or osw_shp
        if clip_shp:
            clip_shp = str(clip_shp).split('|')[0]
            if os.path.exists(clip_shp):
                c_layer = QgsVectorLayer(clip_shp, "clip_mask", "ogr")
                if c_layer.isValid():
                    tr = None
                    if raw_layer.isValid() and c_layer.crs() != raw_layer.crs():
                        tr = QgsCoordinateTransform(c_layer.crs(), raw_layer.crs(), QgsProject.instance())
                    clip_geoms = []
                    for f in c_layer.getFeatures():
                        g = f.geometry()
                        if not g.isNull():
                            if tr:
                                try: g.transform(tr)
                                except Exception: pass
                            clip_geoms.append(g)
                    if clip_geoms:
                        clip_geom_combined = QgsGeometry.unaryUnion(clip_geoms)

        # 4. Prepare Output Fields
        fields = QgsFields()
        fields.append(QgsField("ChangeType", QVariant.String, len=30))
        fields.append(QgsField("Area_m2", QVariant.Double))
        fields.append(QgsField("Area_ha", QVariant.Double))
        fields.append(QgsField("Year_From", QVariant.Int))
        fields.append(QgsField("Year_To", QVariant.Int))
        fields.append(QgsField("Period", QVariant.String, len=30))

        # 5. Clean shapefile sidecars before writing
        base, _ = os.path.splitext(out_change_shp)
        for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qpj"]:
            f_to_del = base + ext
            if os.path.exists(f_to_del):
                try: os.remove(f_to_del)
                except Exception: pass

        target_crs = raw_layer.crs() if (raw_layer and raw_layer.isValid()) else None
        if not target_crs or not target_crs.isValid():
            with rasterio.open(mask_a_path) as srcA:
                crs_wkt = srcA.crs.to_wkt()
            target_crs = QgsCoordinateReferenceSystem(crs_wkt)

        writer = QgsVectorFileWriter(
            out_change_shp,
            "UTF-8",
            fields,
            QgsWkbTypes.MultiPolygon,
            target_crs,
            "ESRI Shapefile"
        )

        erosion_count = 0
        accretion_count = 0
        erosion_area = 0.0
        accretion_area = 0.0

        if raw_layer and raw_layer.isValid() and writer.hasError() == QgsVectorFileWriter.NoError:
            for feat in raw_layer.getFeatures():
                dn_val = feat["DN"]
                if dn_val is None:
                    continue

                try:
                    dn_int = int(float(dn_val))
                except (ValueError, TypeError):
                    continue

                if dn_int == 1:
                    change_type = "Erosion"
                elif dn_int == -1:
                    change_type = "Accretion"
                else:
                    continue

                geom = feat.geometry()
                if geom.isNull() or geom.isEmpty():
                    continue

                # Clip against ROI / OSW if provided
                if clip_geom_combined and not clip_geom_combined.isNull():
                    geom = geom.intersection(clip_geom_combined)
                    if geom.isNull() or geom.isEmpty():
                        continue

                area_m2 = float(geom.area())
                if area_m2 < 1.0: # Skip sub-pixel sliver noise
                    continue

                area_ha = round(area_m2 / 10000.0, 4)
                area_m2_rounded = round(area_m2, 2)

                if change_type == "Erosion":
                    erosion_count += 1
                    erosion_area += area_m2
                else:
                    accretion_count += 1
                    accretion_area += area_m2

                out_feat = QgsFeature(fields)
                out_feat.setGeometry(geom)
                out_feat.setAttributes([
                    change_type,
                    area_m2_rounded,
                    area_ha,
                    int(year_a),
                    int(year_b),
                    f"{year_a}-{year_b}"
                ])
                writer.addFeature(out_feat)

        del writer

        if os.path.exists(diff_path):
            try: os.remove(diff_path)
            except Exception: pass
        
        feedback.pushInfo(
            f"✅ Generated Change Polygons: {os.path.basename(out_change_shp)} "
            f"(Erosion: {erosion_count} polygons / {erosion_area:,.1f}m², Accretion: {accretion_count} polygons / {accretion_area:,.1f}m²)"
        )
        return out_change_shp
