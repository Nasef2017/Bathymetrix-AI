import os
import math
import numpy as np
import rasterio
from typing import Dict, List, Optional, Tuple, Any
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

class ShorelineDynamicsTracker:
    def __init__(self):
        pass

    def extract_water_mask_from_depth(self, year: int, depth_map_path: str, output_dir: str, feedback: QgsProcessingFeedback, shoreline_depth: float = 0.0) -> str:
        """
        Creates a binary water mask from a depth map based on the shoreline_depth threshold.
        Pixels > shoreline_depth are water (1), otherwise land (0).
        """
        mask_path = os.path.join(output_dir, f"temp_water_mask_{year}.tif")
        with rasterio.open(depth_map_path) as src:
            depth = src.read(1)
            water_mask = np.zeros_like(depth, dtype=np.uint8)
            # Valid data pixels
            valid = depth != src.nodata
            # Apply threshold for water (Depth is negative, so < shoreline_depth is water)
            water_mask[valid & (depth < shoreline_depth)] = 1
            
            profile = src.profile.copy()
            profile.update(count=1, dtype=rasterio.uint8, nodata=255)
            # Set nodata to 255 for the output mask so polygonize doesn't struggle
            water_mask[~valid] = 255
            
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
                    
        # 3. Always compute Overall Trend (First vs Last) if not already done in baseline
        if len(years) > 2 or "Sequential" in comparison_mode:
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
            green = img[green_idx - 1].astype(np.float32)
            nir = img[nir_idx - 1].astype(np.float32)

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
            
            # CRITICAL FIX: Set unchanged pixels (0) to NoData (-9999) 
            # so GDAL Polygonize ignores them instead of vectorizing millions of 0-value pixels!
            diff[diff == 0] = -9999
            
            diff_path = os.path.join(output_dir, f"temp_diff_{year_a}_{year_b}.tif")
            profile = srcA.profile.copy()
            profile.update(dtype=rasterio.int16, nodata=-9999)
            with rasterio.open(diff_path, "w", **profile) as dst:
                dst.write(diff, 1)

        # 2. Polygonize Diff (Now ultra-fast as it only processes non-zero change pixels!)
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

        # 3. Filter Erosion (1) and Accretion (-1) using expression to avoid string/int mismatch
        erosion_poly = processing.run(
            "native:extractbyexpression",
            {
                "INPUT": poly_res["OUTPUT"],
                "EXPRESSION": '"DN" = 1',
                "OUTPUT": "TEMPORARY_OUTPUT"
            }
        )["OUTPUT"]

        accretion_poly = processing.run(
            "native:extractbyexpression",
            {
                "INPUT": poly_res["OUTPUT"],
                "EXPRESSION": '"DN" = -1',
                "OUTPUT": "TEMPORARY_OUTPUT"
            }
        )["OUTPUT"]

        # 4. Clip by OSW or Shoreline ROI if provided
        if osw_shp and os.path.exists(osw_shp):
            erosion_poly = processing.run("native:clip", {"INPUT": erosion_poly, "OVERLAY": osw_shp, "OUTPUT": "TEMPORARY_OUTPUT"})["OUTPUT"]
            accretion_poly = processing.run("native:clip", {"INPUT": accretion_poly, "OVERLAY": osw_shp, "OUTPUT": "TEMPORARY_OUTPUT"})["OUTPUT"]
            
        if shoreline_roi_shp and os.path.exists(shoreline_roi_shp):
            erosion_poly = processing.run("native:clip", {"INPUT": erosion_poly, "OVERLAY": shoreline_roi_shp, "OUTPUT": "TEMPORARY_OUTPUT"})["OUTPUT"]
            accretion_poly = processing.run("native:clip", {"INPUT": accretion_poly, "OVERLAY": shoreline_roi_shp, "OUTPUT": "TEMPORARY_OUTPUT"})["OUTPUT"]

        # 5. Add fields and labels
        def update_layer_field(layer, val):
            if isinstance(layer, str):
                layer = QgsVectorLayer(layer, "temp", "ogr")
            if not layer.isValid():
                return layer
            pr = layer.dataProvider()
            pr.addAttributes([QgsField("ChangeType", QVariant.String, len=50)])
            layer.updateFields()
            idx = layer.fields().indexFromName("ChangeType")
            
            layer.startEditing()
            for f in layer.getFeatures():
                f[idx] = val
                layer.updateFeature(f)
            layer.commitChanges()
            return layer

        e_layer = update_layer_field(erosion_poly, 'Erosion')
        a_layer = update_layer_field(accretion_poly, 'Accretion')
        
        # 6. Merge and save
        from qgis.core import QgsCoordinateReferenceSystem
        with rasterio.open(mask_a_path) as srcA:
            crs_wkt = srcA.crs.to_wkt()
        target_crs = QgsCoordinateReferenceSystem(crs_wkt)

        processing.run("native:mergevectorlayers", {
            "LAYERS": [e_layer, a_layer],
            "CRS": target_crs,
            "OUTPUT": out_change_shp
        })

        if os.path.exists(diff_path):
            try: os.remove(diff_path)
            except: pass
        
        feedback.pushInfo(f"✅ Generated Change Polygons: {os.path.basename(out_change_shp)}")
        return out_change_shp
