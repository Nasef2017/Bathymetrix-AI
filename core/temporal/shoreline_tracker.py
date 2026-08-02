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

        # 3. Filter Erosion (1) and Accretion (-1)
        erosion_poly = processing.run(
            "native:extractbyattribute",
            {
                "INPUT": poly_res["OUTPUT"],
                "FIELD": "DN",
                "OPERATOR": 0, # =
                "VALUE": "1",
                "OUTPUT": "TEMPORARY_OUTPUT"
            }
        )["OUTPUT"]

        accretion_poly = processing.run(
            "native:extractbyattribute",
            {
                "INPUT": poly_res["OUTPUT"],
                "FIELD": "DN",
                "OPERATOR": 0, # =
                "VALUE": "-1",
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
