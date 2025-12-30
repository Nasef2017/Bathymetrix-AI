# SDB_02_Data_Filtering.py
# ---------------------------------------------------------------------------
# MODULE 02: ICESAT-2 DATA FILTERING (ROBUST DEBUG EDITION)
# ---------------------------------------------------------------------------
# Features:
# - Detailed Logging for CRS and Spatial Overlap.
# - Robust Coordinate Transformation.
# - RANSAC Outlier Removal.
# ---------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

from qgis.core import (
    QgsProcessing, QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer, QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField, QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber, QgsCoordinateTransform, QgsProject,
    QgsVectorFileWriter, QgsWkbTypes, QgsRasterLayer, QgsProcessingException, QgsVectorLayer
)

from sklearn.linear_model import RANSACRegressor, LinearRegression

class SDBModule02(QgsProcessingAlgorithm):
    INPUT_STACK = 'INPUT_STACK'
    INPUT_POINTS = 'INPUT_POINTS'
    FIELD_DEPTH = 'FIELD_DEPTH'
    RATIO_BAND_INDEX = 'RATIO_BAND_INDEX'
    RESIDUAL_THRESHOLD = 'RESIDUAL_THRESHOLD'
    OUTPUT_FOLDER = 'OUTPUT_FOLDER'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_STACK, 'Input Feature Stack'))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_POINTS, 'Raw ICESat-2 Points'))
        self.addParameter(QgsProcessingParameterField(self.FIELD_DEPTH, 'Depth Field', parentLayerParameterName=self.INPUT_POINTS, type=QgsProcessingParameterField.Numeric))
        self.addParameter(QgsProcessingParameterNumber(self.RATIO_BAND_INDEX, 'Band Index of Ratio (LogB/LogG)', type=QgsProcessingParameterNumber.Integer, defaultValue=11))
        self.addParameter(QgsProcessingParameterNumber(self.RESIDUAL_THRESHOLD, 'RANSAC Threshold (0 = Auto)', type=QgsProcessingParameterNumber.Double, defaultValue=0.0))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, 'Output Folder'))

    def name(self): return 'sdb_02_filtering'
    def displayName(self): return '2. SDB Module 02: Data Filtering (Robust)'
    def group(self): return 'SDB Research Tools'
    def groupId(self): return 'sdb_tools'
    def createInstance(self): return SDBModule02()

    def processAlgorithm(self, parameters, context, feedback):
        out_dir = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        os.makedirs(out_dir, exist_ok=True)
        
        stack_path = self.parameterAsRasterLayer(parameters, self.INPUT_STACK, context).source()
        points_layer = self.parameterAsVectorLayer(parameters, self.INPUT_POINTS, context)
        depth_fld = self.parameterAsString(parameters, self.FIELD_DEPTH, context)
        ratio_idx = self.parameterAsInt(parameters, self.RATIO_BAND_INDEX, context)
        user_threshold = self.parameterAsDouble(parameters, self.RESIDUAL_THRESHOLD, context)

        feedback.pushInfo("\n>>> MODULE 02: STARTING DATA FILTERING...")

        # 1. Extract Data with Debugging
        feedback.pushInfo("   [1/3] Extracting Data...")
        X_ratio, y_depth, features_list = self.extract_data_debug(stack_path, points_layer, depth_fld, ratio_idx, feedback)
        
        if len(y_depth) < 10: 
            raise QgsProcessingException("Not enough valid points found! See log above for reasons (Outside Extent / NoData).")

        # 2. Threshold Calculation
        X = X_ratio.reshape(-1, 1)
        y = y_depth
        
        final_thresh = user_threshold
        if user_threshold <= 0:
            feedback.pushInfo("   [Auto] Calculating MAD Threshold...")
            lr = LinearRegression().fit(X, y)
            preds = lr.predict(X)
            residuals = np.abs(y - preds)
            mad = np.median(residuals)
            final_thresh = 2.5 * mad
            feedback.pushInfo(f"      Calculated Threshold: {final_thresh:.4f}m (MAD={mad:.4f})")
        else:
            feedback.pushInfo(f"   Using Manual Threshold: {final_thresh}m")

        # 3. RANSAC
        feedback.pushInfo("   [2/3] Running RANSAC...")
        ransac = RANSACRegressor(random_state=42, min_samples=10, residual_threshold=final_thresh)
        ransac.fit(X, y)
        
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        
        n_in = np.sum(inlier_mask); n_out = np.sum(outlier_mask)
        feedback.pushInfo(f"      Kept: {n_in}, Rejected: {n_out}")

        # 4. Export
        feedback.pushInfo("   [3/3] Exporting...")
        clean_shp = os.path.join(out_dir, '2_Cleaned_Training_Data.shp')
        reject_shp = os.path.join(out_dir, '2_Outliers_Rejected.shp')
        
        self.save_subset(points_layer, features_list, inlier_mask, clean_shp, "Cleaned Data")
        self.save_subset(points_layer, features_list, outlier_mask, reject_shp, "Rejected Data")
        self.save_ransac_plot(X, y, inlier_mask, outlier_mask, ransac, os.path.join(out_dir, '2_RANSAC_Plot.png'), final_thresh)
        
        return {'OUTPUT_CLEAN_VEC': clean_shp}

    # --- Helpers ---
    def extract_data_debug(self, ras_path, vec_layer, depth_fld, band_idx, fb):
        rlayer = QgsRasterLayer(ras_path)
        source_crs = vec_layer.sourceCrs()
        dest_crs = rlayer.crs()
        
        fb.pushInfo(f"      Points CRS: {source_crs.authid()}")
        fb.pushInfo(f"      Raster CRS: {dest_crs.authid()}")
        
        tr = QgsCoordinateTransform(source_crs, dest_crs, QgsProject.instance())
        
        with rasterio.open(ras_path) as src:
            try:
                d = src.read(band_idx)
            except IndexError:
                raise QgsProcessingException(f"Band {band_idx} not found. Raster has {src.count} bands.")
            
            nodata = src.nodata if src.nodata is not None else -9999.0
            h, w = src.height, src.width
            
            vr, vd, f_obj = [], [], []
            
            # Counters for debugging
            total = vec_layer.featureCount()
            outside = 0
            is_nodata = 0
            transform_fail = 0
            success = 0
            
            for f in vec_layer.getFeatures():
                geom = f.geometry()
                try: geom.transform(tr)
                except: 
                    transform_fail += 1
                    continue
                
                pt = geom.asPoint()
                try: r, c = src.index(pt.x(), pt.y())
                except: 
                    outside += 1
                    continue
                
                if 0 <= r < h and 0 <= c < w:
                    val = d[r, c]
                    # Check validity
                    if val != nodata and np.isfinite(val) and val != -9999:
                        # Check depth
                        depth = f[depth_fld]
                        if depth is not None:
                            vr.append(val)
                            vd.append(depth)
                            f_obj.append(f)
                            success += 1
                    else:
                        is_nodata += 1
                else:
                    outside += 1
            
            fb.pushInfo(f"      --- EXTRACTION SUMMARY ---")
            fb.pushInfo(f"      Total Points: {total}")
            fb.pushInfo(f"      Successful:   {success}")
            fb.pushInfo(f"      Outside Img:  {outside}")
            fb.pushInfo(f"      NoData Pixels:{is_nodata}")
            fb.pushInfo(f"      Transfrm Err: {transform_fail}")
            
            return np.array(vr), np.array(vd), f_obj

    def save_subset(self, original_layer, all_features, mask, out_path, layer_name):
        subset = [f for f, m in zip(all_features, mask) if m]
        if not subset: return
        
        # Ensure directory exists and delete old file if locked (try)
        if os.path.exists(out_path):
            try: os.remove(out_path)
            except: pass 
            
        writer = QgsVectorFileWriter(out_path, "UTF-8", original_layer.fields(), QgsWkbTypes.Point, original_layer.sourceCrs(), "ESRI Shapefile")
        if writer.hasError() == QgsVectorFileWriter.NoError:
            for f in subset: writer.addFeature(f)
            del writer
            QgsProject.instance().addMapLayer(QgsVectorLayer(out_path, layer_name, "ogr"))

    def save_ransac_plot(self, X, y, inlier, outlier, model, path, thresh):
        plt.figure(figsize=(8, 6))
        plt.scatter(X[inlier], y[inlier], c='blue', marker='.', label='Inliers')
        plt.scatter(X[outlier], y[outlier], c='red', marker='x', label='Outliers')
        
        if len(X[inlier]) > 1:
            line_x = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            plt.plot(line_x, model.predict(line_x), 'k-', lw=2, label='RANSAC Model')
        
        plt.title(f"RANSAC Filtering (Threshold={thresh:.2f}m)")
        plt.xlabel("Log-Ratio"); plt.ylabel("Depth")
        plt.legend(); plt.grid(True, alpha=0.5)
        plt.savefig(path); plt.close()