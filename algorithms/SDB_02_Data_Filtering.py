# SDB_02_Data_Filtering.py
# ---------------------------------------------------------------------------
# MODULE 02: SDB DATA FILTERING & ROBUST MODELING
# ---------------------------------------------------------------------------
# ALGORITHM FEATURES:
# 1. Robust Auto-Scaling: Maps 5th percentile of data to 10.0.
#    - Ensures logs are safe and data is well-spread for RANSAC.
# 2. Physics Guard: Pre-filters ratios to valid range [0.1 - 5.0].
#    - Eliminates extreme math artifacts (garbage data).
# 3. Smart Plotting: Clean axis (no duplicates) & auto-zoom.
# ---------------------------------------------------------------------------

import os
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings

warnings.filterwarnings("ignore")

from qgis.core import (
    QgsProcessing, QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer, QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField, QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber, QgsCoordinateTransform, QgsProject,
    QgsVectorFileWriter, QgsWkbTypes, QgsRasterLayer, QgsProcessingException, QgsVectorLayer,
    QgsProcessingParameterBand
)

from sklearn.linear_model import RANSACRegressor, LinearRegression

class SDBModule02(QgsProcessingAlgorithm):
    INPUT_STACK = 'INPUT_STACK'
    INPUT_POINTS = 'INPUT_POINTS'
    FIELD_DEPTH = 'FIELD_DEPTH'
    
    BLUE_BAND = 'BLUE_BAND'
    GREEN_BAND = 'GREEN_BAND'
    
    RESIDUAL_THRESHOLD = 'RESIDUAL_THRESHOLD'
    OUTPUT_FOLDER = 'OUTPUT_FOLDER'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_STACK, 'Input Feature Stack (Phase 1)'))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_POINTS, 'Raw ICESat-2 Points'))
        self.addParameter(QgsProcessingParameterField(self.FIELD_DEPTH, 'Depth Field', parentLayerParameterName=self.INPUT_POINTS, type=QgsProcessingParameterField.Numeric))
        
        self.addParameter(QgsProcessingParameterBand(self.BLUE_BAND, 'Blue Band Number', parentLayerParameterName=self.INPUT_STACK, defaultValue=2))
        self.addParameter(QgsProcessingParameterBand(self.GREEN_BAND, 'Green Band Number', parentLayerParameterName=self.INPUT_STACK, defaultValue=3))
        
        self.addParameter(QgsProcessingParameterNumber(self.RESIDUAL_THRESHOLD, 'RANSAC Threshold (0 = Auto-Calculate)', type=QgsProcessingParameterNumber.Double, defaultValue=0.0))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, 'Output Folder'))

    def name(self): return 'sdb_02_filtering'
    def displayName(self): return '2. SDB Module 02: Filtering (Final Robust)'
    def group(self): return 'SDB Research Tools'
    def groupId(self): return 'sdb_tools'
    def createInstance(self): return SDBModule02()

    def processAlgorithm(self, parameters, context, feedback):
        out_dir = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        os.makedirs(out_dir, exist_ok=True)
        
        stack_path = self.parameterAsRasterLayer(parameters, self.INPUT_STACK, context).source()
        points_layer = self.parameterAsVectorLayer(parameters, self.INPUT_POINTS, context)
        depth_fld = self.parameterAsString(parameters, self.FIELD_DEPTH, context)
        
        b_idx = self.parameterAsInt(parameters, self.BLUE_BAND, context)
        g_idx = self.parameterAsInt(parameters, self.GREEN_BAND, context)
        user_threshold = self.parameterAsDouble(parameters, self.RESIDUAL_THRESHOLD, context)

        feedback.pushInfo("\n>>> MODULE 02: STARTING ROBUST FILTERING...")
        
        # ---------------------------------------------------------
        # 1. EXTRACT DATA & CALCULATE RATIOS (ROBUST METHOD)
        # ---------------------------------------------------------
        feedback.pushInfo("   [1/3] Extracting Data & Applying Robust Scaling...")
        X_ratio, y_depth, features_list, calc_scale = self.extract_and_calc_robust(
            stack_path, points_layer, depth_fld, b_idx, g_idx, feedback
        )
        
        if len(y_depth) < 10: 
            raise QgsProcessingException("Not enough valid points found! Check input data or projection.")

        feedback.pushInfo(f"   >>> Auto-Scale Factor Used: {calc_scale:.2f}")

        # ---------------------------------------------------------
        # 2. DETERMINE THRESHOLD
        # ---------------------------------------------------------
        X = X_ratio.reshape(-1, 1)
        y = y_depth
        
        final_thresh = user_threshold
        if user_threshold <= 0:
            # Calculate MAD (Median Absolute Deviation)
            lr = LinearRegression().fit(X, y)
            preds = lr.predict(X)
            residuals = np.abs(y - preds)
            mad = np.median(residuals)
            final_thresh = 2.5 * mad 
            feedback.pushInfo(f"   [Auto] Calculated Threshold: {final_thresh:.4f}m (MAD={mad:.4f})")
        else:
            feedback.pushInfo(f"   Using Manual Threshold: {final_thresh}m")

        # ---------------------------------------------------------
        # 3. RUN RANSAC
        # ---------------------------------------------------------
        feedback.pushInfo("   [2/3] Running RANSAC Model...")
        # Max trials increased to 1000 to find best line in spread data
        ransac = RANSACRegressor(random_state=42, min_samples=0.1, residual_threshold=final_thresh, max_trials=1000)
        try:
            ransac.fit(X, y)
        except ValueError as e:
            raise QgsProcessingException(f"RANSAC Failed: {e}")
        
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        
        n_in = np.sum(inlier_mask); n_out = np.sum(outlier_mask)
        feedback.pushInfo(f"      Status: {n_in} Inliers (Kept), {n_out} Outliers (Rejected)")

        # ---------------------------------------------------------
        # 4. EXPORT RESULTS
        # ---------------------------------------------------------
        feedback.pushInfo("   [3/3] Exporting Results...")
        clean_shp = os.path.join(out_dir, '2_Cleaned_Training_Data.shp')
        reject_shp = os.path.join(out_dir, '2_Outliers_Rejected.shp')
        plot_path = os.path.join(out_dir, '2_RANSAC_Plot.png')
        
        self.save_subset(points_layer, features_list, inlier_mask, clean_shp, "Cleaned Data")
        self.save_subset(points_layer, features_list, outlier_mask, reject_shp, "Rejected Data")
        self.save_ransac_plot(X, y, inlier_mask, outlier_mask, ransac, plot_path, final_thresh, calc_scale)
        
        return {'OUTPUT_CLEAN_VEC': clean_shp}

    # =========================================================================
    # HELPER FUNCTIONS
    # =========================================================================
    
    def extract_and_calc_robust(self, ras_path, vec_layer, depth_fld, b_idx, g_idx, fb):
        """
        Extracts raw values, calculates robust scale based on 5th percentile,
        and filters impossible physics ratios.
        """
        rlayer = QgsRasterLayer(ras_path)
        source_crs = vec_layer.sourceCrs()
        dest_crs = rlayer.crs()
        tr = QgsCoordinateTransform(source_crs, dest_crs, QgsProject.instance())
        
        with rasterio.open(ras_path) as src:
            if b_idx > src.count or g_idx > src.count: raise QgsProcessingException("Band index error.")
            band_b = src.read(b_idx)
            band_g = src.read(g_idx)
            nodata = src.nodata if src.nodata is not None else -9999.0
            h, w = src.height, src.width

            raw_b_list = []
            raw_g_list = []
            depths_list = []
            feats_list = []
            
            # --- Pass 1: Collect Raw Data ---
            total = vec_layer.featureCount()
            for f in vec_layer.getFeatures():
                d = f[depth_fld]
                # Filter NULL and Land (Depth > 0)
                if d is None or d > 0: continue 

                geom = f.geometry()
                try: geom.transform(tr)
                except: continue
                pt = geom.asPoint()
                try: r, c = src.index(pt.x(), pt.y())
                except: continue
                
                if 0 <= r < h and 0 <= c < w:
                    val_b = band_b[r, c]
                    val_g = band_g[r, c]
                    
                    if (val_b > 0 and val_g > 0 and 
                        val_b != nodata and val_g != nodata and 
                        np.isfinite(val_b) and np.isfinite(val_g)):
                        
                        raw_b_list.append(val_b)
                        raw_g_list.append(val_g)
                        depths_list.append(d)
                        feats_list.append(f)
            
            if not raw_b_list: return np.array([]), np.array([]), [], 1.0

            # --- Pass 2: Calculate Robust Scale ---
            np_b = np.array(raw_b_list)
            np_g = np.array(raw_g_list)
            
            # Use 5th Percentile to ignore noise/dead pixels
            min_b = np.percentile(np_b, 5)
            min_g = np.percentile(np_g, 5)
            min_val = min(min_b, min_g)
            
            if min_val <= 1e-6: min_val = 0.0001 # Prevent zero division

            # MAP BASELINE TO 10.0
            # This ensures Logs are in a safe, steep range for good spread.
            TARGET_BASE = 10.0
            robust_scale = TARGET_BASE / min_val
            
            # Clamp scale to prevent overflow
            if robust_scale > 100000: robust_scale = 100000.0

            # --- Pass 3: Apply Scale & Log ---
            np_b_scaled = np_b * robust_scale
            np_g_scaled = np_g * robust_scale
            
            # Safety Clamp: Force values > 1.1 to avoid Log(1)=0 or negative
            np_b_scaled = np.maximum(np_b_scaled, 1.1)
            np_g_scaled = np.maximum(np_g_scaled, 1.1)

            lb = np.log(np_b_scaled)
            lg = np.log(np_g_scaled)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                ratios = lb / lg
            
            # --- Pass 4: PHYSICS FILTER ---
            # Remove math artifacts (e.g. Ratio = 4000)
            # Valid SDB ratios are usually between 0.5 and 2.0. We allow 0.1 - 5.0.
            valid_physics = (ratios > 0.1) & (ratios < 5.0) & (np.isfinite(ratios))
            
            ratios_final = ratios[valid_physics]
            depths_final = np.array(depths_list)[valid_physics]
            feats_final = [feats_list[i] for i in range(len(valid_physics)) if valid_physics[i]]
            
            fb.pushInfo(f"      Extracted {len(ratios_final)} points (Physics Filtered).")
            
            return ratios_final, depths_final, feats_final, robust_scale

    def save_subset(self, original_layer, all_features, mask, out_path, layer_name):
        subset = [f for f, m in zip(all_features, mask) if m]
        if not subset: return
        if os.path.exists(out_path):
            try: os.remove(out_path)
            except: pass 
        writer = QgsVectorFileWriter(out_path, "UTF-8", original_layer.fields(), QgsWkbTypes.Point, original_layer.sourceCrs(), "ESRI Shapefile")
        if writer.hasError() == QgsVectorFileWriter.NoError:
            for f in subset: writer.addFeature(f)
            del writer
            QgsProject.instance().addMapLayer(QgsVectorLayer(out_path, layer_name, "ogr"))

    def save_ransac_plot(self, X, y, inlier, outlier, model, path, thresh, scale):
        plt.figure(figsize=(12, 8))
        
        # 1. Plot Points
        plt.scatter(X[inlier], y[inlier], c='dodgerblue', s=25, alpha=0.7, label='Inliers (Clean)')
        plt.scatter(X[outlier], y[outlier], c='crimson', s=25, marker='x', alpha=0.4, label='Outliers (Rejected)')
        
        # 2. Plot Model Line
        x_min, x_max = X.min(), X.max()
        margin_x = (x_max - x_min) * 0.1
        line_x = np.linspace(x_min - margin_x, x_max + margin_x, 100).reshape(-1, 1)
        plt.plot(line_x, model.predict(line_x), 'k-', lw=3, label='RANSAC Model')
        
        # 3. Titles & Labels
        plt.title(f"SDB Robust Filtering (Scale={scale:.1f})", fontsize=14, fontweight='bold')
        plt.xlabel("Stumpf Ratio (Ln(Blue)/Ln(Green))", fontsize=12)
        plt.ylabel("Depth (m)", fontsize=12)
        
        # 4. FIX AXIS (No Duplicates) & SMART ZOOM
        ax = plt.gca()
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True)) # Force integer ticks
        ax.tick_params(axis='both', which='major', labelsize=11)
        
        # Zoom Y-axis to relevant data (Deepest inlier - 2m buffer)
        if len(y[inlier]) > 0:
            max_depth_inlier = np.min(y[inlier])
            plt.ylim(max_depth_inlier - 2.0, 0.5) 
        
        # Zoom X-axis to avoid showing empty space if an outlier was cut off
        plt.xlim(x_min - margin_x, x_max + margin_x)

        plt.legend(loc='upper right', fontsize=12)
        plt.grid(True, alpha=0.5, linestyle='--', which='both')
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()