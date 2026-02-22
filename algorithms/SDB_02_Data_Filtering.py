import os
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

# ML & Stats Libraries
from sklearn.linear_model import RANSACRegressor, LinearRegression, HuberRegressor
from sklearn.preprocessing import PolynomialFeatures
from PyQt5.QtCore import QVariant

warnings.filterwarnings("ignore")

from qgis.core import (
    QgsProcessing, QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer, QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField, QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber, QgsProcessingParameterEnum,
    QgsCoordinateTransform, QgsProject,
    QgsVectorFileWriter, QgsWkbTypes, QgsRasterLayer, QgsProcessingException, QgsVectorLayer,
    QgsProcessingParameterBand, QgsField, QgsFeature, QgsRectangle
)

class SDBModule02(QgsProcessingAlgorithm):
    INPUT_STACK = 'INPUT_STACK'
    INPUT_POINTS = 'INPUT_POINTS'
    FIELD_DEPTH = 'FIELD_DEPTH'
    BLUE_BAND = 'BLUE_BAND'
    GREEN_BAND = 'GREEN_BAND'
    RESIDUAL_THRESHOLD = 'RESIDUAL_THRESHOLD'
    RANSAC_MAX_TRIALS = 'RANSAC_MAX_TRIALS' 
    FILTER_MODE = 'FILTER_MODE'
    OUTPUT_FOLDER = 'OUTPUT_FOLDER'

    FILTER_MODES = [
        'Linear RANSAC',
        'LS Variance Fit',
        'Huber Variance Fit'
    ]

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_STACK, 'Input Feature Stack (Phase 1)'))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_POINTS, 'Raw ICESat-2 Points'))
        self.addParameter(QgsProcessingParameterField(self.FIELD_DEPTH, 'Depth Field', parentLayerParameterName=self.INPUT_POINTS, type=QgsProcessingParameterField.Numeric))
        self.addParameter(QgsProcessingParameterBand(self.BLUE_BAND, 'Blue Band Number', parentLayerParameterName=self.INPUT_STACK, defaultValue=2))
        self.addParameter(QgsProcessingParameterBand(self.GREEN_BAND, 'Green Band Number', parentLayerParameterName=self.INPUT_STACK, defaultValue=3))
        
        self.addParameter(QgsProcessingParameterEnum(self.FILTER_MODE, 'Filtering Strategy', options=self.FILTER_MODES, defaultValue=2))
        
        # 0 = Auto Calculation
        self.addParameter(QgsProcessingParameterNumber(self.RESIDUAL_THRESHOLD, 'Threshold/Multiplier (0=Auto)', type=QgsProcessingParameterNumber.Double, defaultValue=0.0))
        self.addParameter(QgsProcessingParameterNumber(self.RANSAC_MAX_TRIALS, 'RANSAC Max Trials', type=QgsProcessingParameterNumber.Integer, defaultValue=100))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, 'Output Folder'))

    def name(self): return 'sdb_02_filtering'
    def displayName(self): return '2. SDB Module 02: Filtering (Multi-Mode)'
    def group(self): return 'SDB Research Tools'
    def groupId(self): return 'sdb_tools'
    def createInstance(self): return SDBModule02()

    def shortHelpString(self):
        return """
        <div style="font-family: Arial, sans-serif; line-height: 1.2;">
            <h2 style="margin-bottom: 5px;">🔬 Filtering Strategy Guide</h2>
            <p>This module filters noisy ICESat-2 data using one of three statistical methods.</p>
            <b style="display: block; margin-bottom: 2px;">1. Linear RANSAC</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li><b>Best for:</b> Data with a clear linear relationship but contaminated with significant, random outliers.</li>
            </ul>
            <b style="display: block; margin-bottom: 2px;">2. LS Variance Fit</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li><b>Best for:</b> Data with a non-linear trend where the noise level is expected to be constant across all depths.</li>
            </ul>
            <b style="display: block; margin-bottom: 2px;">3. Huber Variance Fit</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li><b>Best for:</b> Complex scenarios where data uncertainty increases with depth (heteroscedasticity).</li>
            </ul>
        </div>
        """

    def helpString(self): return self.shortHelpString()

    def processAlgorithm(self, parameters, context, feedback):
        out_dir = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        os.makedirs(out_dir, exist_ok=True)
        stack_path = self.parameterAsRasterLayer(parameters, self.INPUT_STACK, context).source()
        points_layer = self.parameterAsVectorLayer(parameters, self.INPUT_POINTS, context)
        depth_fld = self.parameterAsString(parameters, self.FIELD_DEPTH, context)
        b_idx = self.parameterAsInt(parameters, self.BLUE_BAND, context)
        g_idx = self.parameterAsInt(parameters, self.GREEN_BAND, context)
        
        mode_idx = self.parameterAsInt(parameters, self.FILTER_MODE, context)
        user_val = self.parameterAsDouble(parameters, self.RESIDUAL_THRESHOLD, context)
        max_trials = self.parameterAsInt(parameters, self.RANSAC_MAX_TRIALS, context)

        feedback.pushInfo(f"\n>>> MODULE 02: STARTING FILTERING [{self.FILTER_MODES[mode_idx]}]...")
        
        X_ratio, y_depth, features_list, calc_scale = self.extract_and_calc_robust(
            stack_path, points_layer, depth_fld, b_idx, g_idx, feedback
        )
        
        if len(y_depth) < 20: 
            raise QgsProcessingException("Not enough valid points found (<20)!")

        X = X_ratio.reshape(-1, 1)
        y = y_depth
        
        final_mask = None
        est_sigma = None 
        sigma_model_for_plot = None
        
        # =========================================================
        # MODE 0: Linear RANSAC
        # =========================================================
        if mode_idx == 0:
            feedback.pushInfo("   [Logic] Running Linear RANSAC...")
            if user_val <= 0:
                lr = LinearRegression().fit(X, y)
                residuals_raw = np.abs(y - lr.predict(X))
                mad = np.median(residuals_raw)
                thresh = 2.5 * mad 
                feedback.pushInfo(f"   [Auto] Threshold: {thresh:.4f}m (MAD-based)")
            else:
                thresh = user_val
                feedback.pushInfo(f"   [User] Using Threshold: {thresh:.4f}m")
            
            ransac = RANSACRegressor(random_state=42, min_samples=0.1, residual_threshold=thresh, max_trials=max_trials)
            ransac.fit(X, y)
            final_mask = ransac.inlier_mask_
            est_sigma = np.full_like(y, thresh / 3.0)
            residuals = y - ransac.predict(X)

        # =========================================================
        # MODE 1: LS Variance Fit
        # =========================================================
        elif mode_idx == 1:
            feedback.pushInfo("   [Logic] Running LS Variance Fit...")
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly.fit_transform(X)
            ls_model = LinearRegression().fit(X_poly, y)
            residuals = y - ls_model.predict(X_poly)
            mad = np.median(np.abs(residuals))
            sigma_val = 1.4826 * mad
            
            if user_val <= 0:
                multiplier = 3.0
                feedback.pushInfo(f"   [Auto] Global Sigma: {sigma_val:.4f}m | Multiplier: 3.0")
            else: multiplier = user_val
            
            limit = multiplier * sigma_val
            final_mask = np.abs(residuals) <= limit
            est_sigma = np.full_like(y, sigma_val)
            
            res_sq = residuals**2
            depth_poly_feat = poly.fit_transform(y.reshape(-1, 1))
            sigma_model_for_plot = LinearRegression().fit(depth_poly_feat, res_sq)

        # =========================================================
        # MODE 2: Huber Variance Fit
        # =========================================================
        elif mode_idx == 2:
            feedback.pushInfo("   [Logic] Running Huber Variance Fit...")
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly.fit_transform(X)
            ransac_trend = RANSACRegressor(random_state=42, min_samples=0.1, max_trials=max_trials)
            ransac_trend.fit(X_poly, y)
            
            residuals = y - ransac_trend.predict(X_poly)
            mask_trend = ransac_trend.inlier_mask_
            
            y_clean = y[mask_trend]
            res_sq_clean = residuals[mask_trend]**2
            depth_poly_feat = poly.fit_transform(y_clean.reshape(-1, 1))
            
            sigma_model = HuberRegressor(epsilon=1.35, max_iter=max_trials)
            sigma_model.fit(depth_poly_feat, res_sq_clean)
            sigma_model_for_plot = sigma_model 
            
            all_depth_poly = poly.transform(y.reshape(-1, 1))
            est_sigma_sq = sigma_model.predict(all_depth_poly)
            est_sigma = np.sqrt(np.maximum(est_sigma_sq, 0.0025))
            
            if user_val <= 0:
                multiplier = 3.0
                feedback.pushInfo("   [Auto] Using Default 3-Sigma Envelope")
            else: multiplier = user_val
            
            final_mask = np.abs(residuals) <= (multiplier * est_sigma)

        # =========================================================
        # EXPORT & PLOTTING
        # =========================================================
        clean_shp = os.path.join(out_dir, '2_Cleaned_Training_Data.shp')
        reject_shp = os.path.join(out_dir, '2_Outliers_Rejected.shp')
        
        self.save_subset_with_uncert(points_layer, features_list, final_mask, est_sigma, clean_shp, "Cleaned Data")
        self.save_subset_with_uncert(points_layer, features_list, ~final_mask, est_sigma, reject_shp, "Rejected Data")
        
        plot_mult = user_val if user_val > 0 else (2.5 if mode_idx == 0 else 3.0)
        self.plot_physics_trend(X, y, final_mask, mode_idx, os.path.join(out_dir, '2_Plot_1_Trend.png'))
        self.plot_variance_analysis(y, residuals, sigma_model_for_plot, mode_idx, os.path.join(out_dir, '2_Plot_2_Variance.png'))
        self.plot_envelope_analysis(y, residuals, est_sigma, final_mask, plot_mult, mode_idx, os.path.join(out_dir, '2_Plot_3_Envelope.png'))
        
        return {'OUTPUT_CLEAN_VEC': clean_shp}

    def extract_and_calc_robust(self, ras_path, vec_layer, depth_fld, b_idx, g_idx, fb):
        rlayer = QgsRasterLayer(ras_path)
        tr = QgsCoordinateTransform(vec_layer.sourceCrs(), rlayer.crs(), QgsProject.instance())
        
        # DEBUG COUNTERS
        total_pts = vec_layer.featureCount()
        count_invalid_depth = 0
        count_out_of_bounds = 0
        count_nodata = 0
        count_valid = 0

        with rasterio.open(ras_path) as src:
            band_b = src.read(b_idx); band_g = src.read(g_idx)
            raw_b, raw_g, depths_list, feats_list = [], [], [], []
            
            for f in vec_layer.getFeatures():
                d = f[depth_fld]
                # Check 1: Depth Validity
                if d is None or d == 0: 
                    count_invalid_depth += 1
                    continue 
                
                geom = f.geometry(); geom.transform(tr); pt = geom.asPoint()
                
                # Check 2: Bounds
                try: 
                    r, c = src.index(pt.x(), pt.y())
                except:
                    count_out_of_bounds += 1
                    continue
                
                if 0 <= r < src.height and 0 <= c < src.width:
                    vb, vg = band_b[r, c], band_g[r, c]
                    # Check 3: Data Validity (NoData usually -9999 or 0)
                    if (vb > 0 and vg > 0):
                        raw_b.append(vb); raw_g.append(vg); depths_list.append(d); feats_list.append(f)
                        count_valid += 1
                    else:
                        count_nodata += 1
                else:
                    count_out_of_bounds += 1
            
            fb.pushInfo(f"   [DEBUG] Total Points: {total_pts}")
            fb.pushInfo(f"   [DEBUG] Invalid Depth (0/None): {count_invalid_depth}")
            fb.pushInfo(f"   [DEBUG] Out of Bounds: {count_out_of_bounds}")
            fb.pushInfo(f"   [DEBUG] NoData/Masked Pixels: {count_nodata}")
            fb.pushInfo(f"   [DEBUG] Valid for Training: {count_valid}")

            if count_valid == 0:
                hint = "Hint: Uncheck 'Enable Water Masking' in Phase 1 if 'NoData' is high." if count_nodata > 0 else "Hint: Check CRS alignment."
                raise QgsProcessingException(f"No valid training points found! (NoData={count_nodata}, OutBounds={count_out_of_bounds}). {hint}")
            
            np_b, np_g = np.array(raw_b), np.array(raw_g)
            robust_scale = 10.0 / max(np.percentile(np_b, 5), 0.0001)
            lb, lg = np.log(np.maximum(np_b * robust_scale, 1.1)), np.log(np.maximum(np_g * robust_scale, 1.1))
            ratios = lb/lg; valid = (ratios > 0.1) & (ratios < 5.0)
            return ratios[valid], np.array(depths_list)[valid], [feats_list[i] for i in range(len(valid)) if valid[i]], robust_scale

    def save_subset_with_uncert(self, original_layer, all_features, mask, uncert, out_path, layer_name):
        fields = original_layer.fields()
        fields.append(QgsField("SDB_Uncert", QVariant.Double))
        if os.path.exists(out_path):
            try: os.remove(out_path)
            except: pass 
        writer = QgsVectorFileWriter(out_path, "UTF-8", fields, QgsWkbTypes.Point, original_layer.sourceCrs(), "ESRI Shapefile")
        if writer.hasError() == QgsVectorFileWriter.NoError:
            for i, m in enumerate(mask):
                if m:
                    feat = QgsFeature(fields); feat.setGeometry(all_features[i].geometry())
                    attrs = all_features[i].attributes(); attrs.append(float(uncert[i]))
                    feat.setAttributes(attrs); writer.addFeature(feat)
            del writer
            QgsProject.instance().addMapLayer(QgsVectorLayer(out_path, layer_name, "ogr"))

    def plot_physics_trend(self, X, y, mask, mode, path):
        plt.figure(figsize=(10, 6))
        plt.scatter(X[mask], y[mask], c='dodgerblue', s=15, alpha=0.6, label='Accepted')
        plt.scatter(X[~mask], y[~mask], c='gray', marker='x', s=15, alpha=0.4, label='Rejected')
        
        x_rng = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        if mode == 0:
            model = LinearRegression().fit(X[mask], y[mask])
            y_rng = model.predict(x_rng)
        else:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            model = LinearRegression().fit(poly.fit_transform(X[mask]), y[mask])
            y_rng = model.predict(poly.transform(x_rng))
            
        plt.plot(x_rng, y_rng, 'k-', lw=2, label='Trend Model')
        plt.title("Plot 1: Band Ratio vs Depth"); plt.xlabel("Log Ratio"); plt.ylabel("Depth (m)")
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(path, dpi=100); plt.close()

    def plot_variance_analysis(self, depths, residuals, sigma_model, mode, path):
        if mode == 0 or sigma_model is None: return 
        plt.figure(figsize=(10, 6))
        d_abs = np.abs(depths)
        res_sq = residuals**2
        poly = PolynomialFeatures(degree=2, include_bias=False)
        d_rng = np.linspace(d_abs.min(), d_abs.max(), 100).reshape(-1, 1)
        var_pred = sigma_model.predict(poly.fit_transform(d_rng))
        
        plt.scatter(d_abs, res_sq, c='black', s=10, alpha=0.5, label='Residual Squares')
        plt.plot(d_rng, var_pred, 'r-', lw=2.5, label='Variance Trend ($\sigma^2$)')
        plt.title("Plot 2: Depth vs Residual Square (Variance Analysis)"); 
        plt.xlabel("Depth (m)"); plt.ylabel("Residual² ($m^2$)")
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(path, dpi=100); plt.close()

    def plot_envelope_analysis(self, depths, residuals, sigmas, mask, multiplier, mode, path):
        plt.figure(figsize=(10, 6))
        d_abs = np.abs(depths)
        sort_idx = np.argsort(d_abs)
        d_sorted = d_abs[sort_idx]; sigma_sorted = sigmas[sort_idx]
        upper = multiplier * sigma_sorted; lower = -multiplier * sigma_sorted
        
        plt.plot(d_sorted, upper, 'r-', lw=2, label=f'Upper (+{multiplier}$\sigma$)')
        plt.plot(d_sorted, lower, 'r-', lw=2, label=f'Lower (-{multiplier}$\sigma$)')
        plt.fill_between(d_sorted, lower, upper, color='red', alpha=0.1)
        plt.scatter(d_abs[mask], residuals[mask], c='dodgerblue', s=15, alpha=0.6, label='Accepted')
        plt.scatter(d_abs[~mask], residuals[~mask], c='gray', marker='x', s=15, alpha=0.4, label='Rejected')
        plt.axhline(0, c='black', lw=1)
        plt.title(f"Plot 3: Residuals & Uncertainty Envelope (Mode {mode})")
        plt.xlabel("Depth (m)"); plt.ylabel("Residual (m)")
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(path, dpi=100); plt.close()