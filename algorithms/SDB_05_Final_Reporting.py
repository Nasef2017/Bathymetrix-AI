# SDB_05_Final_Reporting.py
# ---------------------------------------------------------------------------
# MODULE 05: THE ULTIMATE REPORTER (FIXED PLOTTING ARGS)
# Features:
# 1. Calculates metrics for BOTH Training and Validation Data.
# 2. Generates Stratified wMAPE Report.
# 3. Fixed Plotting function call.
# ---------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import rasterio
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

from qgis.core import (
    QgsProcessing, QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer, QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField, QgsProcessingParameterFolderDestination,
    QgsCoordinateTransform, QgsProject, QgsRasterLayer, QgsProcessingException
)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class SDBModule05(QgsProcessingAlgorithm):
    # --- INPUT CONSTANTS ---
    INPUT_MAP_P3 = 'INPUT_MAP_P3'       # Initial Depth Map
    INPUT_MAP_P4 = 'INPUT_MAP_P4'       # Refined Depth Map
    
    INPUT_TRAIN = 'INPUT_TRAIN'         # Training Points
    FIELD_TRAIN = 'FIELD_TRAIN'
    
    INPUT_VALIDATION = 'INPUT_VALIDATION' # Unseen Points
    FIELD_VAL_DEPTH = 'FIELD_VAL_DEPTH'
    
    OUTPUT_FOLDER = 'OUTPUT_FOLDER'

    def initAlgorithm(self, config=None):
        # Maps Inputs
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_MAP_P3, 'Input Phase 3 Map (Initial Global Depth)'))
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_MAP_P4, 'Input Phase 4 Map (Final Refined Depth)'))
        
        # Training Data
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_TRAIN, 'Training Points'))
        self.addParameter(QgsProcessingParameterField(self.FIELD_TRAIN, 'Depth Field (Training)', parentLayerParameterName=self.INPUT_TRAIN, type=QgsProcessingParameterField.Numeric))
        
        # Validation Data
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_VALIDATION, 'Unseen Validation Points'))
        self.addParameter(QgsProcessingParameterField(self.FIELD_VAL_DEPTH, 'Depth Field (Validation)', parentLayerParameterName=self.INPUT_VALIDATION, type=QgsProcessingParameterField.Numeric))
        
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, 'Output Folder (Final Reports)'))

    def name(self): return 'sdb_05_reporting'
    def displayName(self): return '5. SDB Module 05: Comprehensive Reporting'
    def group(self): return 'SDB Research Tools'
    def groupId(self): return 'sdb_tools'
    def createInstance(self): return SDBModule05()

    def processAlgorithm(self, parameters, context, feedback):
        out_dir = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        os.makedirs(out_dir, exist_ok=True)
        
        # Paths
        init_depth_path = self.parameterAsRasterLayer(parameters, self.INPUT_MAP_P3, context).source()
        refined_depth_path = self.parameterAsRasterLayer(parameters, self.INPUT_MAP_P4, context).source()
        
        train_lyr = self.parameterAsVectorLayer(parameters, self.INPUT_TRAIN, context)
        train_fld = self.parameterAsString(parameters, self.FIELD_TRAIN, context)
        
        val_lyr = self.parameterAsVectorLayer(parameters, self.INPUT_VALIDATION, context)
        val_fld = self.parameterAsString(parameters, self.FIELD_VAL_DEPTH, context)

        feedback.pushInfo("\n>>> MODULE 05: STARTING FINAL EVALUATION & REPORTING...")

        # 1. Extract Data (Training)
        feedback.pushInfo("   [1/4] Evaluating Training Data...")
        y_train, train_p3, train_p4 = self.extract_values_from_maps(
            train_lyr, train_fld, init_depth_path, refined_depth_path, feedback
        )

        # 2. Extract Data (Validation)
        feedback.pushInfo("   [2/4] Evaluating Unseen Validation Data...")
        y_val, val_p3, val_p4 = self.extract_values_from_maps(
            val_lyr, val_fld, init_depth_path, refined_depth_path, feedback
        )

        if len(y_val) < 5: raise QgsProcessingException("Not enough validation points found!")

        # 3. Calculate Stats
        feedback.pushInfo("   [3/4] Generating Statistics & CSVs...")
        
        # Calculate Stats Objects
        stats_val_p3 = self.calc_stats(y_val, val_p3)
        stats_val_p4 = self.calc_stats(y_val, val_p4)
        
        # A. Stratified Stats
        all_stats = []
        all_stats.extend(self.calc_stratified_stats(y_train, train_p3, "Phase 3 (Global)", "Training"))
        all_stats.extend(self.calc_stratified_stats(y_val, val_p3, "Phase 3 (Global)", "Unseen Validation"))
        all_stats.extend(self.calc_stratified_stats(y_train, train_p4, "Phase 4 (Refined)", "Training"))
        all_stats.extend(self.calc_stratified_stats(y_val, val_p4, "Phase 4 (Refined)", "Unseen Validation"))
        
        df_stats = pd.DataFrame(all_stats)
        
        # Reorder columns
        cols = ['Dataset', 'Phase', 'Depth_Bin', 'Count', 'R2', 'RMSE', 'MAE', 'wMAPE (%)', 'Bias']
        # Filter existing columns only
        cols = [c for c in cols if c in df_stats.columns]
        df_stats = df_stats[cols]
        
        df_stats.to_csv(os.path.join(out_dir, '5_MASTER_STRATIFIED_STATS.csv'), index=False)

        # B. Raw Data CSV
        df_raw_val = pd.DataFrame({
            'Type': 'Validation', 'Observed': y_val,
            'P3_Pred': val_p3, 'P4_Pred': val_p4,
            'P3_MAPE': self.calc_mape_vector(y_val, val_p3),
            'P4_MAPE': self.calc_mape_vector(y_val, val_p4)
        })
        df_raw_train = pd.DataFrame({
            'Type': 'Training', 'Observed': y_train,
            'P3_Pred': train_p3, 'P4_Pred': train_p4,
            'P3_MAPE': self.calc_mape_vector(y_train, train_p3),
            'P4_MAPE': self.calc_mape_vector(y_train, train_p4)
        })
        pd.concat([df_raw_val, df_raw_train]).to_csv(os.path.join(out_dir, '5_MASTER_RAW_DATA.csv'), index=False)

        # C. Text Summary
        self.save_summary_txt(out_dir, stats_val_p3, stats_val_p4, len(y_val))

        # 4. Generate Plots
        feedback.pushInfo("   [4/4] Generating Plots...")
        
        # --- FIX: Passing all required arguments correctly ---
        self.plot_scatter_comparison(
            y_val, val_p3, val_p4, 
            stats_val_p3, stats_val_p4, 
            out_dir
        )
        # ---------------------------------------------------

        feedback.pushInfo(f">>> MODULE 05 COMPLETE. Reports saved to: {out_dir}")
        return {'OUTPUT_REPORT': os.path.join(out_dir, '5_MASTER_STRATIFIED_STATS.csv')}

    # =========================================================================
    # LOGIC
    # =========================================================================

    def extract_values_from_maps(self, vec_layer, fld, path3, path4, fb):
        rlayer = QgsRasterLayer(path3)
        tr = QgsCoordinateTransform(vec_layer.sourceCrs(), rlayer.crs(), QgsProject.instance())
        
        src3 = rasterio.open(path3); d3 = src3.read(1).astype('float32')
        src4 = rasterio.open(path4); d4 = src4.read(1).astype('float32')
        h, w = src3.height, src3.width
        
        y_list, p3_list, p4_list = [], [], []
        total = vec_layer.featureCount()
        
        for f in vec_layer.getFeatures():
            geom = f.geometry()
            try: geom.transform(tr)
            except: continue
            pt = geom.asPoint()
            try: r, c = src3.index(pt.x(), pt.y())
            except: continue
            
            if 0 <= r < h and 0 <= c < w:
                val3 = d3[r, c]
                val4 = d4[r, c]
                if val3 > -9000 and val4 > -9000 and np.isfinite(val3) and np.isfinite(val4):
                    y_list.append(f[fld])
                    p3_list.append(val3)
                    p4_list.append(val4)
        
        src3.close(); src4.close()
        fb.pushInfo(f"      Extracted: {len(y_list)} / {total} points.")
        return np.array(y_list), np.array(p3_list), np.array(p4_list)

    def calc_mape_vector(self, y_true, y_pred):
        safe_obs = y_true.copy(); safe_obs[safe_obs==0] = 0.001
        return np.abs((y_true - y_pred) / safe_obs) * 100

    def calc_stats(self, y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        bias = np.mean(y_pred - y_true)
        
        sum_obs = np.sum(y_true)
        sum_err = np.sum(np.abs(y_true - y_pred))
        wmape = (sum_err / sum_obs) * 100 if sum_obs != 0 else 0
        
        return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'Bias': bias, 'wMAPE': wmape}

    def calc_stratified_stats(self, y_true, y_pred, phase_name, dataset_name):
        if len(y_true) < 2: return []
        max_d = int(np.ceil(max(y_true)))
        bins = range(0, max_d + 5, 5)
        
        rows = []
        
        # Global
        s = self.calc_stats(y_true, y_pred)
        rows.append({
            'Phase': phase_name, 'Dataset': dataset_name, 'Depth_Bin': 'GLOBAL',
            'Count': len(y_true), 'R2': round(s['R2'], 4), 'RMSE': round(s['RMSE'], 4),
            'MAE': round(s['MAE'], 4), 'wMAPE (%)': round(s['wMAPE'], 2), 'Bias': round(s['Bias'], 4)
        })
        
        # Bins
        for i in range(len(bins)-1):
            low, high = bins[i], bins[i+1]
            mask = (y_true >= low) & (y_true < high)
            if np.sum(mask) > 1:
                yo = y_true[mask]; yp = y_pred[mask]
                sb = self.calc_stats(yo, yp)
                rows.append({
                    'Phase': phase_name, 'Dataset': dataset_name, 'Depth_Bin': f"{low}-{high}m",
                    'Count': len(yo), 'R2': round(sb['R2'], 4), 'RMSE': round(sb['RMSE'], 4),
                    'MAE': round(sb['MAE'], 4), 'wMAPE (%)': round(sb['wMAPE'], 2), 'Bias': round(sb['Bias'], 4)
                })
        return rows

    def save_summary_txt(self, out_dir, s3, s4, n):
        imp_r2 = ((s4['R2'] - s3['R2']) / s3['R2']) * 100 if s3['R2']!=0 else 0
        imp_rmse = ((s3['RMSE'] - s4['RMSE']) / s3['RMSE']) * 100
        
        with open(os.path.join(out_dir, '5_FINAL_SUMMARY.txt'), 'w') as f:
            f.write("SDB RESEARCH PROJECT - FINAL REPORT\n=================================\n\n")
            f.write(f"Total Unseen Validation Points: {n}\n\n")
            f.write("PHASE 3 (INITIAL GLOBAL):\n")
            f.write(f"  R2: {s3['R2']:.4f} | RMSE: {s3['RMSE']:.4f}m | wMAPE: {s3['wMAPE']:.2f}%\n")
            f.write("PHASE 4 (FINAL REFINED):\n")
            f.write(f"  R2: {s4['R2']:.4f} | RMSE: {s4['RMSE']:.4f}m | wMAPE: {s4['wMAPE']:.2f}%\n")
            f.write("\nIMPROVEMENT:\n")
            f.write(f"  R2 Gain: {imp_r2:+.2f}%\n  RMSE Reduction: {imp_rmse:+.2f}%\n")

    def plot_scatter_comparison(self, obs, p3, p4, stats3, stats4, out_dir):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Optimized plotting: Use KDE only if points < 2000 to save time
        use_kde = len(obs) < 2000
        
        self._subplot(axes[0], obs, p3, stats3, "Phase 3: Global Model", use_kde)
        self._subplot(axes[1], obs, p4, stats4, "Phase 4: Refined Model", use_kde)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '5_Comparative_Scatter.png'), dpi=150)
        plt.close()

    def _subplot(self, ax, obs, pred, stats, title, use_kde):
        if use_kde:
            try:
                xy = np.vstack([obs, pred])
                z = gaussian_kde(xy)(xy)
                ax.scatter(obs, pred, c=z, s=30, cmap='viridis', edgecolor='')
            except:
                ax.scatter(obs, pred, c='blue', alpha=0.5)
        else:
             ax.scatter(obs, pred, c='blue', alpha=0.3, s=10)

        ax.plot([min(obs), max(obs)], [min(obs), max(obs)], 'r--', lw=2)
        ax.set_title(title); ax.set_xlabel('Observed (m)'); ax.set_ylabel('Predicted (m)')
        ax.grid(True, alpha=0.5)
        text = f"$R^2={stats['R2']:.3f}$\n$RMSE={stats['RMSE']:.2f}m$\n$wMAPE={stats['wMAPE']:.2f}\%$"
        ax.text(0.05, 0.95, text, transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))