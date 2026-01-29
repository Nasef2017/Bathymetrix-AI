import os
import numpy as np
import pandas as pd
import rasterio
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns

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
        
        # Training Data (For Reference Stats)
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_TRAIN, 'Training Points'))
        self.addParameter(QgsProcessingParameterField(self.FIELD_TRAIN, 'Depth Field (Training)', parentLayerParameterName=self.INPUT_TRAIN, type=QgsProcessingParameterField.Numeric))
        
        # Validation Data (The Judge)
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_VALIDATION, 'Unseen Validation Points'))
        self.addParameter(QgsProcessingParameterField(self.FIELD_VAL_DEPTH, 'Depth Field (Validation)', parentLayerParameterName=self.INPUT_VALIDATION, type=QgsProcessingParameterField.Numeric))
        
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, 'Output Folder (Final Reports)'))

    def name(self): return 'sdb_05_reporting'
    def displayName(self): return '5. SDB Module 05: Ultimate Reporting'
    def group(self): return 'SDB Research Tools'
    def groupId(self): return 'sdb_tools'
    def createInstance(self): return SDBModule05()

    def shortHelpString(self):
        return """
        <div style="font-family: Arial, sans-serif; line-height: 1.2;">
            <h2 style="margin-bottom: 5px;">📉 <span style="color: #2E86C1;">SDB Phase 05</span>: Scientific Validation</h2>
            <p style="margin-top: 0; margin-bottom: 10px;">Performs a rigorous independent accuracy assessment of the produced bathymetry models.</p>

            <b style="display: block; margin-bottom: 2px;">📊 Metrics Calculated</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li><b>RMSE (Root Mean Square Error):</b> Standard metric for vertical error.</li>
                <li><b>R² (Coefficient of Determination):</b> Measures goodness of fit (1.0 is perfect).</li>
                <li><b>wMAPE (Weighted MAPE):</b> A strict, percentage-based error metric suited for bathymetry.</li>
            </ul>

            <b style="display: block; margin-bottom: 2px;">🔍 Stratified Analysis</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li>Breaks down accuracy by depth zones (e.g., 0-5m, 5-10m, 10-20m) to identify where the model performs best/worst.</li>
            </ul>

            <b style="display: block; margin-bottom: 2px;">🖼️ Visual Reporting</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li><b>Scatter Plots:</b> Observed vs. Predicted (with density heatmap).</li>
                <li><b>Residual Plots:</b> Error distribution across depth range.</li>
                <li><b>Histograms:</b> Frequency of errors.</li>
            </ul>
        </div>
        """

    def helpString(self): return self.shortHelpString()

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

        feedback.pushInfo("\n>>> MODULE 05: STARTING FINAL SCIENTIFIC EVALUATION...")

        # 1. Extract Data
        feedback.pushInfo("   [1/5] Sampling Maps (Training & Validation)...")
        y_train, train_p3, train_p4 = self.extract_values(train_lyr, train_fld, init_depth_path, refined_depth_path, feedback)
        y_val, val_p3, val_p4 = self.extract_values(val_lyr, val_fld, init_depth_path, refined_depth_path, feedback)

        if len(y_val) < 5: raise QgsProcessingException("Not enough validation points for a robust report!")

        # 2. Calculate Stats & Stratified Analysis
        feedback.pushInfo("   [2/5] Calculating Advanced Statistics...")
        
        # Global Stats (Validation)
        stats_p3 = self.calc_stats(y_val, val_p3)
        stats_p4 = self.calc_stats(y_val, val_p4)
        
        # Stratified Stats (By Depth Bins)
        strat_stats = []
        strat_stats.extend(self.stratified_analysis(y_val, val_p3, "Phase 3 (Global)"))
        strat_stats.extend(self.stratified_analysis(y_val, val_p4, "Phase 4 (Refined)"))
        
        df_strat = pd.DataFrame(strat_stats)
        df_strat.to_csv(os.path.join(out_dir, '5_Stratified_Error_Analysis.csv'), index=False)

        # 3. Raw Data Export
        feedback.pushInfo("   [3/5] Exporting Raw Data...")
        df_raw = pd.DataFrame({
            'Type': 'Validation', 'Observed': y_val,
            'P3_Pred': val_p3, 'P3_Error': val_p3 - y_val,
            'P4_Pred': val_p4, 'P4_Error': val_p4 - y_val
        })
        df_raw.to_csv(os.path.join(out_dir, '5_Validation_Raw_Data.csv'), index=False)

        # 4. Generate Scientific Plots
        feedback.pushInfo("   [4/5] Generating Scientific Plots...")
        self.plot_scatter(y_val, val_p3, val_p4, stats_p3, stats_p4, out_dir)
        self.plot_residuals(y_val, val_p3, val_p4, out_dir)
        self.plot_histograms(y_val, val_p3, val_p4, out_dir)

        # 5. Final Report & Verdict
        feedback.pushInfo("   [5/5] Writing Final Summary...")
        self.write_final_verdict(out_dir, stats_p3, stats_p4, len(y_val))

        feedback.pushInfo(f">>> REPORTING COMPLETE. Check: {out_dir}")
        return {'OUTPUT_REPORT': os.path.join(out_dir, '5_FINAL_SUMMARY.txt')}

    # =========================================================================
    # CORE FUNCTIONS
    # =========================================================================

    def extract_values(self, vec, fld, p3, p4, fb):
        rlayer = QgsRasterLayer(p3)
        tr = QgsCoordinateTransform(vec.sourceCrs(), rlayer.crs(), QgsProject.instance())
        src3 = rasterio.open(p3); d3 = src3.read(1)
        src4 = rasterio.open(p4); d4 = src4.read(1)
        h, w = src3.height, src3.width
        
        y, p3_vals, p4_vals = [], [], []
        for f in vec.getFeatures():
            geom = f.geometry(); geom.transform(tr); pt = geom.asPoint()
            r, c = src3.index(pt.x(), pt.y())
            if 0 <= r < h and 0 <= c < w:
                v3, v4 = d3[r, c], d4[r, c]
                if v3 > -9000 and v4 > -9000 and np.isfinite(v3) and np.isfinite(v4):
                    y.append(f[fld]); p3_vals.append(v3); p4_vals.append(v4)
        
        src3.close(); src4.close()
        return np.array(y), np.array(p3_vals), np.array(p4_vals)

    def calc_stats(self, y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        bias = np.mean(y_pred - y_true)
        
        # STRICT ABSOLUTE wMAPE
        sum_abs_diff = np.sum(np.abs(y_true - y_pred))
        sum_abs_true = np.sum(np.abs(y_true))
        wmape = (sum_abs_diff / sum_abs_true) * 100 if sum_abs_true != 0 else 0
        
        return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'Bias': bias, 'wMAPE': wmape}

    def stratified_analysis(self, y_true, y_pred, model_name):
        bins = [0, 5, 10, 15, 20, 30, 50, 100]
        labels = [f"{bins[i]}-{bins[i+1]}m" for i in range(len(bins)-1)]
        rows = []
        
        # 1. Global Row
        s = self.calc_stats(y_true, y_pred)
        rows.append({'Model': model_name, 'Depth_Bin': 'GLOBAL', 'Count': len(y_true), **s})
        
        # 2. Bin Rows
        for i in range(len(bins)-1):
            mask = (y_true >= bins[i]) & (y_true < bins[i+1])
            if np.sum(mask) > 5:
                sb = self.calc_stats(y_true[mask], y_pred[mask])
                rows.append({'Model': model_name, 'Depth_Bin': labels[i], 'Count': np.sum(mask), **sb})
                
        return rows

    def write_final_verdict(self, out_dir, s3, s4, count):
        # Determine Winner based on RMSE (Primary) and R2 (Secondary)
        if s4['RMSE'] < s3['RMSE']:
            winner = "PHASE 4 (Refined Model)"
            reason = "Lower RMSE indicates better accuracy."
        elif s4['RMSE'] == s3['RMSE'] and s4['R2'] > s3['R2']:
            winner = "PHASE 4 (Refined Model)"
            reason = "Same RMSE but higher R2."
        elif np.isclose(s4['RMSE'], s3['RMSE'], atol=0.01):
             winner = "TIE (No significant improvement)"
             reason = "Both models performed similarly."
        else:
            winner = "PHASE 3 (Global Model)"
            reason = "Phase 4 did not improve RMSE (Possible overfitting in P4)."

        imp_rmse = ((s3['RMSE'] - s4['RMSE']) / s3['RMSE']) * 100
        imp_wmape = ((s3['wMAPE'] - s4['wMAPE']) / s3['wMAPE']) * 100

        report = [
            "===========================================================",
            "              SDB FINAL SCIENTIFIC REPORT                 ",
            "===========================================================",
            f"Validation Points Used: {count}",
            "",
            "--- METRICS COMPARISON (VALIDATION SET) ---",
            f"{'Metric':<10} | {'Phase 3 (Global)':<20} | {'Phase 4 (Refined)':<20} | {'Improvement':<15}",
            "-"*75,
            f"{'RMSE':<10} | {s3['RMSE']:.4f} m            | {s4['RMSE']:.4f} m            | {imp_rmse:+.2f} %",
            f"{'R2':<10} | {s3['R2']:.4f}              | {s4['R2']:.4f}              | {(s4['R2']-s3['R2'])*100:+.2f} pts",
            f"{'wMAPE':<10} | {s3['wMAPE']:.2f} %             | {s4['wMAPE']:.2f} %             | {imp_wmape:+.2f} %",
            f"{'Bias':<10} | {s3['Bias']:.4f} m            | {s4['Bias']:.4f} m            | -",
            "",
            "--- FINAL VERDICT ---",
            f"WINNER: {winner}",
            f"REASON: {reason}",
            "",
            "NOTE: wMAPE is calculated as Sum(|Error|) / Sum(|Observed|).",
            "==========================================================="
        ]
        
        with open(os.path.join(out_dir, '5_FINAL_SUMMARY.txt'), 'w') as f:
            f.write("\n".join(report))

    # =========================================================================
    # PLOTTING FUNCTIONS
    # =========================================================================

    def plot_scatter(self, obs, p3, p4, s3, s4, out_dir):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Use Density plot if points < 5000 for speed/look
        use_kde = len(obs) < 5000
        
        self._subplot_scatter(axes[0], obs, p3, s3, "Phase 3: Global Model", use_kde)
        self._subplot_scatter(axes[1], obs, p4, s4, "Phase 4: Refined Model", use_kde)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '5_Plot_Scatter_Comparison.png'), dpi=150)
        plt.close()

    def _subplot_scatter(self, ax, obs, pred, stats, title, use_kde):
        if use_kde:
            try:
                xy = np.vstack([obs, pred])
                z = gaussian_kde(xy)(xy)
                sc = ax.scatter(obs, pred, c=z, s=20, cmap='viridis', edgecolor='')
                plt.colorbar(sc, ax=ax, label='Point Density')
            except:
                ax.scatter(obs, pred, c='navy', alpha=0.4, s=15)
        else:
            ax.scatter(obs, pred, c='navy', alpha=0.3, s=10)

        # 1:1 Line
        mx = max(obs.max(), pred.max())
        ax.plot([0, mx], [0, mx], 'r--', lw=2, label='1:1 Line')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Observed Depth (m)')
        ax.set_ylabel('Predicted Depth (m)')
        ax.grid(True, alpha=0.3)
        
        text = f"$R^2 = {stats['R2']:.3f}$\n$RMSE = {stats['RMSE']:.2f}m$\n$wMAPE = {stats['wMAPE']:.2f}\%$"
        ax.text(0.05, 0.95, text, transform=ax.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    def plot_residuals(self, obs, p3, p4, out_dir):
        res3 = p3 - obs
        res4 = p4 - obs
        
        plt.figure(figsize=(10, 6))
        plt.scatter(obs, res3, alpha=0.4, label='Phase 3 Residuals', color='gray', s=15)
        plt.scatter(obs, res4, alpha=0.4, label='Phase 4 Residuals', color='dodgerblue', s=15)
        plt.axhline(0, color='red', linestyle='--', lw=2)
        plt.xlabel('Observed Depth (m)')
        plt.ylabel('Residual Error (Pred - Obs) [m]')
        plt.title('Residuals vs Depth Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, '5_Plot_Residuals.png'), dpi=150)
        plt.close()

    def plot_histograms(self, obs, p3, p4, out_dir):
        res3 = p3 - obs
        res4 = p4 - obs
        
        plt.figure(figsize=(10, 6))
        sns.histplot(res3, color="gray", label="Phase 3 Error", kde=True, stat="density", alpha=0.4, element="step")
        sns.histplot(res4, color="dodgerblue", label="Phase 4 Error", kde=True, stat="density", alpha=0.4, element="step")
        
        plt.axvline(0, color='red', linestyle='--')
        plt.title('Error Distribution Histogram')
        plt.xlabel('Error (m)')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(os.path.join(out_dir, '5_Plot_Error_Histogram.png'), dpi=150)
        plt.close()