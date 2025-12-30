# SDB_04_Adaptive.py
# ---------------------------------------------------------------------------
# PHASE 4: SPATIAL STACKING & RE-TRAINING (FIXED FOR ORCHESTRATOR)
# ID: sdb_phase4_adaptive
# ---------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import rasterio
import shutil
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
    QgsProcessingParameterEnum, QgsProcessingException,
    QgsCoordinateTransform, QgsProject, QgsProcessingParameterNumber,
    QgsRasterLayer, QgsProcessingParameterBoolean
)

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    from scipy.ndimage import median_filter
    scipy_is_available = True
except ImportError:
    scipy_is_available = False

class SDBPhase4Adaptive(QgsProcessingAlgorithm):
    # Inputs (Matching Orchestrator P4 Params)
    INPUT_GLOBAL_RASTER = 'INPUT_GLOBAL_RASTER'
    INPUT_ORIGINAL_FEAT = 'INPUT_ORIGINAL_FEAT'
    INPUT_MASK = 'INPUT_MASK'
    INPUT_TRAIN = 'INPUT_TRAIN'; FIELD_TRAIN = 'FIELD_TRAIN'
    INPUT_VALIDATION = 'INPUT_VALIDATION'; FIELD_VALIDATION = 'FIELD_VALIDATION'
    SELECTED_ALGOS = 'SELECTED_ALGOS'; N_ITERATIONS = 'N_ITERATIONS'; MEDIAN_SIZE = 'MEDIAN_SIZE'
    OUTPUT_FOLDER = 'OUTPUT_FOLDER'

    MODEL_LIST = [
        'Linear Regression', 'Random Forest', 'Gradient Boosting', 'Extra Trees',
        'Ridge', 'Lasso', 'ElasticNet', 'KNN', 'Decision Tree', 'MLP (Neural Net)', 'SVR'
    ]

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_GLOBAL_RASTER, 'Input Global SDB (Phase 2)'))
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_ORIGINAL_FEAT, 'Input Original Features (Phase 1)'))
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_MASK, 'Input Water Mask'))
        
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_TRAIN, 'Training Points'))
        self.addParameter(QgsProcessingParameterField(self.FIELD_TRAIN, 'Depth Field (Training)', parentLayerParameterName=self.INPUT_TRAIN, type=QgsProcessingParameterField.Numeric))
        
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_VALIDATION, 'Unseen Validation Points'))
        self.addParameter(QgsProcessingParameterField(self.FIELD_VALIDATION, 'Depth Field (Validation)', parentLayerParameterName=self.INPUT_VALIDATION, type=QgsProcessingParameterField.Numeric))

        self.addParameter(QgsProcessingParameterEnum(self.SELECTED_ALGOS, 'Select Algorithms', options=self.MODEL_LIST, allowMultiple=True, defaultValue=[0, 1, 2, 3]))
        self.addParameter(QgsProcessingParameterNumber(self.N_ITERATIONS, 'Optimization Iterations', type=QgsProcessingParameterNumber.Integer, defaultValue=10, minValue=0))
        self.addParameter(QgsProcessingParameterNumber(self.MEDIAN_SIZE, 'Median Filter Size', type=QgsProcessingParameterNumber.Integer, defaultValue=3, minValue=1))
        
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, 'Output Folder'))

    def name(self): return 'sdb_phase4_adaptive'
    def displayName(self): return '4. SDB Phase 4: Spatial Stacking (Super Learner)'
    def group(self): return 'SDB Research Tools'
    def groupId(self): return 'sdb_tools'
    def createInstance(self): return SDBPhase4Adaptive()

    def processAlgorithm(self, parameters, context, feedback):
        out_dir = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        os.makedirs(out_dir, exist_ok=True)
        
        # Load Inputs
        global_path = self.parameterAsRasterLayer(parameters, self.INPUT_GLOBAL_RASTER, context).source()
        feat_path = self.parameterAsRasterLayer(parameters, self.INPUT_ORIGINAL_FEAT, context).source()
        mask_path = self.parameterAsRasterLayer(parameters, self.INPUT_MASK, context).source()
        
        train_lyr = self.parameterAsVectorLayer(parameters, self.INPUT_TRAIN, context)
        train_fld = self.parameterAsString(parameters, self.FIELD_TRAIN, context)
        val_lyr = self.parameterAsVectorLayer(parameters, self.INPUT_VALIDATION, context)
        val_fld = self.parameterAsString(parameters, self.FIELD_VALIDATION, context)
        
        n_iter = self.parameterAsInt(parameters, self.N_ITERATIONS, context)
        med_size = self.parameterAsInt(parameters, self.MEDIAN_SIZE, context)
        sel_idx = self.parameterAsEnums(parameters, self.SELECTED_ALGOS, context)
        
        phase4_models_dir = os.path.join(out_dir, 'Phase4_Spatial_Models')
        os.makedirs(phase4_models_dir, exist_ok=True)

        if not sel_idx: raise QgsProcessingException("Select at least one algorithm.")

        feedback.pushInfo("\n>>> STARTING PHASE 4: SPATIAL STACKING & RE-TRAINING...")

        # ---------------------------------------------------------------------
        # 1. GENERATE SPATIAL ERROR
        # ---------------------------------------------------------------------
        feedback.pushInfo("   [1/4] Calculating Residuals & Error Map...")
        g_vals, obs_vals, coords_px = self.extract_values_robust(global_path, train_lyr, train_fld, feedback)
        
        if len(g_vals) < 5: raise QgsProcessingException("Not enough points found.")

        residuals = obs_vals - g_vals
        
        # IDW (KNN)
        knn = KNeighborsRegressor(n_neighbors=10, weights='distance', n_jobs=1)
        knn.fit(coords_px, residuals)
        
        with rasterio.open(mask_path) as m: mask_arr = m.read(1)
        with rasterio.open(global_path) as g: h, w = g.height, g.width
        
        water_rows, water_cols = np.where(mask_arr == 1)
        water_coords = np.column_stack((water_rows, water_cols))
        
        error_surface = np.zeros((h, w), dtype='float32')
        chunk_size = 500000
        for i in range(0, len(water_coords), chunk_size):
            chunk = water_coords[i:i+chunk_size]
            error_surface[chunk[:,0], chunk[:,1]] = knn.predict(chunk)
            
        p_error = os.path.join(out_dir, 'Spatial_Error_Feature.tif')
        with rasterio.open(global_path) as src:
            prof = src.profile; prof.update(dtype='float32', count=1, nodata=-9999.0)
            with rasterio.open(p_error, 'w', **prof) as dst: dst.write(error_surface, 1)

        # ---------------------------------------------------------------------
        # 2. CREATE SUPER STACK
        # ---------------------------------------------------------------------
        feedback.pushInfo("   [2/4] Stacking Features [Original + Global + Error]...")
        
        with rasterio.open(feat_path) as src_f: orig_data = src_f.read()
        with rasterio.open(global_path) as src_g: glob_data = src_g.read(1)
            
        # Super Stack Structure: [Original Bands..., Global Depth, Spatial Error]
        super_stack_data = np.concatenate([
            orig_data, 
            glob_data[np.newaxis, :, :], 
            error_surface[np.newaxis, :, :]
        ], axis=0)
        
        p_super_stack = os.path.join(out_dir, 'Super_Feature_Stack.tif')
        prof.update(count=super_stack_data.shape[0])
        with rasterio.open(p_super_stack, 'w', **prof) as dst: dst.write(super_stack_data.astype('float32'))

        # ---------------------------------------------------------------------
        # 3. RE-TRAIN & SELECT (WEIGHTED)
        # ---------------------------------------------------------------------
        feedback.pushInfo(f"   [3/4] Re-Training {len(sel_idx)} Algorithms...")
        
        X_super, y_super, _ = self.extract_values_robust(p_super_stack, train_lyr, train_fld, feedback)
        
        best_name, best_model, best_score, best_rmse = self.evaluate_all_models(
            X_super, y_super, sel_idx, phase4_models_dir, p_super_stack, mask_path, 
            feedback, med_size, n_iter, scipy_is_available
        )
        
        feedback.pushInfo(f"\n   >>> PHASE 4 WINNER: {best_name} (Score: {best_score:.4f})")

        # ---------------------------------------------------------------------
        # 4. FINAL PREDICTION
        # ---------------------------------------------------------------------
        feedback.pushInfo("   [4/4] Final Prediction...")
        
        best_safe_name = best_name.replace(" ", "_").lower()
        src_tif = os.path.join(phase4_models_dir, best_safe_name, f"{best_safe_name}_depth_filtered.tif")
        if not os.path.exists(src_tif): src_tif = os.path.join(phase4_models_dir, best_safe_name, f"{best_safe_name}_depth.tif")
            
        p_final = os.path.join(out_dir, 'FINAL_REFINED_DEPTH.tif')
        if os.path.exists(src_tif): shutil.copy(src_tif, p_final)
        
        QgsProject.instance().addMapLayer(QgsRasterLayer(p_final, "Final Refined SDB"))
        QgsProject.instance().addMapLayer(QgsRasterLayer(p_error, "Spatial Error Map"))

        return {'OUTPUT_FINAL': p_final}

    # =========================================================================
    # HELPERS
    # =========================================================================

    def extract_values_robust(self, ras_path, vec_layer, fld, fb):
        rlayer = QgsRasterLayer(ras_path)
        tr = QgsCoordinateTransform(vec_layer.sourceCrs(), rlayer.crs(), QgsProject.instance())
        with rasterio.open(ras_path) as src:
            d = src.read()
            h, w = src.height, src.width
            vals_ras, vals_vec, coords_px = [], [], []
            for f in vec_layer.getFeatures():
                geom = f.geometry(); geom.transform(tr); pt = geom.asPoint()
                r, c = src.index(pt.x(), pt.y())
                if 0 <= r < h and 0 <= c < w:
                    if d[0, r, c] != -9999 and np.isfinite(d[0, r, c]):
                        if d.shape[0] == 1: vals_ras.append(d[0, r, c])
                        else: vals_ras.append(d[:, r, c])
                        vals_vec.append(f[fld]); coords_px.append([r, c])
            return np.array(vals_ras), np.array(vals_vec), coords_px

    def get_model_and_params(self, index):
        if index == 0: return 'Linear Regression', LinearRegression(), {}
        if index == 1: return 'Random Forest', RandomForestRegressor(n_jobs=1, random_state=42), {'n_estimators': [50, 100], 'max_depth': [None, 10]}
        if index == 2: return 'Gradient Boosting', GradientBoostingRegressor(random_state=42), {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
        if index == 3: return 'Extra Trees', ExtraTreesRegressor(n_jobs=1, random_state=42), {'n_estimators': [50, 100], 'max_depth': [None, 10]}
        if index == 4: return 'Ridge', Ridge(), {'alpha': [0.1, 1.0]}
        if index == 5: return 'Lasso', Lasso(), {'alpha': [0.1, 1.0]}
        if index == 6: return 'ElasticNet', ElasticNet(), {'alpha': [0.1, 1.0], 'l1_ratio': [0.2, 0.5]}
        if index == 7: return 'KNN', KNeighborsRegressor(), {'n_neighbors': [3, 5, 7]}
        if index == 8: return 'Decision Tree', DecisionTreeRegressor(), {'max_depth': [5, 10]}
        if index == 9: return 'MLP', MLPRegressor(max_iter=500), {'hidden_layer_sizes': [(50,), (100,)], 'activation': ['relu']}
        if index == 10: return 'SVR', SVR(), {'C': [1, 10], 'kernel': ['rbf']}
        return 'Unknown', LinearRegression(), {}

    def evaluate_all_models(self, X, y, algo_indices, root_dir, feat_path, mask_path, fb, med_size, n_iter, scipy_ok):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        with rasterio.open(mask_path) as m: mask_arr = m.read(1).flatten()
        with rasterio.open(feat_path) as f:
            f_d = f.read(); h, w = f.height, f.width
            f_flat = f_d.reshape(f_d.shape[0], -1).T
            water_indices = np.where(mask_arr == 1)[0]
            X_water = np.nan_to_num(f_flat[water_indices], nan=0.0, posinf=0.0, neginf=0.0)

        results = []
        for idx in algo_indices:
            name, base, params = self.get_model_and_params(idx)
            mdir = os.path.join(root_dir, name.replace(" ", "_").lower()); os.makedirs(mdir, exist_ok=True)
            try:
                model = base
                if n_iter > 0 and params:
                    search = RandomizedSearchCV(base, params, n_iter=n_iter, cv=3, n_jobs=1, scoring='r2', random_state=42)
                    search.fit(X_tr, y_tr); model = search.best_estimator_
                else: model.fit(X_tr, y_tr)
                
                yp = model.predict(X_val)
                r2 = r2_score(y_val, yp); rmse = np.sqrt(mean_squared_error(y_val, yp))
                sy = y_val.copy(); sy[sy==0]=0.001; mape = np.mean(np.abs((y_val-yp)/sy))*100
                
                results.append({'Algorithm': name, 'model': model, 'R2': r2, 'RMSE': rmse, 'MAPE': mape, 'dir': mdir, 'safe_name': name.replace(" ", "_").lower()})
                
                with open(os.path.join(mdir, "report.txt"), "w") as f: f.write(f"R2: {r2}\nRMSE: {rmse}\nMAPE: {mape}")
            except Exception as e: pass

        if not results: return "None", None, 0, 0

        df = pd.DataFrame(results)
        min_rmse, max_rmse = df['RMSE'].min(), df['RMSE'].max()
        df['n_rmse'] = 1 - ((df['RMSE'] - min_rmse) / (max_rmse - min_rmse + 1e-6))
        min_mape, max_mape = df['MAPE'].min(), df['MAPE'].max()
        df['n_mape'] = 1 - ((df['MAPE'] - min_mape) / (max_mape - min_mape + 1e-6))
        df['score'] = 0.6 * df['R2'].clip(lower=0) + 0.2 * df['n_rmse'] + 0.2 * df['n_mape']
        
        winner = df.loc[df['score'].idxmax()]
        best = winner['model']; best.fit(X, y)
        out = np.full(h*w, -9999.0, dtype='float32')
        out[water_indices] = best.predict(X_water)
        out = out.reshape(h, w)
        
        prof = rasterio.open(feat_path).profile; prof.update(count=1, dtype='float32', nodata=-9999.0)
        with rasterio.open(os.path.join(winner['dir'], f"{winner['safe_name']}_depth.tif"), 'w', **prof) as dst: dst.write(out, 1)
        
        if scipy_ok:
            valid = (out != -9999.0); temp = out.copy(); temp[~valid] = 0
            filt = median_filter(temp, size=med_size); filt[~valid] = -9999.0
            with rasterio.open(os.path.join(winner['dir'], f"{winner['safe_name']}_depth_filtered.tif"), 'w', **prof) as dst: dst.write(filt, 1)

        return winner['Algorithm'], best, winner['score'], winner['RMSE']