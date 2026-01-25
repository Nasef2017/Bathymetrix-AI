# SDB_04_Adaptive.py
# ---------------------------------------------------------------------------
# PHASE 4: SPATIAL RESIDUAL LEARNING (SINGLE THREADED - STABLE)
# Fixes:
# 1. n_jobs=1 (No parallel tasks to prevent QGIS DB crashes)
# 2. Broadcasting error fixed via .flatten()
# 3. Uses exactly the same algorithms selected in Phase 3
# ---------------------------------------------------------------------------

import os
import numpy as np
import rasterio
import shutil
import warnings

# --- SKLEARN IMPORTS ---
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from qgis.core import (
    QgsProcessing, QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer, QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField, QgsProcessingParameterFolderDestination,
    QgsProcessingParameterEnum, QgsProcessingException,
    QgsCoordinateTransform, QgsProject, QgsProcessingParameterNumber,
    QgsRasterLayer
)

try:
    from scipy.ndimage import median_filter
    scipy_is_available = True
except ImportError:
    scipy_is_available = False

warnings.filterwarnings("ignore")

class SDBPhase4Adaptive(QgsProcessingAlgorithm):
    # --- PARAMETER CONSTANTS ---
    INPUT_GLOBAL_RASTER = 'INPUT_GLOBAL_RASTER'
    INPUT_ORIGINAL_FEAT = 'INPUT_ORIGINAL_FEAT'
    INPUT_MASK = 'INPUT_MASK'
    
    INPUT_TRAIN = 'INPUT_TRAIN'
    FIELD_TRAIN = 'FIELD_TRAIN'
    
    INPUT_VALIDATION = 'INPUT_VALIDATION'
    FIELD_VALIDATION = 'FIELD_VALIDATION'
    
    SELECTED_ALGOS = 'SELECTED_ALGOS'
    N_ITERATIONS = 'N_ITERATIONS'
    MEDIAN_SIZE = 'MEDIAN_SIZE'
    OUTPUT_FOLDER = 'OUTPUT_FOLDER'

    MODEL_LIST = [
        'Linear Regression',       # 0
        'Random Forest',           # 1
        'Gradient Boosting',       # 2
        'Extra Trees',             # 3
        'Ridge',                   # 4
        'Lasso',                   # 5
        'ElasticNet',              # 6
        'KNN',                     # 7
        'Decision Tree',           # 8
        'MLP (Neural Net)',        # 9
        'SVR'                      # 10
    ]

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_GLOBAL_RASTER, 'Input Phase 3 Depth'))
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_ORIGINAL_FEAT, 'Input Original Spectral Bands'))
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_MASK, 'Input Water Mask'))
        
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_TRAIN, 'Training Points (Stage 2)'))
        self.addParameter(QgsProcessingParameterField(self.FIELD_TRAIN, 'Depth Field', parentLayerParameterName=self.INPUT_TRAIN, type=QgsProcessingParameterField.Numeric))
        
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_VALIDATION, 'Validation Points (Optional)', optional=True))
        self.addParameter(QgsProcessingParameterField(self.FIELD_VALIDATION, 'Validation Field', parentLayerParameterName=self.INPUT_VALIDATION, type=QgsProcessingParameterField.Numeric, optional=True))

        self.addParameter(QgsProcessingParameterEnum(self.SELECTED_ALGOS, 'Algorithms (Inherited)', options=self.MODEL_LIST, allowMultiple=True, defaultValue=[0, 1])) 
        
        self.addParameter(QgsProcessingParameterNumber(self.N_ITERATIONS, 'Hyperparameter Tuning Iterations', type=QgsProcessingParameterNumber.Integer, defaultValue=10))
        self.addParameter(QgsProcessingParameterNumber(self.MEDIAN_SIZE, 'Post-process Median Filter Size', type=QgsProcessingParameterNumber.Integer, defaultValue=3))
        
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, 'Output Folder'))

    def name(self): return 'sdb_phase4_adaptive'
    def displayName(self): return '4. SDB Phase 4: Spatial Refinement (11 Algos)'
    def group(self): return 'SDB Research Tools'
    def groupId(self): return 'sdb_tools'
    def createInstance(self): return SDBPhase4Adaptive()

    def processAlgorithm(self, parameters, context, feedback):
        out_dir = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        os.makedirs(out_dir, exist_ok=True)
        
        global_path = self.parameterAsRasterLayer(parameters, self.INPUT_GLOBAL_RASTER, context).source()
        feat_path = self.parameterAsRasterLayer(parameters, self.INPUT_ORIGINAL_FEAT, context).source()
        mask_path = self.parameterAsRasterLayer(parameters, self.INPUT_MASK, context).source()
        
        train_lyr = self.parameterAsVectorLayer(parameters, self.INPUT_TRAIN, context)
        train_fld = self.parameterAsString(parameters, self.FIELD_TRAIN, context)
        val_lyr = self.parameterAsVectorLayer(parameters, self.INPUT_VALIDATION, context)
        val_fld = self.parameterAsString(parameters, self.FIELD_VALIDATION, context)
        
        sel_idx = self.parameterAsEnums(parameters, self.SELECTED_ALGOS, context)
        n_iter = self.parameterAsInt(parameters, self.N_ITERATIONS, context)
        med_size = self.parameterAsInt(parameters, self.MEDIAN_SIZE, context)

        feedback.pushInfo("\n>>> STARTING PHASE 4 (STAGE 2 REFINEMENT)...")
        feedback.pushInfo(f"   Selected Algorithms Indices: {sel_idx}")

        # ---------------------------------------------------------------------
        # STEP A: RESIDUAL CALCULATION & SPATIAL INTERPOLATION
        # ---------------------------------------------------------------------
        feedback.pushInfo("   [1/4] Calculating Residuals & Spatial Error Grid...")
        
        z_pred3, z_true, coords_tr = self.extract_values(global_path, train_lyr, train_fld, feedback)
        
        if len(z_pred3) < 5: 
            raise QgsProcessingException("Not enough overlap between Training Points and Phase 3 Raster.")

        # FLATTEN TO FIX BROADCASTING ERROR
        z_pred3 = z_pred3.flatten() 
        z_true = z_true.flatten()
        
        residuals = z_true - z_pred3
        
        # Interpolate Residuals 
        # FIX: n_jobs=1 to prevent QGIS DB collision
        knn_spatial = KNeighborsRegressor(n_neighbors=15, weights='distance', n_jobs=1)
        knn_spatial.fit(coords_tr, residuals)
        
        with rasterio.open(mask_path) as m:
            mask_arr = m.read(1); meta = m.profile; h, w = m.height, m.width
            
        water_indices = np.where(mask_arr == 1)
        water_coords = np.column_stack((water_indices[0], water_indices[1]))
        
        residual_grid = np.zeros((h, w), dtype='float32')
        
        chunk_size = 500000
        for i in range(0, len(water_coords), chunk_size):
            chunk = water_coords[i:i+chunk_size]
            if len(chunk) > 0:
                residual_grid[chunk[:,0], chunk[:,1]] = knn_spatial.predict(chunk)

        p_res = os.path.join(out_dir, 'Stage2_Residual_Map.tif')
        meta.update(dtype='float32', count=1, nodata=-9999)
        with rasterio.open(p_res, 'w', **meta) as dst: dst.write(residual_grid, 1)

        # ---------------------------------------------------------------------
        # STEP B: FEATURE STACKING
        # ---------------------------------------------------------------------
        feedback.pushInfo("   [2/4] Creating Super Stack (Bands + P3 + Error)...")
        
        with rasterio.open(feat_path) as f: orig_bands = f.read()
        with rasterio.open(global_path) as g: p3_map = g.read(1)
            
        stack = np.concatenate([
            orig_bands, 
            p3_map[np.newaxis, :, :], 
            residual_grid[np.newaxis, :, :]
        ], axis=0)
        
        p_stack = os.path.join(out_dir, 'Phase4_Input_Stack.tif')
        meta.update(count=stack.shape[0])
        with rasterio.open(p_stack, 'w', **meta) as dst: dst.write(stack.astype('float32'))

        # ---------------------------------------------------------------------
        # STEP C: RE-TRAINING
        # ---------------------------------------------------------------------
        feedback.pushInfo(f"   [3/4] Re-Training {len(sel_idx)} Algorithms (Single Threaded)...")
        
        X_final, y_final, _ = self.extract_values(p_stack, train_lyr, train_fld, feedback)
        y_final = y_final.flatten()
        
        if val_lyr:
            X_val, y_val, _ = self.extract_values(p_stack, val_lyr, val_fld, feedback)
            y_val = y_val.flatten()
        else:
            X_train, X_val, y_train, y_val = train_test_split(X_final, y_final, test_size=0.2, random_state=42)
            X_final, y_final = X_train, y_train

        best_rmse = float('inf')
        best_model = None
        best_algo_name = ""

        phase4_sub = os.path.join(out_dir, "Models")
        os.makedirs(phase4_sub, exist_ok=True)

        for idx in sel_idx:
            name, model_inst, params = self.get_model_and_params(idx)
            feedback.pushInfo(f"       > Training: {name}...")
            
            try:
                # FIX: n_jobs=1 in GridSearchCV/RandomizedSearchCV to prevent crashes
                if n_iter > 0 and params:
                    search = RandomizedSearchCV(model_inst, params, n_iter=n_iter, cv=3, 
                                              scoring='neg_root_mean_squared_error', n_jobs=1, random_state=42)
                    search.fit(X_final, y_final) 
                    curr_model = search.best_estimator_
                else:
                    curr_model = model_inst
                    curr_model.fit(X_final, y_final)
                
                y_pred = curr_model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                r2 = r2_score(y_val, y_pred)
                
                feedback.pushInfo(f"         Result: RMSE={rmse:.3f}m, R2={r2:.3f}")
                
                with open(os.path.join(phase4_sub, f"{name.replace(' ','_')}_stats.txt"), 'w') as log:
                    log.write(f"RMSE: {rmse}\nR2: {r2}")

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = curr_model
                    best_algo_name = name
                    
            except Exception as e:
                feedback.reportError(f"Error in {name}: {str(e)}")

        # ---------------------------------------------------------------------
        # STEP D: FINAL PREDICTION
        # ---------------------------------------------------------------------
        if best_model is None: 
            raise QgsProcessingException("All selected models failed to train.")
        
        feedback.pushInfo(f"   [4/4] Generating Final Map using Winner: {best_algo_name}")
        
        X_map = stack[:, water_indices[0], water_indices[1]].T
        X_map = np.nan_to_num(X_map, nan=0.0)
        
        z_out = best_model.predict(X_map)
        
        final_map = np.full((h, w), -9999.0, dtype='float32')
        final_map[water_indices] = z_out
        
        if scipy_is_available and med_size > 0:
            valid = final_map != -9999
            temp = final_map.copy(); temp[~valid] = np.nan
            filtered = median_filter(temp, size=med_size)
            final_map[valid] = filtered[valid]

        p_final = os.path.join(out_dir, 'FINAL_PHASE4_DEPTH.tif')
        meta.update(count=1, nodata=-9999.0)
        with rasterio.open(p_final, 'w', **meta) as dst:
            dst.write(final_map, 1)

        QgsProject.instance().addMapLayer(QgsRasterLayer(p_final, f"SDB Phase 4 Final ({best_algo_name})"))
        
        return {'OUTPUT_FINAL': p_final}

    # --- HELPERS ---
    def extract_values(self, ras, vec, fld, fb):
        ds = rasterio.open(ras); d = ds.read(); h, w = ds.height, ds.width
        X, y, c = [], [], []
        tr = QgsCoordinateTransform(vec.sourceCrs(), QgsRasterLayer(ras).crs(), QgsProject.instance())
        
        for f in vec.getFeatures():
            g = f.geometry(); g.transform(tr); pt = g.asPoint()
            r, col = ds.index(pt.x(), pt.y())
            
            if 0<=r<h and 0<=col<w:
                val = d[:, r, col]
                if np.all(np.isfinite(val)) and not np.any(val == -9999):
                    X.append(val)
                    y.append(f[fld])
                    c.append([r, col])
                    
        return np.array(X), np.array(y), np.array(c)

    def get_model_and_params(self, index):
        # ALL n_jobs set to 1 or None (default 1)
        if index == 0: return 'Linear Regression', LinearRegression(), {}
        if index == 1: return 'Random Forest', RandomForestRegressor(n_jobs=1, random_state=42), {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]}
        if index == 2: return 'Gradient Boosting', GradientBoostingRegressor(random_state=42), {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]}
        if index == 3: return 'Extra Trees', ExtraTreesRegressor(n_jobs=1, random_state=42), {'n_estimators': [100], 'max_depth': [None, 10]}
        if index == 4: return 'Ridge', Ridge(), {'alpha': [0.1, 1.0, 10.0]}
        if index == 5: return 'Lasso', Lasso(), {'alpha': [0.01, 0.1, 1.0]}
        if index == 6: return 'ElasticNet', ElasticNet(), {'alpha': [0.1, 1.0], 'l1_ratio': [0.2, 0.5, 0.8]}
        if index == 7: return 'KNN', KNeighborsRegressor(n_jobs=1), {'n_neighbors': [5, 10, 15], 'weights': ['uniform', 'distance']}
        if index == 8: return 'Decision Tree', DecisionTreeRegressor(), {'max_depth': [5, 10, 20]}
        if index == 9: return 'MLP', MLPRegressor(max_iter=500), {'hidden_layer_sizes': [(50,), (100,), (50, 50)]}
        if index == 10: return 'SVR', SVR(), {'C': [1, 10, 100], 'kernel': ['rbf', 'linear']}
        return 'Random Forest', RandomForestRegressor(n_jobs=1), {}