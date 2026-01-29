import os
import numpy as np
import rasterio
import warnings
import ast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
# ADDED r2_score here
from sklearn.metrics import mean_squared_error, r2_score

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Categorical, Integer
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

from qgis.core import (
    QgsProcessing, QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer, QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField, QgsProcessingParameterFolderDestination,
    QgsProcessingParameterEnum, QgsProcessingException,
    QgsCoordinateTransform, QgsProject, QgsProcessingParameterNumber,
    QgsRasterLayer, QgsProcessingParameterFile, QgsProcessingParameterString
)

try:
    from scipy.ndimage import median_filter
    scipy_is_available = True
except ImportError:
    scipy_is_available = False

warnings.filterwarnings("ignore")

class SDBPhase4Adaptive(QgsProcessingAlgorithm):
    # --- INPUTS ---
    INPUT_GLOBAL_RASTER = 'INPUT_GLOBAL_RASTER'
    INPUT_ORIGINAL_FEAT = 'INPUT_ORIGINAL_FEAT'
    INPUT_MASK = 'INPUT_MASK'
    INPUT_TRAIN = 'INPUT_TRAIN'
    FIELD_TRAIN = 'FIELD_TRAIN'
    
    # --- CONFIG ---
    SELECTED_ALGOS = 'SELECTED_ALGOS'
    OPTIMIZER_METHOD = 'OPTIMIZER_METHOD'
    COLLISION_HANDLING = 'COLLISION_HANDLING'
    N_ITERATIONS = 'N_ITERATIONS'
    MEDIAN_SIZE = 'MEDIAN_SIZE'
    
    OUTPUT_FOLDER = 'OUTPUT_FOLDER'
    LOG_FILE = 'LOG_FILE'
    
    # --- HYPERPARAMS (Received as Strings from Master) ---
    PARAM_RF = 'PARAM_RF'
    PARAM_GB = 'PARAM_GB'
    PARAM_ET = 'PARAM_ET'
    PARAM_SVR = 'PARAM_SVR'
    PARAM_MLP = 'PARAM_MLP'

    MODEL_LIST = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'Extra Trees', 'Ridge', 'Lasso', 'ElasticNet', 'KNN', 'Decision Tree', 'MLP (Neural Net)', 'SVR']
    OPTIMIZER_LIST = ['Random Search', 'Grid Search', 'Bayesian Search']
    COLLISION_LIST = ['Keep All', 'Highest Conf', 'Closest', 'Hybrid', 'Strict Center']

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_GLOBAL_RASTER, 'Input Phase 3 Depth Map'))
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_ORIGINAL_FEAT, 'Input Original Feature Stack'))
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_MASK, 'Input Water Mask'))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_TRAIN, 'Adaptive Training Points'))
        self.addParameter(QgsProcessingParameterField(self.FIELD_TRAIN, 'Depth Field', parentLayerParameterName=self.INPUT_TRAIN, type=QgsProcessingParameterField.Numeric))
        
        self.addParameter(QgsProcessingParameterEnum(self.SELECTED_ALGOS, 'Refinement Algorithms', options=self.MODEL_LIST, allowMultiple=True, defaultValue=[0, 1])) 
        self.addParameter(QgsProcessingParameterEnum(self.OPTIMIZER_METHOD, 'Optimizer', options=self.OPTIMIZER_LIST, defaultValue=0))
        self.addParameter(QgsProcessingParameterEnum(self.COLLISION_HANDLING, 'Collision Handling', options=self.COLLISION_LIST, defaultValue=0))
        
        self.addParameter(QgsProcessingParameterNumber(self.N_ITERATIONS, 'Optimization Iterations', type=QgsProcessingParameterNumber.Integer, defaultValue=10))
        self.addParameter(QgsProcessingParameterNumber(self.MEDIAN_SIZE, 'Output Median Filter Size', type=QgsProcessingParameterNumber.Integer, defaultValue=3))
        
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, 'Output Folder'))
        self.addParameter(QgsProcessingParameterFile(self.LOG_FILE, 'Log File (Optional)', optional=True))
        
        # Hyperparams Strings
        self.addParameter(QgsProcessingParameterString(self.PARAM_RF, 'RF Params', defaultValue="", optional=True))
        self.addParameter(QgsProcessingParameterString(self.PARAM_GB, 'GB Params', defaultValue="", optional=True))
        self.addParameter(QgsProcessingParameterString(self.PARAM_ET, 'ET Params', defaultValue="", optional=True))
        self.addParameter(QgsProcessingParameterString(self.PARAM_SVR, 'SVR Params', defaultValue="", optional=True))
        self.addParameter(QgsProcessingParameterString(self.PARAM_MLP, 'MLP Params', defaultValue="", optional=True))

    def name(self): return 'sdb_phase4_adaptive'
    def displayName(self): return '4. SDB Module 04: Spatial Refinement'
    def group(self): return 'SDB Research Tools'
    def groupId(self): return 'sdb_tools'
    def createInstance(self): return SDBPhase4Adaptive()

    def shortHelpString(self):
        return """
        <div style="font-family: Arial, sans-serif; line-height: 1.2;">
            <h2 style="margin-bottom: 5px;">📍 <span style="color: #2E86C1;">SDB Module 04</span>: Spatial Refinement</h2>
            <p style="margin-top: 0; margin-bottom: 10px;">Corrects local biases and spatially varying errors using Residual Analysis (Stacking).</p>

            <b style="display: block; margin-bottom: 2px;">📉 Residual Analysis</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li>Calculates the error <i>(Residual = True Depth - Phase 3 Depth)</i> at training points.</li>
                <li>Uses <b>KNN Spatial Interpolation</b> to create a continuous "Error Grid" across the entire image.</li>
            </ul>

            <b style="display: block; margin-bottom: 2px;">📚 Stacked Learning</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li>Combines: <b>[Original Bands] + [Global Depth] + [Error Grid]</b>.</li>
                <li>Trains a secondary "Refinement Model" (using selected algorithms & custom hyperparameters) to predict the final, corrected bathymetry.</li>
            </ul>
        </div>
        """

    def helpString(self): return self.shortHelpString()

    def append_log(self, msg, log_path, feedback):
        feedback.pushInfo(msg)
        if log_path and os.path.exists(log_path):
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(msg + "\n")

    def parse_param_string(self, param_str):
        if not param_str or param_str.strip() == "": return {}
        try:
            return ast.literal_eval("{" + param_str + "}")
        except:
            return {}

    def convert_to_bayes(self, params_dict):
        bayes_params = {}
        for k, v in params_dict.items():
            if isinstance(v, list) and len(v) > 0:
                if all(isinstance(x, int) for x in v): bayes_params[k] = Integer(min(v), max(v))
                elif all(isinstance(x, (int, float)) for x in v): bayes_params[k] = Real(min(v), max(v))
                else: bayes_params[k] = Categorical(v)
            else:
                bayes_params[k] = Categorical(v)
        return bayes_params

    def processAlgorithm(self, parameters, context, feedback):
        out_dir = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        os.makedirs(out_dir, exist_ok=True)
        log_path = self.parameterAsString(parameters, self.LOG_FILE, context)
        
        # --- PARSE CUSTOM HYPERPARAMS ---
        self.custom_params = {
            'Random Forest': self.parse_param_string(self.parameterAsString(parameters, self.PARAM_RF, context)),
            'Gradient Boosting': self.parse_param_string(self.parameterAsString(parameters, self.PARAM_GB, context)),
            'Extra Trees': self.parse_param_string(self.parameterAsString(parameters, self.PARAM_ET, context)),
            'SVR': self.parse_param_string(self.parameterAsString(parameters, self.PARAM_SVR, context)),
            'MLP': self.parse_param_string(self.parameterAsString(parameters, self.PARAM_MLP, context))
        }

        # Inputs
        global_path = self.parameterAsRasterLayer(parameters, self.INPUT_GLOBAL_RASTER, context).source()
        feat_path = self.parameterAsRasterLayer(parameters, self.INPUT_ORIGINAL_FEAT, context).source()
        mask_path = self.parameterAsRasterLayer(parameters, self.INPUT_MASK, context).source()
        train_lyr = self.parameterAsVectorLayer(parameters, self.INPUT_TRAIN, context)
        train_fld = self.parameterAsString(parameters, self.FIELD_TRAIN, context)
        
        sel_idx = self.parameterAsEnums(parameters, self.SELECTED_ALGOS, context)
        n_iter = self.parameterAsInt(parameters, self.N_ITERATIONS, context)
        med_size = self.parameterAsInt(parameters, self.MEDIAN_SIZE, context)
        opt_idx = self.parameterAsInt(parameters, self.OPTIMIZER_METHOD, context)
        
        col_mode = 0 # Force Keep All for Residuals
        self.append_log(f"MODULE 04 START: Refinement Phase | Opt={self.OPTIMIZER_LIST[opt_idx]}", log_path, feedback)
        
        # 1. CALCULATE RESIDUALS
        z_pred3, z_true, coords_tr = self.extract_values(global_path, train_lyr, train_fld, col_mode, log_path, feedback)
        
        if len(z_pred3) < 5: 
            raise QgsProcessingException("Not enough points for Module 4 (Spatial Refinement).")
            
        residuals = z_true.flatten() - z_pred3.flatten()
        
        # 2. CREATE RESIDUAL GRID (KNN)
        knn_spatial = KNeighborsRegressor(n_neighbors=15, weights='distance', n_jobs=1)
        knn_spatial.fit(coords_tr, residuals)
        
        self.append_log("   Interpolating Residual Grid (KNN)...", log_path, feedback)
        with rasterio.open(mask_path) as m:
            mask_arr = m.read(1); meta = m.profile; h, w = m.height, m.width
        water_indices = np.where(mask_arr == 1)
        water_coords = np.column_stack((water_indices[0], water_indices[1]))
        
        residual_grid = np.zeros((h, w), dtype='float32')
        chunk_size = 500000
        for i in range(0, len(water_coords), chunk_size):
            chunk = water_coords[i:i+chunk_size]
            if len(chunk) > 0: residual_grid[chunk[:,0], chunk[:,1]] = knn_spatial.predict(chunk)
        
        # 3. CREATE STACK (Original + Global + Residual)
        with rasterio.open(feat_path) as f: orig_bands = f.read()
        with rasterio.open(global_path) as g: p3_map = g.read(1)
        
        stack = np.concatenate([orig_bands, p3_map[np.newaxis, :, :], residual_grid[np.newaxis, :, :]], axis=0)
        p_stack = os.path.join(out_dir, 'Phase4_Input_Stack.tif')
        meta.update(count=stack.shape[0], dtype='float32')
        with rasterio.open(p_stack, 'w', **meta) as dst: dst.write(stack.astype('float32'))

        # 4. EXTRACT NEW FEATURES
        X_final, y_final, _ = self.extract_values(p_stack, train_lyr, train_fld, col_mode, log_path, feedback)
        y_final = y_final.flatten()
        
        # Split Train/Val locally for refinement
        X_train, X_val, y_train, y_val = train_test_split(X_final, y_final, test_size=0.2, random_state=42)
        
        # 5. TRAIN REFINEMENT MODELS
        best_rmse = float('inf'); best_model = None; best_algo_name = ""
        # --- NEW VARS FOR METRICS ---
        best_r2 = 0.0; best_wmape = 0.0
        
        for idx in sel_idx:
            name, model_inst, default_params = self.get_model_and_params(idx, opt_idx)
            
            # --- OVERRIDE WITH CUSTOM PARAMS ---
            if name in self.custom_params and self.custom_params[name]:
                if opt_idx == 2 and SKOPT_AVAILABLE:
                    params = self.convert_to_bayes(self.custom_params[name])
                else:
                    params = self.custom_params[name]
            else:
                params = default_params
            # -----------------------------------
            
            try:
                curr_model = model_inst
                if params and n_iter > 0:
                    search = None
                    current_opt_idx = opt_idx
                    if name == 'MLP': current_opt_idx = 0 

                    if current_opt_idx == 0: search = RandomizedSearchCV(model_inst, params, n_iter=n_iter, cv=3, n_jobs=1)
                    elif current_opt_idx == 1: search = GridSearchCV(model_inst, params, cv=3, n_jobs=1)
                    elif current_opt_idx == 2: search = BayesSearchCV(model_inst, params, n_iter=n_iter, cv=3, n_jobs=1) if SKOPT_AVAILABLE else RandomizedSearchCV(model_inst, params, n_iter=n_iter, cv=3, n_jobs=1)
                    
                    if search: search.fit(X_train, y_train); curr_model = search.best_estimator_
                    else: curr_model.fit(X_train, y_train)
                else:
                    curr_model.fit(X_train, y_train)
                
                y_pred = curr_model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                
                self.append_log(f"       > {name}: RMSE={rmse:.3f}m", log_path, feedback)
                
                if rmse < best_rmse: 
                    best_rmse = rmse; best_model = curr_model; best_algo_name = name
                    # --- CALCULATE METRICS FOR WINNER ---
                    best_r2 = r2_score(y_val, y_pred)
                    sum_abs_diff = np.sum(np.abs(y_val - y_pred))
                    sum_abs_true = np.sum(np.abs(y_val))
                    best_wmape = (sum_abs_diff / sum_abs_true) * 100 if sum_abs_true != 0 else 0
                
            except Exception as e:
                self.append_log(f"Error in {name}: {str(e)}", log_path, feedback)

        if best_model is None: raise QgsProcessingException("All refinement models failed.")
        
        # 6. PREDICT FINAL MAP
        self.append_log(f"   Predicting Final Map using {best_algo_name}...", log_path, feedback)
        X_map = stack[:, water_indices[0], water_indices[1]].T
        X_map = np.nan_to_num(X_map, nan=0.0)
        z_out = best_model.predict(X_map)
        
        final_map = np.full((h, w), -9999.0, dtype='float32')
        final_map[water_indices] = z_out
        
        if med_size > 0 and scipy_is_available:
            valid = final_map != -9999; temp = final_map.copy(); temp[~valid] = np.nan
            filtered = median_filter(temp, size=med_size); final_map[valid] = filtered[valid]
        
        p_final = os.path.join(out_dir, '4-Refined_Model.tif')
        meta.update(count=1, nodata=-9999.0)
        with rasterio.open(p_final, 'w', **meta) as dst: dst.write(final_map, 1)
        
        # --- RETURN METRICS ---
        return {
            'OUTPUT_FINAL': p_final,
            'BEST_R2': best_r2,
            'BEST_RMSE': best_rmse,
            'BEST_WMAPE': best_wmape
        }

    def extract_values(self, ras, vec, fld, mode, logger_file, fb):
        ds = rasterio.open(ras); d = ds.read(); h, w = ds.height, ds.width
        tr = QgsCoordinateTransform(vec.sourceCrs(), QgsRasterLayer(ras).crs(), QgsProject.instance())
        X_out, y_out, c_out = [], [], []

        for f in vec.getFeatures():
            g = f.geometry(); g.transform(tr); pt = g.asPoint()
            r, c = ds.index(pt.x(), pt.y())
            if 0 <= r < h and 0 <= c < w:
                val = d[:, r, c]
                if np.all(np.isfinite(val)) and not np.any(val == -9999):
                    X_out.append(val); y_out.append(f[fld]); c_out.append([r, c])
        
        return np.array(X_out), np.array(y_out), np.array(c_out)

    def get_model_and_params(self, index, opt_idx=0):
        is_bayes = (opt_idx == 2 and SKOPT_AVAILABLE)
        
        if index == 0: return 'Linear Regression', LinearRegression(), {}
        if index == 1: return 'Random Forest', RandomForestRegressor(n_jobs=1), ({'n_estimators': Integer(100, 500)} if is_bayes else {'n_estimators': [100, 300]})
        if index == 2: return 'Gradient Boosting', GradientBoostingRegressor(), ({'learning_rate': Real(0.01, 0.2)} if is_bayes else {'learning_rate': [0.05, 0.1]})
        if index == 3: return 'Extra Trees', ExtraTreesRegressor(n_jobs=1), ({'n_estimators': Integer(100, 500)} if is_bayes else {'n_estimators': [100, 300]})
        if index == 4: return 'Ridge', Ridge(), ({'alpha': Real(0.1, 10.0)} if is_bayes else {'alpha': [0.1, 1.0]})
        if index == 5: return 'Lasso', Lasso(), ({'alpha': Real(0.01, 1.0)} if is_bayes else {'alpha': [0.01, 0.1]})
        if index == 6: return 'ElasticNet', ElasticNet(), ({'l1_ratio': Real(0.1, 0.9)} if is_bayes else {'l1_ratio': [0.5]})
        if index == 7: return 'KNN', KNeighborsRegressor(n_jobs=1), ({'n_neighbors': Integer(3, 15)} if is_bayes else {'n_neighbors': [5, 10]})
        if index == 8: return 'Decision Tree', DecisionTreeRegressor(), ({'max_depth': Integer(5, 20)} if is_bayes else {'max_depth': [5, 10]})
        if index == 9: return 'MLP', MLPRegressor(max_iter=500), {'hidden_layer_sizes': [(100,), (100, 50)], 'activation': ['relu', 'tanh'], 'learning_rate_init': [0.001, 0.01]}
        if index == 10: return 'SVR', SVR(cache_size=1000, max_iter=20000), ({'C': Real(1.0, 100.0)} if is_bayes else {'C': [10, 100], 'kernel':['rbf']})
            
        return 'Unknown', LinearRegression(), {}