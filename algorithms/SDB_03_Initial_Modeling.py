import os
import numpy as np
import pandas as pd
import rasterio
import joblib
import warnings
import ast # Essential for parsing params from Master Workflow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from PyQt5.QtCore import QVariant 
from qgis.core import (
    QgsProcessing, QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer, QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField, QgsProcessingParameterFolderDestination,
    QgsProcessingParameterEnum, QgsProcessingException,
    QgsCoordinateTransform, QgsProject, QgsProcessingParameterNumber,
    QgsRasterLayer, QgsVectorFileWriter, QgsFeature, QgsGeometry, 
    QgsPointXY, QgsField, QgsFields, QgsWkbTypes, QgsProcessingParameterFile, QgsProcessingParameterString
)

# ML Imports
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# Try importing BayesSearchCV & Spaces
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Categorical, Integer
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    
try:
    from scipy.ndimage import median_filter
    scipy_is_available = True
except ImportError:
    scipy_is_available = False

class SDBModule03(QgsProcessingAlgorithm):
    # --- INPUTS ---
    INPUT_STACK = 'INPUT_STACK'
    INPUT_MASK = 'INPUT_MASK'
    INPUT_POINTS = 'INPUT_POINTS'
    
    FIELD_DEPTH = 'FIELD_DEPTH'
    FIELD_WEIGHT = 'FIELD_WEIGHT'
    
    # --- MODELING CONFIG ---
    SELECTED_ALGOS = 'SELECTED_ALGOS'
    OPTIMIZER_METHOD = 'OPTIMIZER_METHOD'
    COLLISION_HANDLING = 'COLLISION_HANDLING'
    N_ITERATIONS = 'N_ITERATIONS'
    MEDIAN_SIZE = 'MEDIAN_SIZE'
    
    OUTPUT_FOLDER = 'OUTPUT_FOLDER'
    LOG_FILE = 'LOG_FILE'
    
    # --- HYPERPARAMETERS (Received as Strings from Master) ---
    PARAM_RF = 'PARAM_RF'
    PARAM_GB = 'PARAM_GB'
    PARAM_ET = 'PARAM_ET'
    PARAM_SVR = 'PARAM_SVR'
    PARAM_MLP = 'PARAM_MLP'

    MODEL_LIST = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'Extra Trees', 'Ridge', 'Lasso', 'ElasticNet', 'KNN', 'Decision Tree', 'MLP (Neural Net)', 'SVR']
    OPTIMIZER_LIST = ['Random Search', 'Grid Search', 'Bayesian Search']
    COLLISION_LIST = [
        'Keep All Points(Weighted average)', 
        'Highest Confidence',
        'Closest to Pixel Center',
        'Hybrid (Center-Weighted Half-Pixel)',
        'Strict Center Points Only (Ignore Edges)'
    ]

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_STACK, 'Input Feature Stack'))
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_MASK, 'Input Water Mask'))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_POINTS, 'Cleaned Training Points'))
        
        self.addParameter(QgsProcessingParameterField(self.FIELD_DEPTH, 'Depth Field', parentLayerParameterName=self.INPUT_POINTS, type=QgsProcessingParameterField.Numeric))
        self.addParameter(QgsProcessingParameterField(self.FIELD_WEIGHT, 'Weight Field', parentLayerParameterName=self.INPUT_POINTS, type=QgsProcessingParameterField.Numeric, optional=True))
        
        self.addParameter(QgsProcessingParameterEnum(self.SELECTED_ALGOS, 'Select Algorithms', options=self.MODEL_LIST, allowMultiple=True, defaultValue=[0, 1, 2, 3]))
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

    def name(self): return 'sdb_03_initial_modeling'
    def displayName(self): return '3. SDB Module 03: Global Auto-ML'
    def group(self): return 'SDB Research Tools'
    def groupId(self): return 'sdb_tools'
    def createInstance(self): return SDBModule03()

    def shortHelpString(self):
        return """
        <div style="font-family: Arial, sans-serif; line-height: 1.2;">
            <h2 style="margin-bottom: 5px;">🤖 <span style="color: #2E86C1;">SDB Phase 03</span>: Global Auto-ML</h2>
            <p style="margin-top: 0; margin-bottom: 10px;">Trains and optimizes multiple Machine Learning models to predict bathymetry from satellite features.</p>

            <b style="display: block; margin-bottom: 2px;">⚡ Key Features</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li><b>11 Algorithms:</b> Ranging from Linear Regression to Deep Learning (MLP) and SVR.</li>
                <li><b>Smart Optimization:</b> Supports <i>Random Search</i>, <i>Grid Search</i>, and <i>Bayesian Optimization</i>.</li>
                <li><b>Custom Tuning:</b> Accepts Python-dictionary style strings for full hyperparameter control.</li>
            </ul>

            <b style="display: block; margin-bottom: 2px;">📍 Spatial Handling</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li><b>Collision Handling:</b> Intelligent logic to handle multiple depth points falling into a single pixel (e.g., Average, Nearest to Center, or Highest Confidence).</li>
            </ul>

            <b style="display: block; margin-bottom: 2px;">📂 Output</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li>Comparative CSV Benchmark.</li>
                <li>Best Model (.pkl) & Prediction Map.</li>
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
        # Parses string from Master workflow: "'a': [1,2]" -> {'a': [1,2]}
        if not param_str or param_str.strip() == "": return {}
        try:
            return ast.literal_eval("{" + param_str + "}")
        except:
            return {}

    def convert_to_bayes(self, params_dict):
        # Convert Lists to Scikit-Optimize Spaces
        bayes_params = {}
        for k, v in params_dict.items():
            if isinstance(v, list) and len(v) > 0:
                if all(isinstance(x, int) for x in v):
                    bayes_params[k] = Integer(min(v), max(v))
                elif all(isinstance(x, (int, float)) for x in v):
                    bayes_params[k] = Real(min(v), max(v))
                else:
                    bayes_params[k] = Categorical(v)
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

        stack_path = self.parameterAsRasterLayer(parameters, self.INPUT_STACK, context).source()
        mask_path = self.parameterAsRasterLayer(parameters, self.INPUT_MASK, context).source()
        points_layer = self.parameterAsVectorLayer(parameters, self.INPUT_POINTS, context)
        depth_fld = self.parameterAsString(parameters, self.FIELD_DEPTH, context)
        weight_fld = self.parameterAsString(parameters, self.FIELD_WEIGHT, context)
        if not weight_fld: weight_fld = None
        
        n_iter = self.parameterAsInt(parameters, self.N_ITERATIONS, context)
        med_size = self.parameterAsInt(parameters, self.MEDIAN_SIZE, context)
        sel_idx = self.parameterAsEnums(parameters, self.SELECTED_ALGOS, context)
        opt_idx = self.parameterAsInt(parameters, self.OPTIMIZER_METHOD, context)
        col_mode = self.parameterAsInt(parameters, self.COLLISION_HANDLING, context)
        
        self.append_log(f"MODULE 03 START: Opt={self.OPTIMIZER_LIST[opt_idx]} | Coll={self.COLLISION_LIST[col_mode]}", log_path, feedback)

        # 1. EXTRACT SAMPLES
        X, y, weights, coords = self.extract_samples(stack_path, points_layer, depth_fld, weight_fld, col_mode)
        if len(y) < 10: raise QgsProcessingException("Critically low training points (<10).")
        
        self.append_log(f"   Extracted {len(y)} training pixels.", log_path, feedback)
        
        # Save Used Points for Verification
        self.save_training_points(coords, y, weights, X, stack_path, points_layer.sourceCrs(), out_dir)

        # 2. RUN BENCHMARK
        results_df, best_algo_data = self.run_benchmarking(X, y, weights, sel_idx, n_iter, out_dir, feedback, opt_idx, log_path)
        results_df.to_csv(os.path.join(out_dir, '3_All_Algorithms_Benchmark.csv'), index=False)
        
        win_name = best_algo_data['name']
        self.append_log(f"\n   >>> WINNER: {win_name} (Score={best_algo_data['score']:.4f})", log_path, feedback)
        
        # 3. SAVE & PREDICT
        joblib.dump(best_algo_data['model'], os.path.join(out_dir, '3_Best_Global_Model.pkl'))
        p_map = os.path.join(out_dir, '3_Initial_Global_Depth.tif')
        
        self.append_log("   Generating prediction map...", log_path, feedback)
        self.predict_map(best_algo_data['model'], stack_path, mask_path, p_map, med_size)
        
        QgsProject.instance().addMapLayer(QgsRasterLayer(p_map, f"3_Initial_Global_Depth ({win_name})"))
        
        # --- RETURN DICT WITH METRICS FOR MASTER ---
        return {
            'OUTPUT_DEPTH_MAP': p_map, 
            'OUTPUT_MODEL_PKL': os.path.join(out_dir, '3_Best_Global_Model.pkl'),
            'BEST_R2': best_algo_data['r2'],
            'BEST_RMSE': best_algo_data['rmse'],
            'BEST_WMAPE': best_algo_data['wmape']
        }

    def extract_samples(self, ras_path, vec_layer, d_fld, w_fld, mode):
        rlayer = QgsRasterLayer(ras_path)
        tr = QgsCoordinateTransform(vec_layer.sourceCrs(), rlayer.crs(), QgsProject.instance())
        X_out, y_out, w_out, c_out = [], [], [], []
        
        with rasterio.open(ras_path) as src:
            d = src.read(); h, w = src.height, src.width
            rst_transform = src.transform 
            pixel_size = abs(src.res[0]) 

            if mode == 0: # Keep All
                for f in vec_layer.getFeatures():
                    geom = f.geometry(); geom.transform(tr); pt = geom.asPoint()
                    r, c = src.index(pt.x(), pt.y())
                    if 0 <= r < h and 0 <= c < w:
                        val = d[:, r, c]
                        if np.all(np.isfinite(val)) and not np.any(val == -9999):
                            X_out.append(val); y_out.append(f[d_fld]); c_out.append([r, c])
                            w_out.append(f[w_fld] if w_fld else 1.0)
                return np.array(X_out), np.array(y_out), np.array(w_out), c_out

            pixel_registry = {} 
            for f in vec_layer.getFeatures():
                geom = f.geometry(); geom.transform(tr); pt = geom.asPoint()
                r, c = src.index(pt.x(), pt.y())
                if 0 <= r < h and 0 <= c < w:
                    pixel_registry.setdefault((r, c), []).append({'d': f[d_fld], 'w': f[w_fld] if w_fld else 1.0, 'pt': pt})

            for (r, c), items in pixel_registry.items():
                val = d[:, r, c]
                if not np.all(np.isfinite(val)) or np.any(val == -9999): continue

                final_depth, final_weight = 0.0, 1.0
                
                if mode == 1: # Highest Conf
                    best = sorted(items, key=lambda x: x['w'], reverse=True)[0]
                    final_depth, final_weight = best['d'], best['w']
                elif mode == 2: # Closest
                    cx, cy = rasterio.transform.xy(rst_transform, r, c, offset='center')
                    best = min(items, key=lambda x: (x['pt'].x()-cx)**2 + (x['pt'].y()-cy)**2)
                    final_depth, final_weight = best['d'], best['w']
                elif mode == 3: # Hybrid
                    cx, cy = rasterio.transform.xy(rst_transform, r, c, offset='center')
                    close = [i for i in items if ((i['pt'].x()-cx)**2 + (i['pt'].y()-cy)**2)**0.5 <= pixel_size/2]
                    target = close if close else items
                    vals = np.array([i['d'] for i in target]); ws = np.array([i['w'] for i in target])
                    final_depth = np.average(vals, weights=ws) if np.sum(ws)>0 else np.mean(vals)
                    final_weight = np.mean(ws)
                elif mode == 4: # Strict Center
                    cx, cy = rasterio.transform.xy(rst_transform, r, c, offset='center')
                    close = [i for i in items if ((i['pt'].x()-cx)**2 + (i['pt'].y()-cy)**2)**0.5 <= pixel_size/2]
                    if not close: continue
                    vals = np.array([i['d'] for i in close]); ws = np.array([i['w'] for i in close])
                    final_depth = np.average(vals, weights=ws) if np.sum(ws)>0 else np.mean(vals)
                    final_weight = np.mean(ws)

                X_out.append(val); y_out.append(final_depth); w_out.append(final_weight); c_out.append([r, c])

        return np.array(X_out), np.array(y_out), np.array(w_out), c_out

    def save_training_points(self, coords, y_list, w_list, X_list, ras_path, crs, out_dir):
        if X_list.ndim == 1: X_list = X_list.reshape(-1, 1)
        fields = QgsFields()
        fields.append(QgsField("Depth_Used", QVariant.Double))
        fields.append(QgsField("Weight_Used", QVariant.Double))
        fields.append(QgsField("Row_Idx", QVariant.Int))
        fields.append(QgsField("Col_Idx", QVariant.Int))
        for b in range(X_list.shape[1]): fields.append(QgsField(f"Band_{b+1}", QVariant.Double))

        out_file = os.path.join(out_dir, "3_Actual_Model_Input_Points.gpkg")
        writer = QgsVectorFileWriter(out_file, "UTF-8", fields, QgsWkbTypes.Point, crs, "GPKG")
        
        with rasterio.open(ras_path) as src:
            transform = src.transform
            for i, (r, c) in enumerate(coords):
                x, y = rasterio.transform.xy(transform, r, c, offset='center')
                fet = QgsFeature()
                fet.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
                attrs = [float(y_list[i]), float(w_list[i]), int(r), int(c)]
                attrs.extend([float(v) for v in X_list[i]])
                fet.setAttributes(attrs)
                writer.addFeature(fet)
        del writer

    def run_benchmarking(self, X, y, weights, indices, n_iter, out_dir, fb, opt_idx, log_path):
        X = np.nan_to_num(X, nan=0.0)
        X_tr, X_val, y_tr, y_val, w_tr, _ = train_test_split(X, y, weights, test_size=0.2, random_state=42)
        
        results = []
        for idx in indices:
            name, base_model, default_params = self.get_model_and_params(idx, opt_idx)
            
            # --- OVERRIDE WITH CUSTOM PARAMS IF PRESENT ---
            if name in self.custom_params and self.custom_params[name]:
                if opt_idx == 2 and SKOPT_AVAILABLE:
                    params = self.convert_to_bayes(self.custom_params[name])
                else:
                    params = self.custom_params[name]
            else:
                params = default_params
            # ----------------------------------------------

            algo_dir = os.path.join(out_dir, name.replace(" ", "_"))
            os.makedirs(algo_dir, exist_ok=True)
            
            try:
                model = base_model
                if params and n_iter > 0:
                    search = None
                    current_opt_idx = opt_idx
                    if name == 'MLP': current_opt_idx = 0 

                    if current_opt_idx == 0: search = RandomizedSearchCV(base_model, params, n_iter=n_iter, cv=3, n_jobs=1)
                    elif current_opt_idx == 1: search = GridSearchCV(base_model, params, cv=3, n_jobs=1)
                    elif current_opt_idx == 2: search = BayesSearchCV(base_model, params, n_iter=n_iter, cv=3, n_jobs=1) if SKOPT_AVAILABLE else RandomizedSearchCV(base_model, params, n_iter=n_iter, cv=3, n_jobs=1)
                    
                    if search: 
                        search.fit(X_tr, y_tr, sample_weight=w_tr) if "MLP" not in name and "KNN" not in name else search.fit(X_tr, y_tr)
                        model = search.best_estimator_
                        params_str = str(search.best_params_)
                    else:
                        model.fit(X_tr, y_tr, sample_weight=w_tr) if "MLP" not in name and "KNN" not in name else model.fit(X_tr, y_tr)
                        params_str = "Default"
                else:
                    model.fit(X_tr, y_tr, sample_weight=w_tr) if "MLP" not in name and "KNN" not in name else model.fit(X_tr, y_tr)
                    params_str = "Default"

                y_p = model.predict(X_val)
                r2 = r2_score(y_val, y_p)
                rmse = np.sqrt(mean_squared_error(y_val, y_p))
                sum_abs_diff = np.sum(np.abs(y_val - y_p))
                sum_abs_true = np.sum(np.abs(y_val))
                wmape = (sum_abs_diff / sum_abs_true) * 100 if sum_abs_true != 0 else 0
                
                self.save_algo_artifacts(y_val, y_p, np.abs(y_val-y_p), name, algo_dir, r2, rmse, wmape, params_str)
                results.append({'Algorithm': name, 'Model': model, 'R2': r2, 'RMSE': rmse, 'wMAPE': wmape})
                
                self.append_log(f"      > {name}: R2={r2:.3f}, RMSE={rmse:.2f}m", log_path, fb)

            except Exception as e:
                self.append_log(f"      ! Failed {name}: {e}", log_path, fb)

        if not results: raise QgsProcessingException("All selected algorithms failed.")
        df = pd.DataFrame(results)
        df['score'] = (0.6 * df['R2'].clip(lower=0)) + (0.4 * (1 - (df['RMSE']/df['RMSE'].max())))
        winner = df.loc[df['score'].idxmax()]
        
        final_model = winner['Model']
        final_model.fit(X, y, sample_weight=weights) if "MLP" not in winner['Algorithm'] and "KNN" not in winner['Algorithm'] else final_model.fit(X, y)
        
        # --- PASS WINNER METRICS ---
        return df, {
            'name': winner['Algorithm'], 
            'model': final_model, 
            'score': winner['score'],
            'r2': winner['R2'],
            'rmse': winner['RMSE'],
            'wmape': winner['wMAPE']
        }

    def predict_map(self, model, stack_path, mask_path, out_path, med_size):
        with rasterio.open(mask_path) as m: mask_arr = m.read(1).flatten()
        with rasterio.open(stack_path) as s:
            d = s.read(); h, w = s.height, s.width
            d_flat = d.reshape(d.shape[0], -1).T
            water_idx = np.where(mask_arr == 1)[0]
            
            if len(water_idx) == 0: return # No water
            
            X_pixels = np.nan_to_num(d_flat[water_idx], nan=0.0)
            preds = model.predict(X_pixels)
            
            out_img = np.full(h*w, -9999.0, dtype='float32')
            out_img[water_idx] = preds
            out_img = out_img.reshape(h, w)
            
            if med_size > 0:
                valid = (out_img != -9999.0)
                if scipy_is_available:
                    temp = out_img.copy(); temp[~valid] = 0
                    filt = median_filter(temp, size=med_size); filt[~valid] = -9999.0
                    out_img = filt
            
            prof = s.profile; prof.update(count=1, dtype='float32', nodata=-9999.0)
            with rasterio.open(out_path, 'w', **prof) as dst: dst.write(out_img, 1)

    def save_algo_artifacts(self, y_t, y_p, pct, name, folder, r2, rmse, mape, params):
        with open(os.path.join(folder, 'Results.txt'), 'w') as f:
            f.write(f"Algo: {name}\nR2: {r2:.4f}\nRMSE: {rmse:.4f}\nwMAPE: {mape:.2f}%\nParams: {params}")

    def get_model_and_params(self, index, opt_idx=0):
        is_bayes = (opt_idx == 2 and SKOPT_AVAILABLE)
        
        if index == 0: return 'Linear Regression', LinearRegression(), {}
        
        if index == 1: 
            return 'Random Forest', RandomForestRegressor(n_jobs=1), ({'n_estimators': Integer(100, 500)} if is_bayes else {'n_estimators': [100, 300]})
        
        if index == 2: 
            return 'Gradient Boosting', GradientBoostingRegressor(), ({'learning_rate': Real(0.01, 0.2)} if is_bayes else {'learning_rate': [0.05, 0.1]})
        
        if index == 3: 
            return 'Extra Trees', ExtraTreesRegressor(n_jobs=1), ({'n_estimators': Integer(100, 500)} if is_bayes else {'n_estimators': [100, 300]})
        
        if index == 4: 
            return 'Ridge', Ridge(), ({'alpha': Real(0.1, 10.0)} if is_bayes else {'alpha': [0.1, 1.0]})
        
        if index == 5: 
            return 'Lasso', Lasso(), ({'alpha': Real(0.01, 1.0)} if is_bayes else {'alpha': [0.01, 0.1]})
        
        if index == 6: 
            return 'ElasticNet', ElasticNet(), ({'l1_ratio': Real(0.1, 0.9)} if is_bayes else {'l1_ratio': [0.5]})
        
        if index == 7: 
            return 'KNN', KNeighborsRegressor(n_jobs=1), ({'n_neighbors': Integer(3, 15)} if is_bayes else {'n_neighbors': [5, 10]})
        
        if index == 8: 
            return 'Decision Tree', DecisionTreeRegressor(), ({'max_depth': Integer(5, 20)} if is_bayes else {'max_depth': [5, 10]})
        
        if index == 9: 
            return 'MLP', MLPRegressor(max_iter=500), { 
                'hidden_layer_sizes': [(100,), (100, 50)],
                'activation': ['relu', 'tanh'],
                'learning_rate_init': [0.001, 0.01]
            }
            
        if index == 10: 
            return 'SVR', SVR(cache_size=1000, max_iter=20000), ({'C': Real(1.0, 100.0)} if is_bayes else {'C': [10, 100], 'kernel':['rbf']})
            
        return 'Unknown', LinearRegression(), {}