# SDB_03_Initial_Modeling.py
# ---------------------------------------------------------------------------
# MODULE 03: INITIAL MODELING & BENCHMARKING (FIXED WEIGHTS & ABSOLUTE WMAPE)
# Fixes:
# - wMAPE is now strictly absolute (Sum(|Diff|) / Sum(|Actual|)).
# - Lower wMAPE improves the score (like RMSE).
# ---------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import rasterio
import joblib
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from qgis.core import (
    QgsProcessing, QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer, QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField, QgsProcessingParameterFolderDestination,
    QgsProcessingParameterEnum, QgsProcessingException,
    QgsCoordinateTransform, QgsProject, QgsProcessingParameterNumber,
    QgsRasterLayer
)

# ML Imports
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

class SDBModule03(QgsProcessingAlgorithm):
    # --- INPUTS ---
    INPUT_STACK = 'INPUT_STACK'
    INPUT_MASK = 'INPUT_MASK'
    INPUT_POINTS = 'INPUT_POINTS'
    FIELD_DEPTH = 'FIELD_DEPTH'
    FIELD_WEIGHT = 'FIELD_WEIGHT'
    
    SELECTED_ALGOS = 'SELECTED_ALGOS'
    N_ITERATIONS = 'N_ITERATIONS'
    MEDIAN_SIZE = 'MEDIAN_SIZE'
    
    OUTPUT_FOLDER = 'OUTPUT_FOLDER'

    MODEL_LIST = [
        'Linear Regression', 'Random Forest', 'Gradient Boosting', 'Extra Trees',
        'Ridge', 'Lasso', 'ElasticNet', 'KNN', 'Decision Tree', 'MLP (Neural Net)', 'SVR'
    ]

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_STACK, 'Input Feature Stack'))
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_MASK, 'Input Water Mask'))
        
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_POINTS, 'Cleaned Training Points'))
        self.addParameter(QgsProcessingParameterField(self.FIELD_DEPTH, 'Depth Field', parentLayerParameterName=self.INPUT_POINTS, type=QgsProcessingParameterField.Numeric))
        self.addParameter(QgsProcessingParameterField(self.FIELD_WEIGHT, 'Confidence/Weight Field (Optional)', parentLayerParameterName=self.INPUT_POINTS, type=QgsProcessingParameterField.Numeric, optional=True))

        self.addParameter(QgsProcessingParameterEnum(self.SELECTED_ALGOS, 'Select Algorithms', 
                                                     options=self.MODEL_LIST, allowMultiple=True, defaultValue=[0, 1, 2, 3]))
        
        self.addParameter(QgsProcessingParameterNumber(self.N_ITERATIONS, 'Optimization Iterations', 
                                                       type=QgsProcessingParameterNumber.Integer, defaultValue=10, minValue=0))
        self.addParameter(QgsProcessingParameterNumber(self.MEDIAN_SIZE, 'Median Filter Size', 
                                                       type=QgsProcessingParameterNumber.Integer, defaultValue=3, minValue=1))
        
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, 'Output Folder'))

    def name(self): return 'sdb_03_initial_modeling'
    def displayName(self): return '3. SDB Module 03: Initial Modeling (Smart Weights)'
    def group(self): return 'SDB Research Tools'
    def groupId(self): return 'sdb_tools'
    def createInstance(self): return SDBModule03()

    def processAlgorithm(self, parameters, context, feedback):
        out_dir = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        os.makedirs(out_dir, exist_ok=True)
        
        stack_path = self.parameterAsRasterLayer(parameters, self.INPUT_STACK, context).source()
        mask_path = self.parameterAsRasterLayer(parameters, self.INPUT_MASK, context).source()
        points_layer = self.parameterAsVectorLayer(parameters, self.INPUT_POINTS, context)
        depth_fld = self.parameterAsString(parameters, self.FIELD_DEPTH, context)
        weight_fld = self.parameterAsString(parameters, self.FIELD_WEIGHT, context)
        if weight_fld == '' or weight_fld is None: weight_fld = None
        
        n_iter = self.parameterAsInt(parameters, self.N_ITERATIONS, context)
        med_size = self.parameterAsInt(parameters, self.MEDIAN_SIZE, context)
        sel_idx = self.parameterAsEnums(parameters, self.SELECTED_ALGOS, context)
        
        if not sel_idx: raise QgsProcessingException("Select at least one algorithm.")

        bench_dir = os.path.join(out_dir, '3_Benchmarks')
        os.makedirs(bench_dir, exist_ok=True)

        feedback.pushInfo("\n>>> MODULE 03: STARTING INITIAL MODELING...")

        # 1. Extract
        feedback.pushInfo("   [1/5] Extracting Training Data...")
        X, y, weights, _ = self.extract_samples(stack_path, points_layer, depth_fld, weight_fld, feedback)
        
        if len(y) < 10: raise QgsProcessingException("Critically low training points (<10). Check inputs.")
        if weights is not None: feedback.pushInfo(f"   Using Sample Weights (Or Default=1s) from: {weight_fld if weight_fld else 'Auto-Generated'}")

        # 2. Benchmark
        feedback.pushInfo(f"   [2/5] Benchmarking {len(sel_idx)} Algorithms...")
        results_df, best_algo_data = self.run_benchmarking(
            X, y, weights, sel_idx, n_iter, bench_dir, feedback
        )

        # 3. Save Report
        feedback.pushInfo("   [3/5] Saving Benchmark Reports...")
        results_df.to_csv(os.path.join(out_dir, '3_All_Algorithms_Benchmark.csv'), index=False)
        
        winner_name = best_algo_data['name']
        winner_model = best_algo_data['model']
        feedback.pushInfo(f"\n   >>> WINNER: {winner_name} (Score={best_algo_data['score']:.4f})")

        # 4. Save Model
        feedback.pushInfo("   [4/5] Saving Best Model...")
        model_path = os.path.join(out_dir, '3_Best_Global_Model.pkl')
        joblib.dump(winner_model, model_path)

        # 5. Predict
        feedback.pushInfo("   [5/5] Generating Initial Depth Map...")
        p_map = os.path.join(out_dir, '3_Initial_Global_Depth.tif')
        self.predict_map(winner_model, stack_path, mask_path, p_map, med_size)
        
        QgsProject.instance().addMapLayer(QgsRasterLayer(p_map, f"Initial Global SDB ({winner_name})"))

        return {'OUTPUT_DEPTH_MAP': p_map, 'OUTPUT_MODEL_PKL': model_path}

    # =========================================================================
    # CORE LOGIC
    # =========================================================================

    def extract_samples(self, ras_path, vec_layer, d_fld, w_fld, fb):
        rlayer = QgsRasterLayer(ras_path)
        tr = QgsCoordinateTransform(vec_layer.sourceCrs(), rlayer.crs(), QgsProject.instance())
        with rasterio.open(ras_path) as src:
            d = src.read(); h, w = src.height, src.width
            X_list, y_list, w_list, coords = [], [], [], []
            total = vec_layer.featureCount()
            extracted = 0
            for f in vec_layer.getFeatures():
                geom = f.geometry(); geom.transform(tr); pt = geom.asPoint()
                r, c = src.index(pt.x(), pt.y())
                if 0 <= r < h and 0 <= c < w:
                    if d[0, r, c] != -9999:
                        val_vector = d[:, r, c]
                        if np.all(np.isfinite(val_vector)):
                            X_list.append(val_vector); y_list.append(f[d_fld]); coords.append([r, c])
                            
                            # <<< TWEAK 1: Handle Missing Confidence >>>
                            # If w_fld is provided, use it. If not, append 1.0
                            if w_fld: 
                                w_list.append(f[w_fld])
                            else:
                                w_list.append(1.0)
                            
                            extracted += 1
            fb.pushInfo(f"      Extracted {extracted}/{total} samples.")
            # Always return w_list as numpy array (now contains 1s if null)
            return np.array(X_list), np.array(y_list), np.array(w_list), coords

    def run_benchmarking(self, X, y, weights, indices, n_iter, out_dir, fb):
        X = np.nan_to_num(X, nan=0.0)
        
        if weights is not None:
            X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(X, y, weights, test_size=0.2, random_state=42)
        else:
            X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            w_tr = None

        results = []
        for idx in indices:
            name, base_model, params = self.get_model_and_params(idx)
            algo_dir = os.path.join(out_dir, name.replace(" ", "_"))
            os.makedirs(algo_dir, exist_ok=True)
            
            # --- Check if Algorithm supports weights ---
            supports_weights = True
            if "MLP" in name or "KNN" in name:
                supports_weights = False
            
            try:
                model = base_model
                if n_iter > 0 and params:
                    search = RandomizedSearchCV(estimator=base_model, param_distributions=params, 
                                                n_iter=n_iter, cv=3, n_jobs=1, scoring='r2', random_state=42)
                    if w_tr is not None and supports_weights:
                        search.fit(X_tr, y_tr, sample_weight=w_tr)
                    else:
                        search.fit(X_tr, y_tr)
                    
                    model = search.best_estimator_
                    best_params = str(search.best_params_)
                else:
                    if w_tr is not None and supports_weights:
                        model.fit(X_tr, y_tr, sample_weight=w_tr)
                    else:
                        model.fit(X_tr, y_tr)
                    best_params = "Default"

                y_p = model.predict(X_val)
                r2 = r2_score(y_val, y_p)
                rmse = np.sqrt(mean_squared_error(y_val, y_p))
                mae = mean_absolute_error(y_val, y_p)
                
                # Weighted MAPE
                safe_y = y_val.copy(); safe_y[safe_y==0] = 0.001
                pct_err = np.abs((y_val - y_p) / safe_y) * 100
                
                if w_val is not None:
                     w_norm = w_val / np.sum(w_val)
                     mape = np.sum(pct_err * w_norm)
                else:
                     mape = np.mean(pct_err)

                # --- MODIFIED: STRICTLY ABSOLUTE wMAPE ---
                # Formula: Sum(|Actual - Pred|) / Sum(|Actual|)
                sum_abs_diff = np.sum(np.abs(y_val - y_p))
                sum_abs_true = np.sum(np.abs(y_val)) # ABS added to denominator
                
                wmape_score = (sum_abs_diff / sum_abs_true) * 100 if sum_abs_true != 0 else 0

                self.save_algo_artifacts(y_val, y_p, pct_err, name, algo_dir, r2, rmse, mape, best_params)
                
                results.append({
                    'Algorithm': name, 'Model': model, 'Params': best_params,
                    'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'wMAPE_Score': wmape_score
                })
                fb.pushInfo(f"      > {name}: R2={r2:.3f}, RMSE={rmse:.2f}m, wMAPE={wmape_score:.2f}%")

            except Exception as e:
                fb.pushWarning(f"      ! Failed {name}: {e}")

        if not results: raise QgsProcessingException("All algorithms failed training.")
        
        df = pd.DataFrame(results)
        
        # Ranking Logic
        # 1. Normalize RMSE (Lower is Better -> Higher Score)
        min_rmse, max_rmse = df['RMSE'].min(), df['RMSE'].max()
        df['n_rmse'] = 1 - ((df['RMSE'] - min_rmse) / (max_rmse - min_rmse + 1e-6))
        
        # 2. Normalize wMAPE (Lower is Better -> Higher Score)
        min_wmape, max_wmape = df['wMAPE_Score'].min(), df['wMAPE_Score'].max()
        df['n_wmape'] = 1 - ((df['wMAPE_Score'] - min_wmape) / (max_wmape - min_wmape + 1e-6))
        
        # 3. Calculate Final Score (60% R2 + 20% RMSE + 20% wMAPE)
        df['score'] = (0.6 * df['R2'].clip(lower=0)) + (0.2 * df['n_rmse']) + (0.2 * df['n_wmape'])
        
        winner_row = df.loc[df['score'].idxmax()]
        
        # Retrain Winner
        final_model = winner_row['Model']
        win_name = winner_row['Algorithm']
        supports_weights = "MLP" not in win_name and "KNN" not in win_name
        
        if weights is not None and supports_weights:
            final_model.fit(X, y, sample_weight=weights)
        else:
            final_model.fit(X, y)
        
        return df.drop(columns=['Model', 'n_rmse', 'n_wmape']), {'name': win_name, 'model': final_model, 'score': winner_row['score']}

    def predict_map(self, model, stack_path, mask_path, out_path, med_size):
        with rasterio.open(mask_path) as m: mask_arr = m.read(1).flatten()
        with rasterio.open(stack_path) as s:
            d = s.read(); h, w = s.height, s.width
            d_flat = d.reshape(d.shape[0], -1).T
            water_idx = np.where(mask_arr == 1)[0]
            X_pixels = np.nan_to_num(d_flat[water_idx], nan=0.0, posinf=0.0, neginf=0.0)
            preds = model.predict(X_pixels)
            out_img = np.full(h*w, -9999.0, dtype='float32')
            out_img[water_idx] = preds
            out_img = out_img.reshape(h, w)
            if scipy_is_available:
                valid = (out_img != -9999.0)
                temp = out_img.copy(); temp[~valid] = 0
                filt = median_filter(temp, size=med_size); filt[~valid] = -9999.0
                out_img = filt
            
            prof = s.profile; prof.update(count=1, dtype='float32', nodata=-9999.0)
            with rasterio.open(out_path, 'w', **prof) as dst: dst.write(out_img, 1)

    def save_algo_artifacts(self, y_t, y_p, pct, name, folder, r2, rmse, mape, params):
        with open(os.path.join(folder, 'Results.txt'), 'w') as f:
            f.write(f"Algo: {name}\nR2: {r2:.4f}\nRMSE: {rmse:.4f}\nMAPE: {mape:.2f}%\nParams: {params}")
        pd.DataFrame({'Obs': y_t, 'Pred': y_p, 'MAPE': pct}).to_csv(os.path.join(folder, 'Validation_Samples.csv'), index=False)
        plt.figure(figsize=(5,5))
        plt.scatter(y_t, y_p, c=pct, cmap='viridis_r', alpha=0.7)
        plt.plot([min(y_t), max(y_t)], [min(y_t), max(y_t)], 'r--')
        plt.title(f"{name} (R2={r2:.2f})"); plt.savefig(os.path.join(folder, 'Scatter.png')); plt.close()

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