import ast
import os
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from qgis.PyQt.QtCore import QVariant
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsRasterLayer,
    QgsCoordinateTransform,
    QgsProject,
    QgsPointXY,
)
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge, HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class CustomEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimators, method="Average"):
        self.estimators = estimators  # List of (name, model)
        self.method = method
        self.fitted_estimators_ = []
        self.meta_learner_ = None

    def fit(self, X, y, sample_weight=None):
        self.fitted_estimators_ = []
        for name, est in self.estimators:
            model_to_fit = clone(est)
            fit_params = {}
            if sample_weight is not None:
                import inspect
                sig = inspect.signature(model_to_fit.fit)
                if "sample_weight" in sig.parameters:
                    fit_params["sample_weight"] = sample_weight
            model_to_fit.fit(X, y, **fit_params)
            self.fitted_estimators_.append(model_to_fit)

        if self.method == "Stacking":
            base_preds = np.column_stack([est.predict(X) for est in self.fitted_estimators_])
            self.meta_learner_ = Ridge(alpha=1.0)
            self.meta_learner_.fit(base_preds, y, sample_weight=sample_weight)
        
        return self

    def predict(self, X):
        if not self.fitted_estimators_:
            raise ValueError("Ensemble is not fitted yet.")
        preds = np.column_stack([est.predict(X) for est in self.fitted_estimators_])
        if self.method == "Average":
            return np.mean(preds, axis=1)
        elif self.method == "Median":
            return np.median(preds, axis=1)
        elif self.method == "Stacking":
            if self.meta_learner_ is None:
                raise ValueError("Meta learner is not fitted.")
            return self.meta_learner_.predict(preds)
        return np.mean(preds, axis=1)

from qgis.core import (
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,  # <--- تم إضافة هذا السطر لحل المشكلة
    QgsFields,
    QgsGeometry,
    QgsPointXY,
    QgsProcessingException,
    QgsProject,
    QgsRasterLayer,
    QgsVectorFileWriter,
    QgsWkbTypes,
)

from ...infrastructure.logging import append_log

try:
    from skopt import BayesSearchCV
    from skopt.space import Categorical, Integer, Real

    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

try:
    from scipy.ndimage import median_filter

    scipy_is_available = True
except ImportError:
    scipy_is_available = False

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

OPTIMIZER_LIST = ["Random Search", "Grid Search", "Bayesian Search"]


def parse_param_string(param_str):
    if not param_str:
        return {}
    try:
        return ast.literal_eval("{" + param_str + "}")
    except Exception:
        return {}


def convert_to_bayes(params_dict):
    bayes_params = {}
    for k, v in params_dict.items():
        if isinstance(v, list) and len(v) > 0:
            if len(v) == 1:
                bayes_params[k] = Categorical(v)
            elif all(isinstance(x, int) for x in v):
                if min(v) == max(v):
                    bayes_params[k] = Categorical(v)
                else:
                    bayes_params[k] = Integer(min(v), max(v))
            elif all(isinstance(x, (int, float)) for x in v):
                if min(v) == max(v):
                    bayes_params[k] = Categorical(v)
                else:
                    bayes_params[k] = Real(min(v), max(v))
            else:
                bayes_params[k] = Categorical(v)
        else:
            bayes_params[k] = Categorical([v])
    return bayes_params


def save_training_points(out_path, coords, depths, weights, X_data, ref_raster, crs, feature_names=None):
    fields = QgsFields()
    fields.append(QgsField("Depth_Used", QVariant.Double))
    fields.append(QgsField("Weight_Used", QVariant.Double))
    fields.append(QgsField("Row_Idx", QVariant.Int))
    fields.append(QgsField("Col_Idx", QVariant.Int))

    num_bands = X_data.shape[1]
    for b in range(num_bands):
        name = feature_names[b] if feature_names and b < len(feature_names) else f"Band_{b + 1}"
        fields.append(QgsField(name[:10], QVariant.Double))

    writer = QgsVectorFileWriter(
        out_path, "UTF-8", fields, QgsWkbTypes.Point, crs, "ESRI Shapefile"
    )
    if writer.hasError() != QgsVectorFileWriter.NoError:
        return

    with rasterio.open(ref_raster) as src:
        transform = src.transform
        for i, (r, c) in enumerate(coords):
            x, y = rasterio.transform.xy(transform, r, c, offset="center")
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))

            attrs = [float(depths[i]), float(weights[i]), int(r), int(c)]
            band_values = [float(val) for val in X_data[i]]
            attrs.extend(band_values)

            feat.setAttributes(attrs)
            writer.addFeature(feat)
    del writer


def _get_pixel_coords(geom, src, v_crs, r_crs):
    raw_pt = geom.asMultiPoint()[0] if geom.isMultipart() else geom.asPoint()
    h, w = src.height, src.width
    try:
        r_raw, c_raw = src.index(raw_pt.x(), raw_pt.y())
        if 0 <= r_raw < h and 0 <= c_raw < w:
            return r_raw, c_raw, raw_pt
    except Exception:
        pass

    try:
        tr = QgsCoordinateTransform(v_crs, r_crs, QgsProject.instance())
        geom_trans = QgsGeometry(geom)
        geom_trans.transform(tr)
        trans_pt = geom_trans.asMultiPoint()[0] if geom_trans.isMultipart() else geom_trans.asPoint()
        r_trans, c_trans = src.index(trans_pt.x(), trans_pt.y())
        if 0 <= r_trans < h and 0 <= c_trans < w:
            return r_trans, c_trans, trans_pt
    except Exception:
        pass

    return None, None, None


def extract_samples(ras_path, vec_layer, d_fld, w_fld, mode):
    rlayer = QgsRasterLayer(ras_path)
    
    v_crs = vec_layer.crs() if vec_layer and vec_layer.crs().isValid() else QgsCoordinateReferenceSystem("EPSG:4326")
    r_crs = rlayer.crs() if rlayer and rlayer.crs().isValid() else QgsCoordinateReferenceSystem("EPSG:4326")
    
    from ...infrastructure.vector_io import resolve_depth_field
    actual_d_fld = resolve_depth_field(vec_layer, d_fld)
    
    fields_lower = [f.name().lower() for f in vec_layer.fields()]
    actual_w_fld = None
    if w_fld and w_fld.lower() in fields_lower:
        actual_w_fld = [f.name() for f in vec_layer.fields() if f.name().lower() == w_fld.lower()][0]
    else:
        for fallback in ['confidence', 'sdb_uncert', 'weight', 'uncertainty']:
            if fallback in fields_lower:
                actual_w_fld = [f.name() for f in vec_layer.fields() if f.name().lower() == fallback][0]
                break
                
    X_out, y_out, w_out, c_out = [], [], [], []

    with rasterio.open(ras_path) as src:
        d = src.read()
        h, w = src.height, src.width
        rst_transform = src.transform
        pixel_size = abs(src.res[0])

        if mode == 0:
            for f in vec_layer.getFeatures():
                geom = QgsGeometry(f.geometry())
                r, c, pt = _get_pixel_coords(geom, src, v_crs, r_crs)
                if r is not None and c is not None:
                    val = d[:, r, c]
                    if np.all(np.isfinite(val)) and not np.any(val == -9999):
                        X_out.append(val)
                        y_out.append(f[actual_d_fld])
                        c_out.append([r, c])
                        w_out.append(f[actual_w_fld] if actual_w_fld else 1.0)
                    else:
                        from qgis.core import QgsMessageLog, Qgis
                        QgsMessageLog.logMessage(
                            f"Point dropped at [r={r}, c={c}]. val: {val.tolist()}", 
                            "Bathymetrix_AI_Debug", Qgis.Warning)
                else:
                    from qgis.core import QgsMessageLog, Qgis
                    QgsMessageLog.logMessage(
                        f"Point out of bounds or transformation failed. Geom: {geom.asWkt()}", 
                        "Bathymetrix_AI_Debug", Qgis.Warning)
            return np.array(X_out), np.array(y_out), np.array(w_out), c_out

        pixel_registry = {}
        for f in vec_layer.getFeatures():
            geom = QgsGeometry(f.geometry())
            r, c, pt = _get_pixel_coords(geom, src, v_crs, r_crs)
            if r is not None and c is not None:
                pixel_registry.setdefault((r, c), []).append(
                    {"d": f[actual_d_fld], "w": f[actual_w_fld] if actual_w_fld else 1.0, "pt": pt}
                )

        for (r, c), items in pixel_registry.items():
            val = d[:, r, c]
            if not np.all(np.isfinite(val)) or np.any(val == -9999):
                from qgis.core import QgsMessageLog, Qgis
                QgsMessageLog.logMessage(
                    f"Point dropped at [r={r}, c={c}]. val: {val.tolist()}", 
                    "Bathymetrix_AI_Debug", Qgis.Warning)
                continue

            final_depth, final_weight = 0.0, 1.0
            if mode == 1:
                best = sorted(items, key=lambda x: x["w"], reverse=True)[0]
                final_depth, final_weight = best["d"], best["w"]
            elif mode == 2:
                cx, cy = rasterio.transform.xy(rst_transform, r, c, offset="center")
                best = min(
                    items,
                    key=lambda x: (x["pt"].x() - cx) ** 2 + (x["pt"].y() - cy) ** 2,
                )
                final_depth, final_weight = best["d"], best["w"]
            elif mode == 3:
                cx, cy = rasterio.transform.xy(rst_transform, r, c, offset="center")
                close = [
                    i
                    for i in items
                    if ((i["pt"].x() - cx) ** 2 + (i["pt"].y() - cy) ** 2) ** 0.5 <= pixel_size / 2
                ]
                target = close if close else items
                vals = np.array([i["d"] for i in target])
                ws = np.array([i["w"] for i in target])
                final_depth = (
                    np.average(vals, weights=ws) if np.sum(ws) > 0 else np.mean(vals)
                )
                final_weight = np.mean(ws)
            elif mode == 4:
                cx, cy = rasterio.transform.xy(rst_transform, r, c, offset="center")
                close = [
                    i
                    for i in items
                    if ((i["pt"].x() - cx) ** 2 + (i["pt"].y() - cy) ** 2) ** 0.5 <= pixel_size / 2
                ]
                if not close:
                    continue
                vals = np.array([i["d"] for i in close])
                ws = np.array([i["w"] for i in close])
                final_depth = (
                    np.average(vals, weights=ws) if np.sum(ws) > 0 else np.mean(vals)
                )
                final_weight = np.mean(ws)

            X_out.append(val)
            y_out.append(final_depth)
            w_out.append(final_weight)
            c_out.append([r, c])

    return np.array(X_out), np.array(y_out), np.array(w_out), c_out


def save_algo_artifacts(y_t, y_p, pct, name, folder, r2, rmse, mape, params, scaler_name="None"):
    with open(os.path.join(folder, "Results.txt"), "w") as f:
        f.write(
            f"Algo: {name}\nFeature Scaling: {scaler_name}\nR2: {r2:.4f}\nRMSE: {rmse:.4f}\nwMAPE: {mape:.2f}%\nParams: {params}"
        )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.figure(figsize=(8, 6))
        # Plot predictions vs actuals
        plt.scatter(y_t, y_p, color='#1f77b4', alpha=0.6, edgecolors='none', s=20, label='Validation Points')
        
        # Perfect fit line
        min_val = min(float(np.min(y_t)), float(np.min(y_p)))
        max_val = max(float(np.max(y_t)), float(np.max(y_p)))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line')
        
        plt.title(f'{name} - Validation Scatter Plot\nR² = {r2:.3f} | RMSE = {rmse:.2f}m | wMAPE = {mape:.2f}%', fontsize=11, fontweight='bold', pad=10)
        plt.xlabel('Observed Depth (m)', fontsize=10)
        plt.ylabel('Predicted Depth (m)', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(folder, "Validation_Scatter_Plot.png"), dpi=150)
        plt.close()
    except Exception:  # nosec B110
        pass


def get_model_and_params(index, opt_idx=0, random_state=42, n_jobs=-1):
    is_bayes = opt_idx == 2 and SKOPT_AVAILABLE
    if index == 0:
        return "Linear Regression", LinearRegression(), {}
    if index == 1:
        return (
            "Random Forest",
            RandomForestRegressor(random_state=random_state, n_jobs=n_jobs),
            (
                {"n_estimators": Integer(100, 500)}
                if is_bayes
                else {"n_estimators": [100, 500]}
            ),
        )
    if index == 2:
        return (
            "Gradient Boosting",
            GradientBoostingRegressor(random_state=random_state),
            (
                {"learning_rate": Real(0.01, 0.2)}
                if is_bayes
                else {"learning_rate": [0.05, 0.1]}
            ),
        )
    if index == 3:
        return (
            "Extra Trees",
            ExtraTreesRegressor(random_state=random_state, n_jobs=n_jobs),
            (
                {"n_estimators": Integer(100, 500)}
                if is_bayes
                else {"n_estimators": [100, 500]}
            ),
        )
    if index == 4:
        return (
            "Ridge",
            Ridge(),
            ({"alpha": Real(0.1, 10.0)} if is_bayes else {"alpha": [0.1, 1.0]}),
        )
    if index == 5:
        return (
            "Lasso",
            Lasso(),
            ({"alpha": Real(0.01, 1.0)} if is_bayes else {"alpha": [0.01, 0.1]}),
        )
    if index == 6:
        return (
            "ElasticNet",
            ElasticNet(),
            ({"l1_ratio": Real(0.1, 0.9)} if is_bayes else {"l1_ratio": [0.5]}),
        )
    if index == 7:
        return (
            "KNN",
            KNeighborsRegressor(n_jobs=n_jobs),
            ({"n_neighbors": Integer(3, 15)} if is_bayes else {"n_neighbors": [5, 10]}),
        )
    if index == 8:
        return (
            "Decision Tree",
            DecisionTreeRegressor(random_state=random_state),
            ({"max_depth": Integer(5, 20)} if is_bayes else {"max_depth": [5, 10]}),
        )
    if index == 9:
        return (
            "MLP (Neural Net)",
            MLPRegressor(random_state=random_state),
            {
                "hidden_layer_sizes": [(100,), (100, 50)],
                "activation": ["relu", "tanh"],
                "learning_rate_init": [0.001, 0.01],
            },
        )
    if index == 10:
        return (
            "SVR",
            SVR(),
            (
                {"C": Real(1.0, 100.0)}
                if is_bayes
                else {"C": [10, 100], "kernel": ["rbf"]}
            ),
        )
    if index == 11:
        return (
            "Huber Regressor",
            HuberRegressor(),
            (
                {"epsilon": Real(1.1, 1.5)}
                if is_bayes
                else {"epsilon": [1.1, 1.35, 1.5]}
            ),
        )
    if index == 12:
        if not XGB_AVAILABLE:
            return "XGBoost", None, {}
        return (
            "XGBoost",
            xgb.XGBRegressor(random_state=random_state, n_jobs=n_jobs),
            (
                {
                    "n_estimators": Integer(50, 300),
                    "max_depth": Integer(3, 10),
                    "learning_rate": Real(0.01, 0.2),
                }
                if is_bayes
                else {"n_estimators": [100, 200], "max_depth": [4, 6], "learning_rate": [0.05, 0.1]}
            ),
        )
    if index == 13:
        if not LGBM_AVAILABLE:
            return "LightGBM", None, {}
        return (
            "LightGBM",
            lgb.LGBMRegressor(random_state=random_state, n_jobs=n_jobs, verbose=-1),
            (
                {
                    "n_estimators": Integer(50, 300),
                    "max_depth": Integer(3, 10),
                    "learning_rate": Real(0.01, 0.2),
                }
                if is_bayes
                else {"n_estimators": [100, 200], "max_depth": [4, 6], "learning_rate": [0.05, 0.1]}
            ),
        )
    if index == 14:
        if not CATBOOST_AVAILABLE:
            return "CatBoost", None, {}
        return (
            "CatBoost",
            cb.CatBoostRegressor(random_state=random_state, thread_count=n_jobs, verbose=0),
            (
                {
                    "iterations": Integer(50, 300),
                    "depth": Integer(3, 10),
                    "learning_rate": Real(0.01, 0.2),
                }
                if is_bayes
                else {"iterations": [100, 200], "depth": [4, 6], "learning_rate": [0.05, 0.1]}
            ),
        )
    return "Unknown", LinearRegression(), {}


def run_optuna_search(base_model, name, param_distributions, X, y, fit_params, n_iter=20, cv=3, random_state=42, n_jobs=-1, groups=None):
    import optuna
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from sklearn.base import clone

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {}
        for param_name, dist in param_distributions.items():
            cls_name = type(dist).__name__
            if "Integer" in cls_name or "Real" in cls_name:
                low = dist.low
                high = dist.high
                if "Integer" in cls_name:
                    params[param_name] = trial.suggest_int(param_name, int(low), int(high))
                else:
                    is_log = getattr(dist, "prior", None) == "log-uniform"
                    params[param_name] = trial.suggest_float(param_name, float(low), float(high), log=is_log)
            elif "Categorical" in cls_name:
                params[param_name] = trial.suggest_categorical(param_name, dist.categories)
            elif hasattr(dist, 'bounds') and hasattr(dist, 'low') and hasattr(dist, 'high'):
                low = dist.low
                high = dist.high
                params[param_name] = trial.suggest_float(param_name, float(low), float(high))
            elif isinstance(dist, list):
                if all(isinstance(x, int) for x in dist):
                    params[param_name] = trial.suggest_int(param_name, min(dist), max(dist))
                elif all(isinstance(x, (int, float)) for x in dist):
                    params[param_name] = trial.suggest_float(param_name, min(dist), max(dist))
                else:
                    params[param_name] = trial.suggest_categorical(param_name, dist)
            else:
                params[param_name] = trial.suggest_categorical(param_name, [dist])

        if groups is not None:
            from sklearn.model_selection import GroupKFold
            n_unique_groups = len(np.unique(groups))
            if n_unique_groups >= 2:
                actual_cv = min(cv, n_unique_groups)
                kf = GroupKFold(n_splits=actual_cv)
                split_iterator = kf.split(X, y, groups=groups)
            else:
                kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
                split_iterator = kf.split(X)
        else:
            kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
            split_iterator = kf.split(X)

        scores = []
        for train_idx, val_idx in split_iterator:
            X_tr_cv, X_val_cv = X[train_idx], X[val_idx]
            y_tr_cv, y_val_cv = y[train_idx], y[val_idx]
            
            split_fit_params = {}
            if "sample_weight" in fit_params:
                split_fit_params["sample_weight"] = fit_params["sample_weight"][train_idx]

            fold_model = clone(base_model)
            fold_model.set_params(**params)
            fold_model.fit(X_tr_cv, y_tr_cv, **split_fit_params)
            preds = fold_model.predict(X_val_cv)
            scores.append(mean_squared_error(y_val_cv, preds))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_iter, n_jobs=1)
    
    best_params = study.best_params
    best_model = clone(base_model)
    best_model.set_params(**best_params)
    return best_model, best_params


def export_feature_importance(model, win_name, X_val, y_val, out_dir, log_path, feedback, selected_indices=None, feature_names=None):
    """
    Extracts, plots, and saves feature importances (or coefficients/permutation importance)
    for the winning model or ensemble.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    num_features = X_val.shape[1]
    if feature_names is not None:
        if selected_indices is not None and len(feature_names) > len(selected_indices):
            final_names = [feature_names[idx] for idx in selected_indices]
        else:
            final_names = feature_names
    else:
        final_names = [f"Band_{i+1}" for i in range(num_features)]
        if selected_indices is not None:
            final_names = [f"Band_{idx+1}" for idx in selected_indices]
            
    importances = None
    method_used = "Feature Importance"
    
    # 1. Check for native feature importances (Tree-based models)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        method_used = "Native Feature Importance"
    # 2. Check for coefficients (Linear models)
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        total = np.sum(importances)
        if total > 0:
            importances = importances / total
        method_used = "Normalized Absolute Coefficients"
    # 3. Check if it's our CustomEnsembleRegressor
    elif win_name.startswith("Ensemble") and hasattr(model, "fitted_estimators_"):
        base_importances = []
        for est in model.fitted_estimators_:
            if hasattr(est, "feature_importances_"):
                base_importances.append(est.feature_importances_)
            elif hasattr(est, "coef_"):
                coefs = np.abs(est.coef_)
                total = np.sum(coefs)
                if total > 0:
                    base_importances.append(coefs / total)
        if len(base_importances) > 0:
            importances = np.mean(base_importances, axis=0)
            method_used = "Ensemble Average Importance"
            
    # 4. Fallback to Permutation Importance
    if importances is None:
        append_log("   Calculating Permutation Importance (Fallback)...", log_path, feedback)
        from sklearn.inspection import permutation_importance
        subset_size = min(1000, len(X_val))
        X_sub = X_val[:subset_size]
        y_sub = y_val[:subset_size]
        
        result = permutation_importance(model, X_sub, y_sub, n_repeats=3, random_state=42, n_jobs=1)
        importances = np.clip(result.importances_mean, 0, None)
        total = np.sum(importances)
        if total > 0:
            importances = importances / total
        method_used = "Permutation Importance (Normalized)"
        
    # Save report
    df_imp = pd.DataFrame({
        "Feature": final_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    csv_path = os.path.join(out_dir, "3_Model_Feature_Importance_Report.csv")
    df_imp.to_csv(csv_path, index=False)
    
    # Plot importances
    plt.figure(figsize=(10, 6))
    df_plot = df_imp.sort_values(by="Importance", ascending=True)
    plt.barh(df_plot["Feature"], df_plot["Importance"], color="dodgerblue", edgecolor="black", alpha=0.8)
    plt.xlabel("Normalized Importance / Contribution")
    plt.ylabel("Features (Spectral Bands)")
    plt.title(f"Model Feature Importance ({win_name})\nMethod: {method_used}")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    img_path = os.path.join(out_dir, "3_Model_Feature_Importance.png")
    plt.savefig(img_path, dpi=150)
    plt.close()
    
    append_log(f"   Saved feature importance plot and report to: {out_dir}", log_path, feedback)


def run_benchmarking(
    X, y, weights, indices, n_iter, out_dir, feedback, opt_idx, log_path, custom_params,
    test_size=0.2, random_state=42, n_jobs=-1, enable_ensemble=False, ensemble_method="Average",
    spatial_cv=False, coords=None, selected_indices=None, ensemble_size=3, feature_names=None,
    cv_folds=3
):
    X = np.nan_to_num(X, nan=0.0)
    
    groups_tr = None
    if spatial_cv and coords is not None:
        from sklearn.cluster import KMeans
        from sklearn.model_selection import GroupShuffleSplit
        kmeans = KMeans(n_clusters=5, random_state=random_state, n_init=10)
        spatial_groups = kmeans.fit_predict(coords)
        
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, val_idx = next(gss.split(X, y, groups=spatial_groups))
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        w_tr = weights[train_idx]
        groups_tr = spatial_groups[train_idx]
        append_log(f"   [Spatial CV] Split data into 5 geographic clusters using KMeans.", log_path, feedback)
    else:
        X_tr, X_val, y_tr, y_val, w_tr, _ = train_test_split(
            X, y, weights, test_size=test_size, random_state=random_state
        )

    results = []
    for idx in [int(i) for i in indices]:
        name, raw_base_model, default_params = get_model_and_params(idx, opt_idx, random_state, n_jobs)
        if raw_base_model is None:
            append_log(f"      ! Skipping {name}: Library is not installed.", log_path, feedback)
            continue

        if name in custom_params and custom_params[name]:
            parsed_dict = custom_params[name]
            base_params = {k: (v[0] if isinstance(v, list) and len(v)>0 else v) for k, v in parsed_dict.items()}
            raw_base_model.set_params(**base_params)

            if opt_idx == 2 and SKOPT_AVAILABLE:
                params = convert_to_bayes(parsed_dict)
            else:
                params = parsed_dict
        else:
            params = default_params

        algo_dir = os.path.join(out_dir, name.replace(" ", "_"))
        os.makedirs(algo_dir, exist_ok=True)

        try:
            fit_params = {}
            if name in [
                "Random Forest",
                "Gradient Boosting",
                "Extra Trees",
                "Ridge",
                "Lasso",
                "Decision Tree",
                "SVR",
                "Linear Regression",
                "Huber Regressor",
                "XGBoost",
                "LightGBM",
                "CatBoost",
            ]:
                fit_params["sample_weight"] = w_tr

            search_n_jobs = 1 if name in ["Random Forest", "Extra Trees", "KNN", "XGBoost", "LightGBM", "CatBoost"] else n_jobs

            with joblib.parallel_backend("threading", n_jobs=search_n_jobs):
                if params and n_iter > 0:
                    search = None
                    current_opt_idx = opt_idx
                    if name == "MLP (Neural Net)":
                        current_opt_idx = 0
                    
                    cv_splitter = cv_folds
                    if groups_tr is not None:
                        n_unique_groups = len(np.unique(groups_tr))
                        if n_unique_groups >= 2:
                            from sklearn.model_selection import GroupKFold
                            cv_splitter = GroupKFold(n_splits=min(cv_folds, n_unique_groups))
                        else:
                            groups_tr = None

                    if current_opt_idx == 0:
                        import scipy.stats as stats
                        opt_params = {}
                        for k, v in params.items():
                            if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                                if all(isinstance(x, int) for x in v) and v[0] < v[1]:
                                    opt_params[k] = stats.randint(v[0], v[1] + 1)
                                else:
                                    opt_params[k] = stats.uniform(v[0], v[1] - v[0])
                            else:
                                opt_params[k] = v
                        search = RandomizedSearchCV(
                            clone(raw_base_model), opt_params, n_iter=n_iter, cv=cv_splitter, n_jobs=search_n_jobs, random_state=random_state
                        )
                    elif current_opt_idx == 1:
                        opt_params = {}
                        for k, v in params.items():
                            if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                                if all(isinstance(x, int) for x in v) and v[0] < v[1]:
                                    opt_params[k] = list(np.linspace(v[0], v[1], 5, dtype=int))
                                else:
                                    opt_params[k] = list(np.linspace(v[0], v[1], 5))
                            else:
                                opt_params[k] = v
                        search = GridSearchCV(clone(raw_base_model), opt_params, cv=cv_splitter, n_jobs=search_n_jobs)
                    elif current_opt_idx == 2:
                        if OPTUNA_AVAILABLE:
                            best_model_opt, best_params_opt = run_optuna_search(
                                clone(raw_base_model), name, params, X_tr, y_tr, fit_params,
                                n_iter=n_iter, cv=cv_folds, random_state=random_state, n_jobs=n_jobs,
                                groups=groups_tr
                            )
                            model = best_model_opt
                            model.fit(X_tr, y_tr, **fit_params)
                            params_str = str(best_params_opt)
                            search = None
                        else:
                            import scipy.stats as stats
                            opt_params = {}
                            for k, v in params.items():
                                if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                                    if all(isinstance(x, int) for x in v) and v[0] < v[1]:
                                        opt_params[k] = stats.randint(v[0], v[1] + 1)
                                    else:
                                        opt_params[k] = stats.uniform(v[0], v[1] - v[0])
                                else:
                                    opt_params[k] = v
                            search = (
                                BayesSearchCV(clone(raw_base_model), params, n_iter=n_iter, cv=cv_splitter, n_jobs=search_n_jobs)
                                if SKOPT_AVAILABLE
                                else RandomizedSearchCV(
                                    clone(raw_base_model), opt_params, n_iter=n_iter, cv=cv_splitter, n_jobs=search_n_jobs, random_state=random_state
                                )
                            )

                    if search:
                        if groups_tr is not None:
                            search.fit(X_tr, y_tr, groups=groups_tr, **fit_params)
                        else:
                            search.fit(X_tr, y_tr, **fit_params)
                        model = search.best_estimator_
                        params_str = str(search.best_params_)
                    elif not OPTUNA_AVAILABLE or current_opt_idx != 2:
                        model = clone(raw_base_model)
                        model.fit(X_tr, y_tr, **fit_params)
                        params_str = "Default"
                else:
                    model = clone(raw_base_model)
                    model.fit(X_tr, y_tr, **fit_params)
                    params_str = "Default"

            y_p = model.predict(X_val)
            r2 = r2_score(y_val, y_p)
            rmse = np.sqrt(mean_squared_error(y_val, y_p))
            sum_abs_diff = np.sum(np.abs(y_val - y_p))
            sum_abs_true = np.sum(np.abs(y_val))
            wmape = (sum_abs_diff / sum_abs_true) * 100 if sum_abs_true != 0 else 0

            save_algo_artifacts(
                y_val,
                y_p,
                np.abs(y_val - y_p),
                name,
                algo_dir,
                r2,
                rmse,
                wmape,
                params_str,
                scaler_name="None",
            )

            results.append(
                {
                    "Algorithm": name,
                    "Feature Scaling": "None",
                    "Model": model,
                    "R2": r2,
                    "RMSE": rmse,
                    "wMAPE": wmape,
                }
            )
            append_log(
                f"      > {name}: R2={r2:.3f}, RMSE={rmse:.2f}m", log_path, feedback
            )

        except Exception as e:
            append_log(f"      ! Failed {name}: {e}", log_path, feedback)

    if not results:
        raise QgsProcessingException("All selected algorithms failed.")

    if enable_ensemble and len(results) >= 2:
        df_temp = pd.DataFrame(results)
        max_rmse = df_temp["RMSE"].max() if len(df_temp) > 0 else 1.0
        temp_scores = []
        for r in results:
            s_r2 = max(0, r["R2"])
            s_rmse = 1.0 - (r["RMSE"] / max_rmse) if max_rmse != 0 else 0
            temp_scores.append(0.6 * s_r2 + 0.4 * s_rmse)

        sorted_indices = np.argsort(temp_scores)[::-1]
        top_results = [results[i] for i in sorted_indices[:ensemble_size]]

        estimators = [(r["Algorithm"], r["Model"]) for r in top_results]
        append_log(f"   [Ensemble] Blending top models: {[r['Algorithm'] for r in top_results]} using {ensemble_method}", log_path, feedback)

        ensemble_model = CustomEnsembleRegressor(estimators=estimators, method=ensemble_method)
        ensemble_model.fit(X_tr, y_tr, sample_weight=w_tr)

        y_p = ensemble_model.predict(X_val)
        r2 = r2_score(y_val, y_p)
        rmse = np.sqrt(mean_squared_error(y_val, y_p))
        sum_abs_diff = np.sum(np.abs(y_val - y_p))
        sum_abs_true = np.sum(np.abs(y_val))
        wmape = (sum_abs_diff / sum_abs_true) * 100 if sum_abs_true != 0 else 0

        ensemble_dir = os.path.join(out_dir, "Ensemble_Model")
        os.makedirs(ensemble_dir, exist_ok=True)
        save_algo_artifacts(
            y_val,
            y_p,
            np.abs(y_val - y_p),
            f"Ensemble ({ensemble_method})",
            ensemble_dir,
            r2,
            rmse,
            wmape,
            f"Method: {ensemble_method}, Models: {[r['Algorithm'] for r in top_results]}",
            scaler_name="Composite",
        )

        results.append(
            {
                "Algorithm": f"Ensemble ({ensemble_method})",
                "Feature Scaling": "Composite",
                "Model": ensemble_model,
                "R2": r2,
                "RMSE": rmse,
                "wMAPE": wmape,
            }
        )
        append_log(
            f"      > Ensemble ({ensemble_method}): R2={r2:.3f}, RMSE={rmse:.2f}m", log_path, feedback
        )

    df = pd.DataFrame(results)
    df["score"] = (0.6 * df["R2"].clip(lower=0)) + (
        0.4 * (1 - (df["RMSE"] / df["RMSE"].max()))
    )
    
    winner = df.loc[df["score"].idxmax()]

    if enable_ensemble:
        ens_name = f"Ensemble ({ensemble_method})"
        if any(df["Algorithm"] == ens_name):
            ens_row = df[df["Algorithm"] == ens_name].iloc[0]
            if winner["Algorithm"] != ens_name:
                append_log(f"   [Ensemble] Ensemble did not beat winner {winner['Algorithm']} (Ensemble Score={ens_row['score']:.4f} vs Winner Score={winner['score']:.4f})", log_path, feedback)
            else:
                append_log(f"   [Ensemble] Ensemble wins the benchmark! Score={winner['score']:.4f}", log_path, feedback)

    final_model = winner["Model"]
    fit_params_final = {}
    if winner["Algorithm"].startswith("Ensemble"):
        fit_params_final["sample_weight"] = weights
    elif winner["Algorithm"] in [
        "Random Forest",
        "Gradient Boosting",
        "Extra Trees",
        "Ridge",
        "Lasso",
        "Decision Tree",
        "SVR",
        "Linear Regression",
        "Huber Regressor",
        "XGBoost",
        "LightGBM",
        "CatBoost",
    ]:
        fit_params_final["sample_weight"] = weights

    final_search_n_jobs = 1 if winner["Algorithm"] in ["Random Forest", "Extra Trees", "KNN", "XGBoost", "LightGBM", "CatBoost"] else n_jobs
    with joblib.parallel_backend("threading", n_jobs=final_search_n_jobs):
        final_model.fit(X, y, **fit_params_final)

    try:
        export_feature_importance(
            final_model,
            winner["Algorithm"],
            X_val,
            y_val,
            out_dir,
            log_path,
            feedback,
            selected_indices,
            feature_names
        )
    except Exception as e:
        append_log(f"   [Warning] Failed to generate feature importance plot: {e}", log_path, feedback)

    return df, {
        "name": winner["Algorithm"],
        "scaler": winner.get("Feature Scaling", "None"),
        "model": final_model,
        "score": winner["score"],
        "r2": winner["R2"],
        "rmse": winner["RMSE"],
        "wmape": winner["wMAPE"],
    }


def predict_map(model, stack_path, mask_path, out_path, med_size, output_format="float32", selected_indices=None):
    with rasterio.open(stack_path) as s:
        d = s.read()
        h, w = s.height, s.width
        d_flat = d.reshape(d.shape[0], -1).T
        prof = s.profile.copy()
    
    if mask_path and str(mask_path).strip() and mask_path != "None":
        with rasterio.open(mask_path) as m:
            mask_arr = m.read(1).flatten()
    else:
        mask_arr = np.ones(h * w, dtype=np.uint8)
        
    water_idx = np.where(mask_arr == 1)[0]
    if len(water_idx) == 0:
        return
        
    out_img = np.full(h * w, -9999.0, dtype="float32")
    model_str = str(model)
    if "KNeighbors" in model_str or "GaussianProcess" in model_str or "RobustSpatialKNN" in model_str or "KNN" in model_str:
        chunk_size = 5000
    else:
        chunk_size = 500000
    n_pixels = len(water_idx)
    
    for start in range(0, n_pixels, chunk_size):
        end = min(start + chunk_size, n_pixels)
        batch_idx = water_idx[start:end]
        
        X_chunk = d_flat[batch_idx]
        if selected_indices is not None and len(selected_indices) > 0:
            X_chunk = X_chunk[:, selected_indices]
            
        np.nan_to_num(X_chunk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        chunk_preds = model.predict(X_chunk)
        out_img[batch_idx] = chunk_preds

    out_img = out_img.reshape(h, w)
    if med_size > 0 and scipy_is_available:
        valid = out_img != -9999.0
        temp = out_img.copy()
        temp[~valid] = 0
        filt = median_filter(temp, size=med_size)
        filt[~valid] = -9999.0
        out_img = filt
        
    nodata_val = -9999.0
    if output_format == "uint16":
        valid = out_img != -9999.0
        out_img[valid] = np.clip(out_img[valid], 0, None)
        out_img[~valid] = 65535
        out_img = out_img.astype("uint16")
        nodata_val = 65535.0
    elif output_format == "float64":
        out_img = out_img.astype("float64")
    else:
        out_img = out_img.astype("float32")

    prof.update(count=1, dtype=output_format, nodata=nodata_val)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(out_img, 1)


def run_phase03_initial_modeling(algorithm, parameters, context, feedback):
    out_dir = algorithm.parameterAsString(parameters, algorithm.OUTPUT_FOLDER, context)
    os.makedirs(out_dir, exist_ok=True)
    log_path = algorithm.parameterAsString(parameters, algorithm.LOG_FILE, context)

    custom_params = {
        "Random Forest": parse_param_string(
            algorithm.parameterAsString(parameters, algorithm.PARAM_RF, context)
        ),
        "Gradient Boosting": parse_param_string(
            algorithm.parameterAsString(parameters, algorithm.PARAM_GB, context)
        ),
        "Extra Trees": parse_param_string(
            algorithm.parameterAsString(parameters, algorithm.PARAM_ET, context)
        ),
        "SVR": parse_param_string(
            algorithm.parameterAsString(parameters, algorithm.PARAM_SVR, context)
        ),
        "MLP (Neural Net)": parse_param_string(
            algorithm.parameterAsString(parameters, algorithm.PARAM_MLP, context)
        ),
        "Ridge": parse_param_string(
            algorithm.parameterAsString(parameters, algorithm.PARAM_RIDGE, context)
        ),
        "Lasso": parse_param_string(
            algorithm.parameterAsString(parameters, algorithm.PARAM_LASSO, context)
        ),
        "ElasticNet": parse_param_string(
            algorithm.parameterAsString(parameters, algorithm.PARAM_ELASTICNET, context)
        ),
        "KNN": parse_param_string(
            algorithm.parameterAsString(parameters, algorithm.PARAM_KNN, context)
        ),
        "Decision Tree": parse_param_string(
            algorithm.parameterAsString(parameters, algorithm.PARAM_DT, context)
        ),
        "Huber Regressor": parse_param_string(
            algorithm.parameterAsString(parameters, "PARAM_HUBER", context)
        ),
        "XGBoost": parse_param_string(
            algorithm.parameterAsString(parameters, "PARAM_XGB", context)
        ),
        "LightGBM": parse_param_string(
            algorithm.parameterAsString(parameters, "PARAM_LGBM", context)
        ),
        "CatBoost": parse_param_string(
            algorithm.parameterAsString(parameters, "PARAM_CATBOOST", context)
        ),
    }

    train_ratio = algorithm.parameterAsDouble(parameters, algorithm.TRAIN_TEST_SPLIT, context)
    test_size = 1.0 - train_ratio
    if test_size <= 0.0 or test_size >= 1.0:
        test_size = 0.2
    
    random_state = algorithm.parameterAsInt(parameters, algorithm.RANDOM_STATE, context)
    n_jobs = algorithm.parameterAsInt(parameters, algorithm.NUM_THREADS, context)
    
    cv_folds = 3
    if algorithm.parameterDefinition("CV_FOLDS"):
        try:
            val = algorithm.parameterAsInt(parameters, "CV_FOLDS", context)
            if val > 1: cv_folds = val
        except: pass

    uncert_trees = 50
    if algorithm.parameterDefinition("UNCERT_TREES"):
        try:
            val = algorithm.parameterAsInt(parameters, "UNCERT_TREES", context)
            if val > 0: uncert_trees = val
        except: pass

    fmt_idx = algorithm.parameterAsEnum(parameters, algorithm.OUTPUT_FORMAT, context)
    fmt_map = {0: "float32", 1: "float64", 2: "uint16"}
    output_format = fmt_map.get(fmt_idx, "float32")

    stack_path = algorithm.parameterAsRasterLayer(
        parameters, algorithm.INPUT_STACK, context
    ).source()
    mask_layer = algorithm.parameterAsRasterLayer(
        parameters, algorithm.INPUT_MASK, context
    )
    mask_path = mask_layer.source() if mask_layer else None
    points_layer = algorithm.parameterAsVectorLayer(
        parameters, algorithm.INPUT_POINTS, context
    )
    depth_fld = algorithm.parameterAsString(parameters, algorithm.FIELD_DEPTH, context)
    weight_fld = algorithm.parameterAsString(
        parameters, algorithm.FIELD_WEIGHT, context
    )
    if not weight_fld:
        weight_fld = None

    n_iter = algorithm.parameterAsInt(parameters, algorithm.N_ITERATIONS, context)
    med_size = algorithm.parameterAsInt(parameters, algorithm.MEDIAN_SIZE, context)
    sel_idx = algorithm.parameterAsEnums(parameters, algorithm.SELECTED_ALGOS, context)
    opt_idx = algorithm.parameterAsInt(parameters, algorithm.OPTIMIZER_METHOD, context)
    col_mode = algorithm.parameterAsInt(
        parameters, algorithm.COLLISION_HANDLING, context
    )
    
    try:
        val_idx = algorithm.parameterAsEnum(parameters, algorithm.FEATURE_CORR_THRESHOLD, context)
        corr_threshold = val_idx * 0.1
    except:
        try:
            corr_threshold = algorithm.parameterAsDouble(parameters, algorithm.FEATURE_CORR_THRESHOLD, context)
        except:
            corr_threshold = 0.2

    try:
        corr_method_idx = algorithm.parameterAsEnum(parameters, algorithm.FEATURE_CORR_METHOD, context)
    except:
        corr_method_idx = 3

    append_log(
        f"MODULE 03 START: Optimizer = {OPTIMIZER_LIST[opt_idx]}", log_path, feedback
    )

    X, y, final_weights, coords = extract_samples(
        stack_path, points_layer, depth_fld, weight_fld, col_mode
    )
    if len(y) < 10:
        raise QgsProcessingException("Critically low training points (<10).")
    append_log(f"   Extracted {len(y)} training pixels.", log_path, feedback)

    actual_pts_path = os.path.join(out_dir, "3_Actual_Model_Input_Points.shp")
    
    # Try to extract feature names
    feature_names = []
    try:
        import rasterio
        with rasterio.open(stack_path) as src:
            for i, desc in enumerate(src.descriptions):
                if desc and str(desc).strip():
                    feature_names.append(str(desc).strip())
                else:
                    feature_names.append(f"Band_{i+1}")
    except Exception:
        feature_names = [f"Band_{i+1}" for i in range(X.shape[1])]
    
    if not feature_names or len(feature_names) != X.shape[1]:
        feature_names = [f"Band_{i+1}" for i in range(X.shape[1])]

    save_training_points(
        actual_pts_path,
        coords,
        y,
        final_weights,
        X,
        stack_path,
        QgsRasterLayer(stack_path).crs(),
        feature_names
    )
    
    selected_indices = None
    if corr_method_idx == 3:
        append_log(f"   [Feature Analysis] Running Automatic-RANSAC Selection...", log_path, feedback)
        num_bands = X.shape[1]
        correlations = []
        method_name = "Automatic-RANSAC (Robust Pearson)"
        
        try:
            from sklearn.linear_model import RANSACRegressor, LinearRegression
        except ImportError:
            append_log(f"   [Warning] sklearn not found. Falling back to Pearson.", log_path, feedback)
            corr_method_idx = 1
            
    if corr_method_idx == 4:
        append_log(f"   [Feature Analysis] Running Automatic-Random Forest Selection...", log_path, feedback)
        num_bands = X.shape[1]
        method_name = "Automatic-Random Forest (Importance)"
        
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            append_log(f"   [Warning] sklearn not found. Falling back to Pearson.", log_path, feedback)
            corr_method_idx = 1

    if corr_method_idx == 3:
        for b in range(num_bands):
            X_b = X[:, b].reshape(-1, 1)
            try:
                ransac = RANSACRegressor(estimator=LinearRegression(), random_state=42)
                ransac.fit(X_b, y)
                inlier_mask = ransac.inlier_mask_
                
                if np.sum(inlier_mask) > 1:
                    r = np.corrcoef(X[inlier_mask, b], y[inlier_mask])[0, 1]
                else:
                    r = 0.0
            except Exception:
                r = 0.0
                
            if np.isnan(r):
                r = 0.0
            correlations.append(r)
            
        correlations = np.array(correlations)
        abs_correlations = np.abs(correlations)
        plot_scores = abs_correlations
        
        valid_scores = abs_correlations[abs_correlations > 0]
        if len(valid_scores) > 0:
            mean_score = float(np.mean(valid_scores))
            std_score = float(np.std(valid_scores))
            corr_threshold = max(0.3, mean_score - std_score)
        else:
            corr_threshold = 0.0
            
        append_log(f"   [Feature Analysis] Auto-Calculated Threshold = {corr_threshold:.3f}", log_path, feedback)
        selected_indices = np.where(abs_correlations >= corr_threshold)[0]
        
    elif corr_method_idx == 4:
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        plot_scores = rf.feature_importances_
        correlations = plot_scores
        
        corr_threshold = max(0.02, 1.0 / (num_bands * 2))
        append_log(f"   [Feature Analysis] Auto-Calculated RF Threshold = {corr_threshold:.3f}", log_path, feedback)
        selected_indices = np.where(plot_scores >= corr_threshold)[0]

    elif corr_method_idx in [1, 2] and corr_threshold > 0.0:
        append_log(f"   [Feature Analysis] Running with threshold >= {corr_threshold}", log_path, feedback)
        num_bands = X.shape[1]
        correlations = []
        method_name = "Spearman" if corr_method_idx == 2 else "Pearson"
        
        if corr_method_idx == 2:
            try:
                from scipy.stats import spearmanr
            except ImportError:
                method_name = "Pearson (Fallback)"
                corr_method_idx = 1
                
        for b in range(num_bands):
            std_X = np.std(X[:, b])
            std_y = np.std(y)
            if std_X == 0 or std_y == 0:
                r = 0.0
            else:
                if corr_method_idx == 2:
                    r = spearmanr(X[:, b], y)[0]
                else:
                    r = np.corrcoef(X[:, b], y)[0, 1]
            if np.isnan(r):
                r = 0.0
            correlations.append(r)
        
        correlations = np.array(correlations)
        abs_correlations = np.abs(correlations)
        plot_scores = abs_correlations
        selected_indices = np.where(abs_correlations >= corr_threshold)[0]
        
    if corr_method_idx > 0 and (corr_threshold > 0.0 or corr_method_idx in [3, 4]):
        if len(selected_indices) == 0:
            append_log(f"   [Warning] No bands met threshold {corr_threshold:.3f}. Using all bands.", log_path, feedback)
            selected_indices = np.arange(num_bands)
        else:
            append_log(f"   [Feature Analysis] Selected {len(selected_indices)} bands: {list(selected_indices)}", log_path, feedback)
            X = X[:, selected_indices]
            
        report_path = os.path.join(out_dir, "3_Feature_Analysis_Report.txt")
        with open(report_path, "w") as f:
            f.write(f"Feature Analysis - {method_name}\n")
            if corr_method_idx in [3, 4]:
                f.write(f"Automatically Calculated Threshold: {corr_threshold:.3f}\n")
            f.write("-" * 50 + "\n")
            for b in range(num_bands):
                status = "Selected" if b in selected_indices else "Discarded"
                fname = feature_names[b]
                score_label = "Importance" if corr_method_idx == 4 else "abs(r)"
                f.write(f"{fname}: {score_label} = {plot_scores[b]:.4f} [{status}]\n")

        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            bars = plt.bar(range(1, num_bands + 1), plot_scores, color='skyblue')
            plt.axhline(y=corr_threshold, color='r', linestyle='--', label=f'Threshold ({corr_threshold:.3f})')
            for i, b_bar in enumerate(bars):
                if i not in selected_indices:
                    b_bar.set_color('lightgray')
            plt.xlabel('Band Number')
            y_label = "Feature Importance" if corr_method_idx == 4 else f"Absolute {method_name} Correlation (|r|)"
            plt.ylabel(y_label)
            plt.title(f'Feature Analysis: {method_name}')
            plt.xticks(range(1, num_bands + 1), feature_names, rotation=45, ha='right')
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "3_Feature_Correlation_Plot.png"), dpi=150)
            plt.close()
        except Exception as e:
            append_log(f"   [Warning] Failed to generate correlation plot: {e}", log_path, feedback)

    # Direct layer addition removed to prevent auto-loading in panel.
    # The output is returned to the processing framework instead.
    # QgsProject.instance().addMapLayer(QgsVectorLayer(actual_pts_path, "3_Actual_Model_Input_Points", "ogr"))

    try:
        enable_ensemble = algorithm.parameterAsBool(parameters, "ENABLE_ENSEMBLE", context)
    except Exception:
        enable_ensemble = False

    try:
        ens_idx = algorithm.parameterAsEnum(parameters, "ENSEMBLE_METHOD", context)
        ens_map = {0: "Average", 1: "Median", 2: "Stacking"}
        ensemble_method = ens_map.get(ens_idx, "Average")
    except Exception:
        ensemble_method = "Average"

    try:
        spatial_cv = algorithm.parameterAsBool(parameters, "SPATIAL_CV", context)
    except Exception:
        spatial_cv = False

    try:
        ensemble_size = algorithm.parameterAsInt(parameters, "ENSEMBLE_SIZE", context)
    except Exception:
        ensemble_size = 3



    results_df, best_algo_data = run_benchmarking(
        X,
        y,
        final_weights,
        sel_idx,
        n_iter,
        out_dir,
        feedback,
        opt_idx,
        log_path,
        custom_params,
        test_size,
        random_state,
        n_jobs,
        enable_ensemble,
        ensemble_method,
        spatial_cv,
        coords,
        selected_indices,
        ensemble_size,
        feature_names,
        cv_folds
    )
    results_df.to_csv(
        os.path.join(out_dir, "3_All_Algorithms_Benchmark.csv"), index=False
    )

    # ---------------------------------------------------------
    # [NEW] Generate strictly Linear Regression SDB for Analytics
    # ---------------------------------------------------------
    lr_algo_name = "Linear Regression"
    lr_model = None

    if lr_algo_name in results_df["Algorithm"].values:
        lr_model = results_df[results_df["Algorithm"] == lr_algo_name].iloc[0]["Model"]
        append_log("   [Analytics] Linear Regression was manually selected. Generating its depth map...", log_path, feedback)
    else:
        append_log("   [Analytics] Linear Regression not manually selected. Running it isolated for analytics...", log_path, feedback)
        try:
            lr_df, lr_best = run_benchmarking(
                X, y, final_weights, [0], n_iter, out_dir, feedback, opt_idx, log_path, custom_params,
                test_size, random_state, n_jobs, False, "Average", spatial_cv, coords, selected_indices, 3, feature_names, cv_folds
            )
            lr_model = lr_best["model"]
        except Exception as e:
            append_log(f"   [Warning] Isolated Linear Regression failed: {e}", log_path, feedback)

    if lr_model is not None:
        lr_dir = os.path.join(out_dir, "Linear_Regression")
        os.makedirs(lr_dir, exist_ok=True)
        lr_map_path = os.path.join(lr_dir, "Linear_Regression_Depth.tif")
        lr_uncert_path = os.path.join(lr_dir, "Linear_Regression_Uncertainty.tif")
        try:
            from sklearn.base import clone
            final_lr_model = clone(lr_model)
            fit_kwargs = {"sample_weight": final_weights}
            final_lr_model.fit(X, y, **fit_kwargs)
            
            predict_map(final_lr_model, stack_path, mask_path, lr_map_path, med_size, output_format, selected_indices)
            joblib.dump(final_lr_model, os.path.join(lr_dir, "Linear_Regression_Model.pkl"))

            # Generate Uncertainty map for Linear Regression
            y_lr_pred = final_lr_model.predict(X)
            lr_abs_residuals = np.abs(y - y_lr_pred) * 1.96
            from sklearn.ensemble import RandomForestRegressor
            lr_uncert_model = RandomForestRegressor(n_estimators=uncert_trees, random_state=random_state, n_jobs=n_jobs)
            lr_uncert_model.fit(X, lr_abs_residuals)
            predict_map(lr_uncert_model, stack_path, mask_path, lr_uncert_path, med_size, "float32", selected_indices)

            append_log(f"   [Analytics] Linear Regression depth & uncertainty analytics saved to: {lr_dir}", log_path, feedback)
        except Exception as e:
            append_log(f"   [Warning] Failed to generate Linear Regression map: {e}", log_path, feedback)
    # ---------------------------------------------------------

    win_name = best_algo_data["name"]
    append_log(
        f"\n   >>> WINNER: {win_name} (Score={best_algo_data['score']:.4f})",
        log_path,
        feedback,
    )

    joblib.dump(
        best_algo_data["model"], os.path.join(out_dir, "3_Best_Global_Model.pkl")
    )
    p_map = os.path.join(out_dir, "3_Initial_Global_Depth.tif")

    append_log("   Generating prediction map...", log_path, feedback)
    predict_map(best_algo_data["model"], stack_path, mask_path, p_map, med_size, output_format, selected_indices)
    # Direct layer addition removed to prevent auto-loading in panel.
    # The output is returned to the processing framework instead.
    # QgsProject.instance().addMapLayer(QgsRasterLayer(p_map, f"3_Initial_Global_Depth ({win_name})"))

    p_uncert_map = os.path.join(out_dir, "3_Initial_Global_Uncertainty.tif")
    try:
        append_log("   Fitting Phase 03 uncertainty model (Empirical Residual Regressor)...", log_path, feedback)
        y_train_pred = best_algo_data["model"].predict(X)
        abs_residuals = np.abs(y - y_train_pred)
        uncert_y = abs_residuals * 1.96
        
        from sklearn.ensemble import RandomForestRegressor
        uncertainty_model = RandomForestRegressor(n_estimators=uncert_trees, random_state=random_state, n_jobs=n_jobs)
        uncertainty_model.fit(X, uncert_y)
        
        append_log("   Generating Phase 03 uncertainty prediction map...", log_path, feedback)
        predict_map(uncertainty_model, stack_path, mask_path, p_uncert_map, med_size, "float32", selected_indices)
    except Exception as e:
        append_log(f"   [Warning] Failed to generate Phase 03 uncertainty map: {e}", log_path, feedback)
        p_uncert_map = None

    return {
        "OUTPUT_DEPTH_MAP": p_map,
        "OUTPUT_UNCERT_MAP": p_uncert_map,
        "OUTPUT_MODEL_PKL": os.path.join(out_dir, "3_Best_Global_Model.pkl"),
        "BEST_R2": best_algo_data["r2"],
        "BEST_RMSE": best_algo_data["rmse"],
    }
