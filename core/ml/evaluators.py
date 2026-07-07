import ast
import os
import joblib
import warnings

import matplotlib
import numpy as np
import rasterio
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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from .trainers import (
    CustomEnsembleRegressor,
    run_optuna_search,
    OPTUNA_AVAILABLE,
    XGB_AVAILABLE,
    LGBM_AVAILABLE,
    CATBOOST_AVAILABLE,
)

if XGB_AVAILABLE:
    import xgboost as xgb
if LGBM_AVAILABLE:
    import lightgbm as lgb
if CATBOOST_AVAILABLE:
    import catboost as cb

from qgis.core import (
    QgsCoordinateTransform,
    QgsProcessingException,
    QgsProject,
    QgsRasterLayer,
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
    if not param_str or param_str.strip() == "":
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


def extract_values(ras, vec, fld, mode, logger_file, fb):
    with rasterio.open(ras) as ds:
        d = ds.read()
        h, w = ds.height, ds.width
        tr = QgsCoordinateTransform(
            vec.sourceCrs(), QgsRasterLayer(ras).crs(), QgsProject.instance()
        )
        X_out, y_out, c_out = [], [], []

        for f in vec.getFeatures():
            g = f.geometry()
            g.transform(tr)
            pt = g.asPoint()
            r, c = ds.index(pt.x(), pt.y())
            if 0 <= r < h and 0 <= c < w:
                val = d[:, r, c]
                if np.all(np.isfinite(val)) and not np.any(val == -9999):
                    X_out.append(val)
                    y_out.append(f[fld])
                    c_out.append([r, c])

    return np.array(X_out), np.array(y_out), np.array(c_out)


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


class RobustSpatialKNN:
    def __init__(self, n_neighbors=15):
        self.n_neighbors = n_neighbors
        self.coords_tr = None
        self.residuals = None
        self.huber_w = None

    def fit(self, coords_tr, residuals):
        self.coords_tr = coords_tr
        self.residuals = residuals
        from sklearn.neighbors import KNeighborsRegressor
        knn = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights="distance")
        knn.fit(coords_tr, residuals)
        errors = residuals - knn.predict(coords_tr)
        mad = np.median(np.abs(errors))
        scale = 1.4826 * mad if mad > 0 else 1e-4
        self.huber_w = np.clip(1.35 * scale / (np.abs(errors) + 1e-6), 0.0, 1.0)
        return self

    def predict(self, X_query):
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=self.n_neighbors)
        nn.fit(self.coords_tr)
        dists, indices = nn.kneighbors(X_query)
        inv_dists = 1.0 / (dists + 1e-6)
        w = inv_dists * self.huber_w[indices]
        w_sum = np.sum(w, axis=1, keepdims=True)
        w_sum[w_sum == 0] = 1.0
        predictions = np.sum(w * self.residuals[indices], axis=1) / w_sum.flatten()
        return predictions


def run_phase04_spatial_retraining(algorithm, parameters, context, feedback):
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
    fmt_idx = algorithm.parameterAsEnum(parameters, algorithm.OUTPUT_FORMAT, context)
    fmt_map = {0: "float32", 1: "float64", 2: "uint16"}
    output_format = fmt_map.get(fmt_idx, "float32")

    global_path = algorithm.parameterAsRasterLayer(
        parameters, algorithm.INPUT_GLOBAL_RASTER, context
    ).source()
    feat_path = algorithm.parameterAsRasterLayer(
        parameters, algorithm.INPUT_ORIGINAL_FEAT, context
    ).source()
    mask_layer = algorithm.parameterAsRasterLayer(
        parameters, algorithm.INPUT_MASK, context
    )
    mask_path = mask_layer.source() if mask_layer else None
    train_lyr = algorithm.parameterAsVectorLayer(
        parameters, algorithm.INPUT_TRAIN, context
    )
    train_fld = algorithm.parameterAsString(parameters, algorithm.FIELD_TRAIN, context)

    sel_idx = algorithm.parameterAsEnums(parameters, algorithm.SELECTED_ALGOS, context)
    n_iter = algorithm.parameterAsInt(parameters, algorithm.N_ITERATIONS, context)
    med_size = algorithm.parameterAsInt(parameters, algorithm.MEDIAN_SIZE, context)
    opt_idx = algorithm.parameterAsInt(parameters, algorithm.OPTIMIZER_METHOD, context)

    col_mode = 0
    
    try:
        corr_idx = algorithm.parameterAsEnum(parameters, algorithm.FEATURE_CORR_THRESHOLD, context)
        corr_threshold = float(corr_idx) / 10.0
    except:
        corr_threshold = 0.2

    try:
        corr_method_idx = algorithm.parameterAsEnum(parameters, algorithm.FEATURE_CORR_METHOD, context)
    except:
        corr_method_idx = 1

    append_log(
        f"MODULE 04 START: Refinement Phase | Opt={OPTIMIZER_LIST[opt_idx]}",
        log_path,
        feedback,
    )

    z_pred3, z_true, coords_tr = extract_values(
        global_path, train_lyr, train_fld, col_mode, log_path, feedback
    )

    if len(z_pred3) < 5:
        raise QgsProcessingException(
            "Not enough points for Module 4 (Spatial Refinement)."
        )

    residuals = z_true.flatten() - z_pred3.flatten()

    try:
        interp_idx = algorithm.parameterAsEnum(parameters, "RESIDUAL_INTERP_METHOD", context)
    except Exception:
        interp_idx = 0

    try:
        spatial_cv = algorithm.parameterAsBool(parameters, "SPATIAL_CV", context)
    except Exception:
        spatial_cv = False

    try:
        knn_k = algorithm.parameterAsInt(parameters, "KNN_NEIGHBORS", context)
    except Exception:
        knn_k = 15

    if interp_idx == 0:
        interp_name = "KNN Standard"
        append_log(f"   Fitting Standard Spatial KNN (K={knn_k})...", log_path, feedback)
        spatial_model = KNeighborsRegressor(n_neighbors=knn_k, weights="distance", n_jobs=n_jobs)
        spatial_model.fit(coords_tr, residuals)
    elif interp_idx == 1:
        interp_name = "KNN Robust"
        append_log(f"   Fitting Robust Spatial KNN (Huber weights, K={knn_k})...", log_path, feedback)
        spatial_model = RobustSpatialKNN(n_neighbors=knn_k)
        spatial_model.fit(coords_tr, residuals)
    else:
        interp_name = "Kriging/GP"
        append_log("   Fitting Gaussian Process Regression (Kriging)...", log_path, feedback)
        if len(coords_tr) > 1500:
            np.random.seed(random_state)
            gpr_idx = np.random.choice(len(coords_tr), size=1500, replace=False)
            coords_gpr = coords_tr[gpr_idx]
            residuals_gpr = residuals[gpr_idx]
        else:
            coords_gpr = coords_tr
            residuals_gpr = residuals
        
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=100.0, length_scale_bounds=(10, 10000))
        spatial_model = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=2, random_state=random_state)
        spatial_model.fit(coords_gpr, residuals_gpr)

    append_log(f"   Interpolating Residual Grid ({interp_name})...", log_path, feedback)
    
    if mask_path and str(mask_path).strip() and mask_path != "None":
        with rasterio.open(mask_path) as m:
            mask_arr = m.read(1)
            meta = m.profile
            h, w = m.height, m.width
        water_indices = np.where(mask_arr == 1)
    else:
        with rasterio.open(global_path) as m:
            meta = m.profile
            h, w = m.height, m.width
        mask_arr = np.ones((h, w), dtype=np.uint8)
        water_indices = np.where(mask_arr == 1)
        
    water_coords = np.column_stack((water_indices[0], water_indices[1]))

    residual_grid = np.zeros((h, w), dtype="float32")
    chunk_size = 500000
    for i in range(0, len(water_coords), chunk_size):
        chunk = water_coords[i:i + chunk_size]
        if len(chunk) > 0:
            residual_grid[chunk[:, 0], chunk[:, 1]] = spatial_model.predict(chunk)

    append_log("   Saving Residual Error Map...", log_path, feedback)
    p_residual = os.path.join(out_dir, "4-Residual_Error_Map.tif")
    meta_res = meta.copy()
    meta_res.update(count=1, dtype="float32", nodata=-9999.0)

    res_map_to_save = np.full((h, w), -9999.0, dtype="float32")
    res_map_to_save[water_indices] = residual_grid[water_indices]

    with rasterio.open(p_residual, "w", **meta_res) as dst:
        dst.write(res_map_to_save, 1)

    with rasterio.open(feat_path) as f:
        orig_bands = f.read()
    with rasterio.open(global_path) as g:
        p3_map = g.read(1)

    stack = np.concatenate(
        [orig_bands, p3_map[np.newaxis, :, :], residual_grid[np.newaxis, :, :]], axis=0
    )
    p_stack = os.path.join(out_dir, "Phase4_Input_Stack.tif")
    meta.update(count=stack.shape[0], dtype="float32")
    with rasterio.open(p_stack, "w", **meta) as dst:
        dst.write(stack.astype("float32"))

    X_final, y_final, _ = extract_values(
        p_stack, train_lyr, train_fld, col_mode, log_path, feedback
    )
    y_final = y_final.flatten()

    selected_indices = None
    if corr_threshold > 0.0:
        append_log(f"   [Feature Analysis] Running with threshold >= {corr_threshold}", log_path, feedback)
        num_bands = X_final.shape[1]
        correlations = []
        method_name = "Spearman" if corr_method_idx == 1 else "Pearson"
        
        if corr_method_idx == 1:
            try:
                from scipy.stats import spearmanr
            except ImportError:
                method_name = "Pearson (Fallback)"
                corr_method_idx = 0
                
        for b in range(num_bands):
            std_X = np.std(X_final[:, b])
            std_y = np.std(y_final)
            if std_X == 0 or std_y == 0:
                r = 0.0
            else:
                if corr_method_idx == 1:
                    r = spearmanr(X_final[:, b], y_final)[0]
                else:
                    r = np.corrcoef(X_final[:, b], y_final)[0, 1]
            if np.isnan(r):
                r = 0.0
            correlations.append(r)
        
        abs_correlations = np.abs(np.array(correlations))
        selected_indices = np.where(abs_correlations >= corr_threshold)[0]
        
        if len(selected_indices) == 0:
            append_log(f"   [Warning] No bands met threshold {corr_threshold}. Using all bands.", log_path, feedback)
            selected_indices = np.arange(num_bands)
        else:
            append_log(f"   [Feature Analysis] Selected {len(selected_indices)} bands: {list(selected_indices)}", log_path, feedback)
            X_final = X_final[:, selected_indices]
            
        report_path = os.path.join(out_dir, "4_Feature_Analysis_Report.txt")
        with open(report_path, "w") as f:
            f.write(f"Phase 04 Feature Analysis - {method_name} Correlation with Depth\n")
            f.write("-" * 50 + "\n")
            for b in range(num_bands):
                status = "Selected" if b in selected_indices else "Discarded"
                f.write(f"Feature_{b+1}: r = {correlations[b]:.4f}  | abs(r) = {abs_correlations[b]:.4f}  [{status}]\n")

    groups_tr = None
    if spatial_cv and coords_tr is not None:
        from sklearn.cluster import KMeans
        from sklearn.model_selection import GroupShuffleSplit
        kmeans = KMeans(n_clusters=5, random_state=random_state, n_init=10)
        spatial_groups = kmeans.fit_predict(coords_tr)
        
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, val_idx = next(gss.split(X_final, y_final, groups=spatial_groups))
        X_train, X_val = X_final[train_idx], X_final[val_idx]
        y_train, y_val = y_final[train_idx], y_final[val_idx]
        groups_tr = spatial_groups[train_idx]
        append_log(f"   [Spatial CV] Split Phase 04 retraining data into 5 geographic clusters.", log_path, feedback)
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_final, y_final, test_size=test_size, random_state=random_state
        )

    best_rmse = float("inf")
    best_model = None
    best_algo_name = ""
    best_r2 = 0.0
    best_wmape = 0.0

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
        ensemble_size = algorithm.parameterAsInt(parameters, "ENSEMBLE_SIZE", context)
    except Exception:
        ensemble_size = 3

    all_models_p4 = []

    for idx in sel_idx:
        name, model_inst, default_params = get_model_and_params(idx, opt_idx, random_state, n_jobs)
        if model_inst is None:
            append_log(f"       ! Skipping {name}: Library is not installed.", log_path, feedback)
            continue

        if name in custom_params and custom_params[name]:
            parsed_dict = custom_params[name]
            base_params = {k: (v[0] if isinstance(v, list) and len(v)>0 else v) for k, v in parsed_dict.items()}
            model_inst.set_params(**base_params)

            if opt_idx == 2 and SKOPT_AVAILABLE:
                params = convert_to_bayes(parsed_dict)
            else:
                params = parsed_dict
        else:
            params = default_params

        try:
            curr_model = model_inst
            with joblib.parallel_backend("threading", n_jobs=n_jobs):
                if params and n_iter > 0:
                    search = None
                    current_opt_idx = opt_idx
                    if name == "MLP (Neural Net)":
                        current_opt_idx = 0

                    cv_splitter = 3
                    if groups_tr is not None:
                        n_unique_groups = len(np.unique(groups_tr))
                        if n_unique_groups >= 2:
                            from sklearn.model_selection import GroupKFold
                            cv_splitter = GroupKFold(n_splits=min(3, n_unique_groups))
                        else:
                            groups_tr = None

                    if current_opt_idx == 0:
                        import scipy.stats as stats
                        opt_params = {}
                        for k, v in params.items():
                            if isinstance(v, list) and len(v) == 2:
                                if all(isinstance(x, int) for x in v) and v[0] < v[1]:
                                    opt_params[k] = stats.randint(v[0], v[1] + 1)
                                elif all(isinstance(x, (int, float)) for x in v) and v[0] < v[1]:
                                    opt_params[k] = stats.uniform(v[0], v[1] - v[0])
                                else:
                                    opt_params[k] = v
                            else:
                                opt_params[k] = v
                        search = RandomizedSearchCV(
                            model_inst, opt_params, n_iter=n_iter, cv=cv_splitter, n_jobs=n_jobs, random_state=random_state
                        )
                    elif current_opt_idx == 1:
                        opt_params = {}
                        for k, v in params.items():
                            if isinstance(v, list) and len(v) == 2:
                                if all(isinstance(x, int) for x in v) and v[0] < v[1]:
                                    opt_params[k] = list(np.linspace(v[0], v[1], 5, dtype=int))
                                elif all(isinstance(x, (int, float)) for x in v) and v[0] < v[1]:
                                    opt_params[k] = list(np.linspace(v[0], v[1], 5))
                                else:
                                    opt_params[k] = v
                            else:
                                opt_params[k] = v
                        search = GridSearchCV(model_inst, opt_params, cv=cv_splitter, n_jobs=n_jobs)
                    elif current_opt_idx == 2:
                        if OPTUNA_AVAILABLE:
                            best_m, best_params = run_optuna_search(
                                model_inst, name, params, X_train, y_train, {},
                                n_iter=n_iter, cv=3, random_state=random_state, n_jobs=n_jobs,
                                groups=groups_tr
                            )
                            curr_model = best_m
                            curr_model.fit(X_train, y_train)
                            search = None
                        else:
                            import scipy.stats as stats
                            opt_params = {}
                            for k, v in params.items():
                                if isinstance(v, list) and len(v) == 2:
                                    if all(isinstance(x, int) for x in v) and v[0] < v[1]:
                                        opt_params[k] = stats.randint(v[0], v[1] + 1)
                                    elif all(isinstance(x, (int, float)) for x in v) and v[0] < v[1]:
                                        opt_params[k] = stats.uniform(v[0], v[1] - v[0])
                                    else:
                                        opt_params[k] = v
                                else:
                                    opt_params[k] = v
                            search = (
                                BayesSearchCV(model_inst, params, n_iter=n_iter, cv=cv_splitter, n_jobs=n_jobs)
                                if SKOPT_AVAILABLE
                                else RandomizedSearchCV(
                                    model_inst, opt_params, n_iter=n_iter, cv=cv_splitter, n_jobs=n_jobs, random_state=random_state
                                )
                            )

                    if search:
                        if groups_tr is not None:
                            search.fit(X_train, y_train, groups=groups_tr)
                        else:
                            search.fit(X_train, y_train)
                        curr_model = search.best_estimator_
                    elif not OPTUNA_AVAILABLE or current_opt_idx != 2:
                        curr_model.fit(X_train, y_train)
                else:
                    curr_model.fit(X_train, y_train)

            y_pred = curr_model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            sum_abs_diff = np.sum(np.abs(y_val - y_pred))
            sum_abs_true = np.sum(np.abs(y_val))
            wmape = (sum_abs_diff / sum_abs_true) * 100 if sum_abs_true != 0 else 0

            append_log(f"       > {name}: RMSE={rmse:.3f}m", log_path, feedback)
            all_models_p4.append({
                "Algorithm": name,
                "Model": curr_model,
                "RMSE": rmse,
                "R2": r2,
                "wMAPE": wmape
            })

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = curr_model
                best_algo_name = name
                best_r2 = r2
                best_wmape = wmape

        except Exception as e:
            append_log(f"Error in {name}: {str(e)}", log_path, feedback)

    if enable_ensemble and len(all_models_p4) >= 2:
        sorted_p4 = sorted(all_models_p4, key=lambda x: x["RMSE"])
        top_p4 = sorted_p4[:ensemble_size]
        estimators_p4 = [(m["Algorithm"], m["Model"]) for m in top_p4]
        append_log(f"   [Ensemble] Blending top models: {[m['Algorithm'] for m in top_p4]} using {ensemble_method}", log_path, feedback)

        ensemble_model = CustomEnsembleRegressor(estimators=estimators_p4, method=ensemble_method)
        ensemble_model.fit(X_train, y_train)
        
        y_pred = ensemble_model.predict(X_val)
        ens_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        ens_r2 = r2_score(y_val, y_pred)
        sum_abs_diff = np.sum(np.abs(y_val - y_pred))
        sum_abs_true = np.sum(np.abs(y_val))
        ens_wmape = (sum_abs_diff / sum_abs_true) * 100 if sum_abs_true != 0 else 0

        if ens_rmse < best_rmse:
            append_log(f"      > Ensemble ({ensemble_method}) wins! RMSE={ens_rmse:.3f}m (beats {best_algo_name} with RMSE={best_rmse:.3f}m)", log_path, feedback)
            best_rmse = ens_rmse
            best_model = ensemble_model
            best_algo_name = f"Ensemble ({ensemble_method})"
            best_r2 = ens_r2
            best_wmape = ens_wmape
        else:
            append_log(f"      > Ensemble ({ensemble_method}) did not beat winner {best_algo_name} (Ensemble RMSE={ens_rmse:.3f}m vs Winner RMSE={best_rmse:.3f}m)", log_path, feedback)

    if best_model is None:
        raise QgsProcessingException("All refinement models failed.")

    append_log(f"   Predicting Final Map using {best_algo_name}...", log_path, feedback)
    X_map = stack[:, water_indices[0], water_indices[1]].T
    X_map = np.nan_to_num(X_map, nan=0.0)
    
    if selected_indices is not None and len(selected_indices) > 0:
        X_map = X_map[:, selected_indices]
        
    z_out = best_model.predict(X_map)

    final_map = np.full((h, w), -9999.0, dtype="float32")
    final_map[water_indices] = z_out

    if med_size > 0 and scipy_is_available:
        valid = final_map != -9999
        temp = final_map.copy()
        temp[~valid] = np.nan
        filtered = median_filter(temp, size=med_size)
        final_map[valid] = filtered[valid]

    nodata_val = -9999.0
    if output_format == "uint16":
        valid = final_map != -9999.0
        final_map[valid] = np.clip(final_map[valid], 0, None)
        final_map[~valid] = 65535
        final_map = final_map.astype("uint16")
        nodata_val = 65535.0
    elif output_format == "float64":
        final_map = final_map.astype("float64")
    else:
        final_map = final_map.astype("float32")

    p_final = os.path.join(out_dir, "4-Refined_Model.tif")
    meta.update(count=1, dtype=output_format, nodata=nodata_val)
    with rasterio.open(p_final, "w", **meta) as dst:
        dst.write(final_map, 1)

    return {
        "OUTPUT_FINAL": p_final,
        "OUTPUT_RESIDUAL": p_residual,
        "BEST_R2": best_r2,
        "BEST_RMSE": best_rmse,
        "BEST_WMAPE": best_wmape,
    }
