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
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

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
            if all(isinstance(x, int) for x in v):
                bayes_params[k] = Integer(min(v), max(v))
            elif all(isinstance(x, (int, float)) for x in v):
                bayes_params[k] = Real(min(v), max(v))
            else:
                bayes_params[k] = Categorical(v)
        else:
            bayes_params[k] = Categorical(v)
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
                else {"n_estimators": [100, 300]}
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
                else {"n_estimators": [100, 300]}
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

    return "Unknown", LinearRegression(), {}


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

    knn_spatial = KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=n_jobs)
    knn_spatial.fit(coords_tr, residuals)

    append_log("   Interpolating Residual Grid (KNN)...", log_path, feedback)
    
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
            residual_grid[chunk[:, 0], chunk[:, 1]] = knn_spatial.predict(chunk)

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

    X_train, X_val, y_train, y_val = train_test_split(
        X_final, y_final, test_size=test_size, random_state=random_state
    )

    best_rmse = float("inf")
    best_model = None
    best_algo_name = ""
    best_r2 = 0.0
    best_wmape = 0.0

    for idx in sel_idx:
        name, model_inst, default_params = get_model_and_params(idx, opt_idx, random_state, n_jobs)

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

                    if current_opt_idx == 0:
                        search = RandomizedSearchCV(
                            model_inst, params, n_iter=n_iter, cv=3, n_jobs=n_jobs
                        )
                    elif current_opt_idx == 1:
                        search = GridSearchCV(model_inst, params, cv=3, n_jobs=n_jobs)
                    elif current_opt_idx == 2:
                        search = (
                            BayesSearchCV(model_inst, params, n_iter=n_iter, cv=3, n_jobs=n_jobs)
                            if SKOPT_AVAILABLE
                            else RandomizedSearchCV(
                                model_inst, params, n_iter=n_iter, cv=3, n_jobs=n_jobs
                            )
                        )

                    if search:
                        search.fit(X_train, y_train)
                        curr_model = search.best_estimator_
                    else:
                        curr_model.fit(X_train, y_train)
                else:
                    curr_model.fit(X_train, y_train)

            y_pred = curr_model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))

            append_log(f"       > {name}: RMSE={rmse:.3f}m", log_path, feedback)

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = curr_model
                best_algo_name = name
                best_r2 = r2_score(y_val, y_pred)
                sum_abs_diff = np.sum(np.abs(y_val - y_pred))
                sum_abs_true = np.sum(np.abs(y_val))
                best_wmape = (
                    (sum_abs_diff / sum_abs_true) * 100 if sum_abs_true != 0 else 0
                )

        except Exception as e:
            append_log(f"Error in {name}: {str(e)}", log_path, feedback)

    if best_model is None:
        raise QgsProcessingException("All refinement models failed.")

    append_log(f"   Predicting Final Map using {best_algo_name}...", log_path, feedback)
    X_map = stack[:, water_indices[0], water_indices[1]].T
    X_map = np.nan_to_num(X_map, nan=0.0)
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
