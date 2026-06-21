import ast
import os
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from qgis.PyQt.QtCore import QVariant
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
            if all(isinstance(x, int) for x in v):
                bayes_params[k] = Integer(min(v), max(v))
            elif all(isinstance(x, (int, float)) for x in v):
                bayes_params[k] = Real(min(v), max(v))
            else:
                bayes_params[k] = Categorical(v)
        else:
            bayes_params[k] = Categorical(v)
    return bayes_params


def save_training_points(out_path, coords, depths, weights, X_data, ref_raster, crs):
    fields = QgsFields()
    fields.append(QgsField("Depth_Used", QVariant.Double))
    fields.append(QgsField("Weight_Used", QVariant.Double))
    fields.append(QgsField("Row_Idx", QVariant.Int))
    fields.append(QgsField("Col_Idx", QVariant.Int))

    num_bands = X_data.shape[1]
    for b in range(num_bands):
        fields.append(QgsField(f"Band_{b + 1}", QVariant.Double))

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


def extract_samples(ras_path, vec_layer, d_fld, w_fld, mode):
    rlayer = QgsRasterLayer(ras_path)
    tr = QgsCoordinateTransform(
        vec_layer.sourceCrs(), rlayer.crs(), QgsProject.instance()
    )
    X_out, y_out, w_out, c_out = [], [], [], []

    with rasterio.open(ras_path) as src:
        d = src.read()
        h, w = src.height, src.width
        rst_transform = src.transform
        pixel_size = abs(src.res[0])

        if mode == 0:
            for f in vec_layer.getFeatures():
                geom = f.geometry()
                geom.transform(tr)
                pt = geom.asPoint()
                try:
                    r, c = src.index(pt.x(), pt.y())
                    if 0 <= r < h and 0 <= c < w:
                        val = d[:, r, c]
                        if np.all(np.isfinite(val)) and not np.any(val == -9999):
                            X_out.append(val)
                            y_out.append(f[d_fld])
                            c_out.append([r, c])
                            w_out.append(f[w_fld] if w_fld else 1.0)
                except IndexError:
                    continue
            return np.array(X_out), np.array(y_out), np.array(w_out), c_out

        pixel_registry = {}
        for f in vec_layer.getFeatures():
            geom = f.geometry()
            geom.transform(tr)
            pt = geom.asPoint()
            try:
                r, c = src.index(pt.x(), pt.y())
                if 0 <= r < h and 0 <= c < w:
                    pixel_registry.setdefault((r, c), []).append(
                        {"d": f[d_fld], "w": f[w_fld] if w_fld else 1.0, "pt": pt}
                    )
            except IndexError:
                continue

        for (r, c), items in pixel_registry.items():
            val = d[:, r, c]
            if not np.all(np.isfinite(val)) or np.any(val == -9999):
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


def save_algo_artifacts(y_t, y_p, pct, name, folder, r2, rmse, mape, params):
    with open(os.path.join(folder, "Results.txt"), "w") as f:
        f.write(
            f"Algo: {name}\nR2: {r2:.4f}\nRMSE: {rmse:.4f}\nwMAPE: {mape:.2f}%\nParams: {params}"
        )


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


def run_benchmarking(
    X, y, weights, indices, n_iter, out_dir, feedback, opt_idx, log_path, custom_params,
    test_size=0.2, random_state=42, n_jobs=-1
):
    X = np.nan_to_num(X, nan=0.0)
    X_tr, X_val, y_tr, y_val, w_tr, _ = train_test_split(
        X, y, weights, test_size=test_size, random_state=random_state
    )

    results = []
    for idx in [int(i) for i in indices]:
        name, base_model, default_params = get_model_and_params(idx, opt_idx, random_state, n_jobs)

        if name in custom_params and custom_params[name]:
            parsed_dict = custom_params[name]
            # Extract base parameters
            base_params = {k: (v[0] if isinstance(v, list) and len(v)>0 else v) for k, v in parsed_dict.items()}
            base_model.set_params(**base_params)

            if opt_idx == 2 and SKOPT_AVAILABLE:
                params = convert_to_bayes(parsed_dict)
            else:
                params = parsed_dict
        else:
            params = default_params

        algo_dir = os.path.join(out_dir, name.replace(" ", "_"))
        os.makedirs(algo_dir, exist_ok=True)

        try:
            model = base_model
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
            ]:
                fit_params["sample_weight"] = w_tr

            with joblib.parallel_backend("threading", n_jobs=n_jobs):
                if params and n_iter > 0:
                    search = None
                    current_opt_idx = opt_idx
                    if name == "MLP (Neural Net)":
                        current_opt_idx = 0
                    if current_opt_idx == 0:
                        search = RandomizedSearchCV(
                            base_model, params, n_iter=n_iter, cv=3, n_jobs=n_jobs
                        )
                    elif current_opt_idx == 1:
                        search = GridSearchCV(base_model, params, cv=3, n_jobs=n_jobs)
                    elif current_opt_idx == 2:
                        search = (
                            BayesSearchCV(base_model, params, n_iter=n_iter, cv=3, n_jobs=n_jobs)
                            if SKOPT_AVAILABLE
                            else RandomizedSearchCV(
                                base_model, params, n_iter=n_iter, cv=3, n_jobs=n_jobs
                            )
                        )

                    if search:
                        search.fit(X_tr, y_tr, **fit_params)
                        model = search.best_estimator_
                        params_str = str(search.best_params_)
                    else:
                        model.fit(X_tr, y_tr, **fit_params)
                        params_str = "Default"
                else:
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
            )
            results.append(
                {
                    "Algorithm": name,
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
    df = pd.DataFrame(results)
    df["score"] = (0.6 * df["R2"].clip(lower=0)) + (
        0.4 * (1 - (df["RMSE"] / df["RMSE"].max()))
    )
    winner = df.loc[df["score"].idxmax()]
    final_model = winner["Model"]
    fit_params_final = {}
    if winner["Algorithm"] in [
        "Random Forest",
        "Gradient Boosting",
        "Extra Trees",
        "Ridge",
        "Lasso",
        "Decision Tree",
        "SVR",
        "Linear Regression",
    ]:
        fit_params_final["sample_weight"] = weights
    with joblib.parallel_backend("threading", n_jobs=n_jobs):
        final_model.fit(X, y, **fit_params_final)
    return df, {
        "name": winner["Algorithm"],
        "model": final_model,
        "score": winner["score"],
        "r2": winner["R2"],
        "rmse": winner["RMSE"],
        "wmape": winner["wMAPE"],
    }


def predict_map(model, stack_path, mask_path, out_path, med_size, output_format="float32"):
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
    X_pixels = np.nan_to_num(d_flat[water_idx], nan=0.0)
    preds = model.predict(X_pixels)
    out_img = np.full(h * w, -9999.0, dtype="float32")
    out_img[water_idx] = preds
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
    save_training_points(
        actual_pts_path,
        coords,
        y,
        final_weights,
        X,
        stack_path,
        points_layer.sourceCrs(),
    )
    # Direct layer addition removed to prevent auto-loading in panel.
    # The output is returned to the processing framework instead.
    # QgsProject.instance().addMapLayer(QgsVectorLayer(actual_pts_path, "3_Actual_Model_Input_Points", "ogr"))

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
        n_jobs
    )
    results_df.to_csv(
        os.path.join(out_dir, "3_All_Algorithms_Benchmark.csv"), index=False
    )

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
    predict_map(best_algo_data["model"], stack_path, mask_path, p_map, med_size, output_format)
    # Direct layer addition removed to prevent auto-loading in panel.
    # The output is returned to the processing framework instead.
    # QgsProject.instance().addMapLayer(QgsRasterLayer(p_map, f"3_Initial_Global_Depth ({win_name})"))

    return {
        "OUTPUT_DEPTH_MAP": p_map,
        "OUTPUT_MODEL_PKL": os.path.join(out_dir, "3_Best_Global_Model.pkl"),
        "BEST_R2": best_algo_data["r2"],
        "BEST_RMSE": best_algo_data["rmse"],
    }
