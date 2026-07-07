import datetime
import os

import processing
from qgis.core import (
    QgsProcessingContext,
    QgsProcessingException,
    QgsProject,
    QgsRasterLayer,
)

from ..infrastructure.logging import append_log
from ..infrastructure.raster_io import (
    clean_depth_map,
    remove_positive_pixels,
    slope_filter_depth,
)
from ..infrastructure.vector_io import filter_by_depth, reproject_layer_if_needed


def run_master_pipeline(algorithm, parameters, context, feedback):
    """Execute SDB Master orchestration; `algorithm` is SDBMasterOrchestrator."""
    out_dir = algorithm.parameterAsString(parameters, algorithm.OUTPUT_FOLDER, context)
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "SDB_Full_Log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"SDB LOG - {datetime.datetime.now()}\n\n")

    append_log(">>> Workflow Started...", log_path, feedback)

    input_raster = algorithm.parameterAsRasterLayer(
        parameters, algorithm.INPUT_RASTER, context
    )
    target_crs = input_raster.crs()
    final_water_mask = None

    max_depth = algorithm.parameterAsDouble(
        parameters, algorithm.MAX_DEPTH_THRESHOLD, context
    )
    shrink_dist = algorithm.parameterAsDouble(
        parameters, algorithm.SHRINK_EDGE_DIST, context
    )

    remove_positives_flag = algorithm.parameterAsBool(
        parameters, algorithm.REMOVE_POSITIVES, context
    )
    apply_slope_filter = algorithm.parameterAsBool(
        parameters, algorithm.ENABLE_SLOPE_FILTER, context
    )
    slope_threshold_val = algorithm.parameterAsDouble(
        parameters, algorithm.SLOPE_THRESHOLD, context
    )

    water_mask_poly = algorithm.parameterAsVectorLayer(
        parameters, algorithm.WATER_MASK_POLY, context
    )
    if water_mask_poly:
        append_log(
            "\n>>> Pre-Clipping: Applying Ready-made Water Mask Polygon...",
            log_path,
            feedback,
        )

        temp_mask_path = os.path.join(out_dir, "temp_water_mask.gpkg")
        final_water_mask = reproject_layer_if_needed(
            water_mask_poly, target_crs, temp_mask_path, context, feedback
        )

        fixed_mask_path = os.path.join(out_dir, "temp_water_mask_fixed.gpkg")
        fix_res = processing.run(
            "native:fixgeometries",
            {"INPUT": final_water_mask, "OUTPUT": fixed_mask_path},
            context=context,
            feedback=feedback,
            is_child_algorithm=True,
        )
        final_water_mask = fix_res["OUTPUT"]

        if shrink_dist < 0:
            append_log(
                f">>> Shrinking Water Polygon by {shrink_dist} units to remove Edge Effects...",
                log_path,
                feedback,
            )
            shrunk_path = os.path.join(out_dir, "temp_water_mask_shrunk.gpkg")
            buffer_res = processing.run(
                "native:buffer",
                {
                    "INPUT": final_water_mask,
                    "DISTANCE": shrink_dist,
                    "SEGMENTS": 5,
                    "END_CAP_STYLE": 0,
                    "JOIN_STYLE": 0,
                    "MITER_LIMIT": 2,
                    "DISSOLVE": False,
                    "OUTPUT": shrunk_path,
                },
                context=context,
                feedback=feedback,
                is_child_algorithm=True,
            )
            final_water_mask = buffer_res["OUTPUT"]



    field_depth = algorithm.parameterAsString(
        parameters, algorithm.FIELD_DEPTH, context
    )
    temp_train = os.path.join(out_dir, "temp_reprojected_train.gpkg")

    final_train = reproject_layer_if_needed(
        algorithm.parameterAsVectorLayer(parameters, algorithm.INPUT_TRAIN, context),
        target_crs,
        temp_train,
        context,
        feedback,
    )
    final_train = filter_by_depth(
        final_train, field_depth, max_depth, context, feedback
    )

    enable_val = algorithm.parameterAsBool(
        parameters, algorithm.ENABLE_VALIDATION, context
    )
    final_test = None

    if enable_val:
        t_layer = algorithm.parameterAsVectorLayer(
            parameters, algorithm.INPUT_TEST, context
        )
        if t_layer:
            temp_test = os.path.join(out_dir, "temp_reprojected_test.gpkg")
            final_test = reproject_layer_if_needed(
                t_layer, target_crs, temp_test, context, feedback
            )
            final_test = filter_by_depth(
                final_test,
                algorithm.parameterAsString(
                    parameters, algorithm.FIELD_TEST_DEPTH, context
                ),
                max_depth,
                context,
                feedback,
            )
        else:
            enable_val = False

    append_log("\n>>> Phase 01: Pre-processing...", log_path, feedback)
    p1 = processing.run(
        "sdb_tools:sdb_phase1_preprocessing",
        {
            "INPUT_RASTER": input_raster,
            "COASTAL_BAND": parameters[algorithm.COASTAL_BAND],
            "BLUE_BAND": parameters[algorithm.BLUE_BAND],
            "GREEN_BAND": parameters[algorithm.GREEN_BAND],
            "RED_BAND": parameters[algorithm.RED_BAND],
            "NIR_BAND": parameters[algorithm.NIR_BAND],
            "SWIR_BAND": parameters[algorithm.SWIR_BAND],
            "APPLY_SUNGLINT": parameters[algorithm.APPLY_SUNGLINT],
            "NIR_BAND_SUNGLINT": parameters[algorithm.NIR_BAND_SUNGLINT],
            "SUNGLINT_PERCENTILE": parameters[algorithm.SUNGLINT_PERCENTILE],
            "INPUT_WATER_POLY": final_water_mask if water_mask_poly else None,
            "ENABLE_MASKING": parameters[algorithm.ENABLE_MASKING],
            "MASKING_METHOD": parameters[algorithm.MASKING_METHOD],
            "MANUAL_THRESHOLD": parameters[algorithm.MANUAL_THRESHOLD],
            "OTSU_ADJUSTMENT": parameters[algorithm.OTSU_ADJUSTMENT],
            "MASK_KERNEL_SIZE": parameters[algorithm.MASK_KERNEL_SIZE],
            "FEATURE_SELECTION": parameters[algorithm.FEATURE_SELECTION],
            "ENABLE_BAND_CALC": parameters[algorithm.ENABLE_BAND_CALC],
            "BAND_MATH_FORMULA": parameters[algorithm.BAND_MATH_FORMULA],
            "APPLY_DEEPWATER": parameters[algorithm.APPLY_DEEPWATER],
            "DEEPWATER_METHOD": parameters[algorithm.DEEPWATER_METHOD],
            "DEEPWATER_ROI": parameters.get(algorithm.DEEPWATER_ROI, None),
            "NIR_PERCENTILE_OSW": parameters[algorithm.NIR_PERCENTILE_OSW],
            "OSW_MEDIAN_SIZE": parameters[algorithm.OSW_MEDIAN_SIZE],
            "FILL_INTERNAL_HOLES": parameters.get(algorithm.FILL_INTERNAL_HOLES, True),
            "EXTRACT_POLYGON": parameters.get(algorithm.EXTRACT_POLYGON, True),
            "NUM_THREADS": parameters[algorithm.NUM_THREADS],
            "OUTPUT_FOLDER": out_dir,
        },
        context=context,
        feedback=feedback,
        is_child_algorithm=True,
    )


    path_clean = final_train
    if algorithm.parameterAsBool(parameters, algorithm.ENABLE_RANSAC, context):
        append_log("\n>>> Phase 02: Filtering & Uncertainty...", log_path, feedback)
        p2 = processing.run(
            "sdb_tools:sdb_02_filtering",
            {
                "INPUT_STACK": p1["OUTPUT_FEATURES"],
                "INPUT_POINTS": final_train,
                "FIELD_DEPTH": field_depth,
                "BLUE_BAND": parameters[algorithm.BLUE_BAND],
                "GREEN_BAND": parameters[algorithm.GREEN_BAND],
                "FILTER_MODE": parameters[algorithm.FILTER_MODE],
                "RESIDUAL_THRESHOLD": parameters[algorithm.RANSAC_THRESHOLD],
                "RANSAC_MAX_TRIALS": parameters[algorithm.RANSAC_MAX_TRIALS],
                "OUTPUT_FOLDER": out_dir,
            },
            context=context,
            feedback=feedback,
            is_child_algorithm=True,
        )
        path_clean = p2["OUTPUT_CLEAN_VEC"]

    append_log("\n>>> Phase 03: Global Modeling...", log_path, feedback)
    p3_params = {
        "INPUT_STACK": p1["OUTPUT_FEATURES"],
        "INPUT_POINTS": path_clean,
        "FIELD_DEPTH": field_depth,
        "FIELD_WEIGHT": algorithm.parameterAsString(
            parameters, algorithm.FIELD_WEIGHT, context
        ),
        "SELECTED_ALGOS": parameters[algorithm.SELECTED_ALGOS],
        "OPTIMIZER_METHOD": parameters[algorithm.OPTIMIZER_METHOD],
        "COLLISION_HANDLING": parameters[algorithm.COLLISION_HANDLING],
        "N_ITERATIONS": parameters[algorithm.N_ITERATIONS],
        "MEDIAN_SIZE": parameters[algorithm.MEDIAN_SIZE],
        "FEATURE_CORR_THRESHOLD": parameters.get(algorithm.FEATURE_CORR_THRESHOLD, 2),
        "FEATURE_CORR_METHOD": parameters.get(algorithm.FEATURE_CORR_METHOD, 0),
        "OUTPUT_FOLDER": out_dir,
        "LOG_FILE": log_path,
        "PARAM_RF": parameters[algorithm.PARAM_RF],
        "PARAM_GB": parameters[algorithm.PARAM_GB],
        "PARAM_ET": parameters[algorithm.PARAM_ET],
        "PARAM_SVR": parameters[algorithm.PARAM_SVR],
        "PARAM_MLP": parameters[algorithm.PARAM_MLP],
        "PARAM_RIDGE": parameters.get(algorithm.PARAM_RIDGE, ""),
        "PARAM_LASSO": parameters.get(algorithm.PARAM_LASSO, ""),
        "PARAM_ELASTICNET": parameters.get(algorithm.PARAM_ELASTICNET, ""),
        "PARAM_KNN": parameters.get(algorithm.PARAM_KNN, ""),
        "PARAM_DT": parameters.get(algorithm.PARAM_DT, ""),
        "PARAM_HUBER": parameters.get(algorithm.PARAM_HUBER, ""),
        "PARAM_XGB": parameters.get(algorithm.PARAM_XGB, ""),
        "PARAM_LGBM": parameters.get(algorithm.PARAM_LGBM, ""),
        "PARAM_CATBOOST": parameters.get(algorithm.PARAM_CATBOOST, ""),
        "ENABLE_ENSEMBLE": parameters.get(algorithm.ENABLE_ENSEMBLE, False),
        "ENSEMBLE_METHOD": parameters.get(algorithm.ENSEMBLE_METHOD, 0),
        "ENSEMBLE_SIZE": parameters.get(algorithm.ENSEMBLE_SIZE, 3),
        "SPATIAL_CV": parameters.get(algorithm.SPATIAL_CV_P3, False),
        "TRAIN_TEST_SPLIT": parameters[algorithm.TRAIN_TEST_SPLIT],
        "RANDOM_STATE": parameters[algorithm.RANDOM_STATE],
        "NUM_THREADS": parameters[algorithm.NUM_THREADS],
        "OUTPUT_FORMAT": parameters[algorithm.OUTPUT_FORMAT],
    }
    if p1.get("OUTPUT_MASK"):
        p3_params["INPUT_MASK"] = p1["OUTPUT_MASK"]

    p3 = processing.run(
        "sdb_tools:sdb_03_initial_modeling",
        p3_params,
        context=context,
        feedback=feedback,
        is_child_algorithm=True,
    )

    if "BEST_R2" in p3:
        append_log(f"[Phase 03] R2: {p3['BEST_R2']:.4f}", log_path, feedback)

    path_refined = None
    if algorithm.parameterAsBool(parameters, algorithm.ENABLE_ADAPTIVE, context):
        append_log("\n>>> Phase 04: Adaptive Refinement...", log_path, feedback)

        ad_layer = algorithm.parameterAsVectorLayer(
            parameters, algorithm.INPUT_ADAPTIVE_TRAIN, context
        )
        temp_adapt = os.path.join(out_dir, "temp_reprojected_adaptive.gpkg")
        final_ad = reproject_layer_if_needed(
            ad_layer, target_crs, temp_adapt, context, feedback
        )
        field_ad_depth = algorithm.parameterAsString(
            parameters, algorithm.FIELD_ADAPTIVE_DEPTH, context
        )
        final_ad = filter_by_depth(
            final_ad, field_ad_depth, max_depth, context, feedback
        )

        p4_params = {
            "INPUT_GLOBAL_RASTER": p3["OUTPUT_DEPTH_MAP"],
            "INPUT_ORIGINAL_FEAT": p1["OUTPUT_FEATURES"],
            "INPUT_TRAIN": final_ad,
            "FIELD_TRAIN": field_ad_depth,
            "SELECTED_ALGOS": parameters[algorithm.SELECTED_ALGOS],
            "OPTIMIZER_METHOD": parameters[algorithm.OPTIMIZER_METHOD],
            "COLLISION_HANDLING": parameters[algorithm.COLLISION_HANDLING],
            "N_ITERATIONS": parameters[algorithm.N_ITERATIONS],
            "MEDIAN_SIZE": parameters[algorithm.MEDIAN_SIZE],
            "FEATURE_CORR_THRESHOLD": parameters.get(algorithm.FEATURE_CORR_THRESHOLD, 2),
            "FEATURE_CORR_METHOD": parameters.get(algorithm.FEATURE_CORR_METHOD, 0),
            "OUTPUT_FOLDER": out_dir,
            "LOG_FILE": log_path,
            "PARAM_RF": parameters[algorithm.PARAM_RF],
            "PARAM_GB": parameters[algorithm.PARAM_GB],
            "PARAM_ET": parameters[algorithm.PARAM_ET],
            "PARAM_SVR": parameters[algorithm.PARAM_SVR],
            "PARAM_MLP": parameters[algorithm.PARAM_MLP],
            "PARAM_RIDGE": parameters.get(algorithm.PARAM_RIDGE, ""),
            "PARAM_LASSO": parameters.get(algorithm.PARAM_LASSO, ""),
            "PARAM_ELASTICNET": parameters.get(algorithm.PARAM_ELASTICNET, ""),
            "PARAM_KNN": parameters.get(algorithm.PARAM_KNN, ""),
            "PARAM_DT": parameters.get(algorithm.PARAM_DT, ""),
            "PARAM_HUBER": parameters.get(algorithm.PARAM_HUBER, ""),
            "PARAM_XGB": parameters.get(algorithm.PARAM_XGB, ""),
            "PARAM_LGBM": parameters.get(algorithm.PARAM_LGBM, ""),
            "PARAM_CATBOOST": parameters.get(algorithm.PARAM_CATBOOST, ""),
            "ENABLE_ENSEMBLE": parameters.get(algorithm.ENABLE_ENSEMBLE_P4, False),
            "ENSEMBLE_METHOD": parameters.get(algorithm.ENSEMBLE_METHOD_P4, 0),
            "ENSEMBLE_SIZE": parameters.get(algorithm.ENSEMBLE_SIZE_P4, 3),
            "RESIDUAL_INTERP_METHOD": parameters.get(algorithm.RESIDUAL_INTERP_METHOD, 0),
            "KNN_NEIGHBORS": parameters.get(algorithm.KNN_NEIGHBORS, 15),
            "SPATIAL_CV": parameters.get(algorithm.SPATIAL_CV_P4, False),
            "TRAIN_TEST_SPLIT": parameters[algorithm.TRAIN_TEST_SPLIT],
            "RANDOM_STATE": parameters[algorithm.RANDOM_STATE],
            "NUM_THREADS": parameters[algorithm.NUM_THREADS],
            "OUTPUT_FORMAT": parameters[algorithm.OUTPUT_FORMAT],
        }
        if p1.get("OUTPUT_MASK"):
            p4_params["INPUT_MASK"] = p1["OUTPUT_MASK"]
        
        p4 = processing.run(
            "sdb_tools:sdb_phase4_adaptive",
            p4_params,
            context=context,
            feedback=feedback,
            is_child_algorithm=True,
        )
        path_refined = p4["OUTPUT_FINAL"]

        if "BEST_R2" in p4:
            append_log(f"[Phase 04] R2: {p4['BEST_R2']:.4f}", log_path, feedback)

    feat_stack = p1["OUTPUT_FEATURES"]

    if p3.get("OUTPUT_DEPTH_MAP") and os.path.exists(p3["OUTPUT_DEPTH_MAP"]):
        p3_clamped = os.path.join(out_dir, "Phase3_Depth_Cleaned.tif")
        clean_depth_map(
            p3["OUTPUT_DEPTH_MAP"], feat_stack, max_depth, p3_clamped, context, feedback
        )

        if remove_positives_flag:
            p3_no_pos = os.path.join(out_dir, "Phase03_Depth_Final_NoPositives.tif")
            remove_positive_pixels(p3_clamped, p3_no_pos, feedback)
            current_p3 = p3_no_pos
        else:
            current_p3 = p3_clamped

        if p1.get("OUTPUT_OSW_POLY") and os.path.exists(p1["OUTPUT_OSW_POLY"]):
            append_log("\n>>> Clipping Phase 03 Map with OSW Polygon...", log_path, feedback)
            p3_osw_clipped = os.path.join(out_dir, "Phase03_Depth_OSW_Clipped.tif")
            processing.run(
                "gdal:cliprasterbymasklayer",
                {
                    "INPUT": current_p3,
                    "MASK": p1["OUTPUT_OSW_POLY"],
                    "SOURCE_CRS": target_crs,
                    "TARGET_CRS": target_crs,
                    "NODATA": -9999.0,
                    "ALPHA_BAND": False,
                    "CROP_TO_CUTLINE": False,
                    "KEEP_RESOLUTION": True,
                    "DATA_TYPE": 0,
                    "OUTPUT": p3_osw_clipped,
                },
                context=context,
                feedback=feedback,
                is_child_algorithm=True,
            )
            if os.path.exists(p3_osw_clipped):
                current_p3 = p3_osw_clipped

        p3["OUTPUT_DEPTH_MAP"] = current_p3

    if path_refined and os.path.exists(path_refined):
        p4_clamped = os.path.join(out_dir, "Final_Depth_Cleaned.tif")
        clean_depth_map(
            path_refined, feat_stack, max_depth, p4_clamped, context, feedback
        )

        if apply_slope_filter:
            slope_filtered = os.path.join(out_dir, "Final_Depth_SlopeFiltered.tif")
            path_refined = slope_filter_depth(
                p4_clamped,
                slope_threshold=slope_threshold_val,
                out_path=slope_filtered,
                context=context,
                feedback=feedback,
            )
        else:
            path_refined = p4_clamped

        if remove_positives_flag:
            p4_no_pos = os.path.join(out_dir, "Phase04_Final_Depth_NoPositives.tif")
            remove_positive_pixels(path_refined, p4_no_pos, feedback)
            path_refined = p4_no_pos

        if p1.get("OUTPUT_OSW_POLY") and os.path.exists(p1["OUTPUT_OSW_POLY"]):
            append_log("\n>>> Clipping Phase 04 Map with OSW Polygon...", log_path, feedback)
            p4_osw_clipped = os.path.join(out_dir, "Phase04_Final_Depth_OSW_Clipped.tif")
            processing.run(
                "gdal:cliprasterbymasklayer",
                {
                    "INPUT": path_refined,
                    "MASK": p1["OUTPUT_OSW_POLY"],
                    "SOURCE_CRS": target_crs,
                    "TARGET_CRS": target_crs,
                    "NODATA": -9999.0,
                    "ALPHA_BAND": False,
                    "CROP_TO_CUTLINE": False,
                    "KEEP_RESOLUTION": True,
                    "DATA_TYPE": 0,
                    "OUTPUT": p4_osw_clipped,
                },
                context=context,
                feedback=feedback,
                is_child_algorithm=True,
            )
            if os.path.exists(p4_osw_clipped):
                path_refined = p4_osw_clipped

    if enable_val and final_test:
        append_log("\n>>> Phase 05: Validation...", log_path, feedback)
        processing.run(
            "sdb_tools:sdb_05_reporting",
            {
                "INPUT_MAP_P3": p3["OUTPUT_DEPTH_MAP"],
                "INPUT_MAP_P4": path_refined if path_refined else p3["OUTPUT_DEPTH_MAP"],
                "INPUT_TRAIN": path_clean,
                "FIELD_TRAIN": field_depth,
                "INPUT_VALIDATION": final_test,
                "FIELD_VAL_DEPTH": algorithm.parameterAsString(
                    parameters, algorithm.FIELD_TEST_DEPTH, context
                ),
                "OUTPUT_FOLDER": out_dir,
            },
            context=context,
            feedback=feedback,
            is_child_algorithm=True,
        )

    if p3.get("OUTPUT_DEPTH_MAP") and os.path.exists(p3["OUTPUT_DEPTH_MAP"]):
        details_init = QgsProcessingContext.LayerDetails(
            "Initial SDB Map [Phase 03]", QgsProject.instance(), "Initial SDB"
        )
        context.addLayerToLoadOnCompletion(p3["OUTPUT_DEPTH_MAP"], details_init)

    if path_refined and os.path.exists(path_refined):
        details_ref = QgsProcessingContext.LayerDetails(
            "Refined SDB Map [Phase 04]", QgsProject.instance(), "Refined SDB"
        )
        context.addLayerToLoadOnCompletion(path_refined, details_ref)

    append_log("\n>>> Workflow Complete.", log_path, feedback)

    return {}
