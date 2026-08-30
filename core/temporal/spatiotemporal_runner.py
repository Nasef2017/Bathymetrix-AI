import os
import joblib
import numpy as np
try:
    from qgis.core import QgsProcessingException, QgsProject, QgsRasterLayer
    from qgis import processing
except ImportError:
    QgsProcessingException = Exception
    QgsProject = None
    QgsRasterLayer = None
    processing = None

try:
    from Bathymetrix_AI.infrastructure.logging import append_log
    from Bathymetrix_AI.core.ml.trainers import extract_samples, run_phase03_initial_modeling, predict_map
    from Bathymetrix_AI.core.temporal.temporal_reporting import TemporalReportGenerator
except (ImportError, ValueError):
    from infrastructure.logging import append_log
    from core.ml.trainers import extract_samples, run_phase03_initial_modeling, predict_map
    from core.temporal.temporal_reporting import TemporalReportGenerator


class SpatiotemporalSDBRunner:
    def __init__(self, master_output_folder):
        self.master_output_folder = master_output_folder

    def run_spatiotemporal_flow(self, yearly_datasets, masterflow_params, algorithm, context, feedback):
        import time
        start_time = time.time()
        start_str = time.strftime('%H:%M:%S', time.localtime(start_time))
        
        log_path = os.path.join(self.master_output_folder, "Spatiotemporal_Flow_Log.txt")
        append_log("════════════════════════════════════════════════════════════", log_path, feedback)
        append_log("SDB Spatiotemporal Masterflow".center(60), log_path, feedback)
        append_log("════════════════════════════════════════════════════════════", log_path, feedback)
        append_log(f"Started: {start_str}", log_path, feedback)
        append_log(f"Years detected: {len(yearly_datasets)}", log_path, feedback)
        append_log("════════════════════════════════════════════════════════════\n", log_path, feedback)
        
        # --- Pre-Scan Validation: Check if Depth Fields exist ---
        depth_field_train = masterflow_params.get("FIELD_DEPTH", "")
        depth_field_val = masterflow_params.get("FIELD_TEST_DEPTH", "") or depth_field_train
        depth_field_ctrl = masterflow_params.get("FIELD_ADAPTIVE_DEPTH", "") or depth_field_train
        
        append_log("→ Validating Depth Fields...", log_path, feedback)
        from qgis.core import QgsVectorLayer
        for year, year_info in yearly_datasets.items():
            train_src = year_info.get("icesat_file_path") or year_info.get("icesat_layer")
            unseen_src = year_info.get("unseen_file_path") or year_info.get("unseen_layer")
            ctrl_src = year_info.get("control_path")
            
            checks = [
                (train_src, "Training", depth_field_train),
                (unseen_src, "Validation", depth_field_val)
            ]
            if ctrl_src:
                checks.append((ctrl_src, "Control Points", depth_field_ctrl))
            
            for src, name, expected_field in checks:
                if src and expected_field:
                    layer = src if isinstance(src, QgsVectorLayer) else QgsVectorLayer(src, f"{name}_{year}", "ogr")
                    if layer and layer.isValid():
                        fields = layer.fields().names()
                        matched = False
                        for f in fields:
                            if f.lower() == expected_field.lower() or f.lower() == expected_field.lower()[:10] or expected_field.lower().startswith(f.lower()[:8]):
                                matched = True
                                break
                        if not matched:
                            err_msg = f"✗ ERROR: Depth field '{expected_field}' not found in {name} dataset for year {year}."
                            append_log(err_msg, log_path, feedback)
                            raise QgsProcessingException(err_msg)
        from qgis.core import QgsRasterLayer, QgsCoordinateTransform, QgsProject, QgsGeometry
        
        for year, year_info in yearly_datasets.items():
            # Check overlap between Image and Training Points
            img_src = year_info.get("image_path")
            train_src = year_info.get("icesat_file_path") or year_info.get("icesat_layer")
            
            if img_src and train_src:
                rl = QgsRasterLayer(img_src, f"raster_{year}")
                vl = train_src if isinstance(train_src, QgsVectorLayer) else QgsVectorLayer(train_src, f"train_{year}", "ogr")
                
                if rl.isValid() and vl.isValid():
                    r_crs = rl.crs()
                    v_crs = vl.crs()
                    
                    r_geom = QgsGeometry.fromRect(rl.extent())
                    v_geom = QgsGeometry.fromRect(vl.extent())
                    
                    if r_crs != v_crs:
                        try:
                            transform = QgsCoordinateTransform(r_crs, v_crs, QgsProject.instance())
                            r_geom.transform(transform)
                        except Exception:
                            pass
                            
                    if not r_geom.intersects(v_geom):
                            err_msg = f"✗ ERROR: Image '{year}' failed spatial overlap."
                            append_log(err_msg, log_path, feedback)
                            raise QgsProcessingException(err_msg)
        append_log("✓ Validation completed\n", log_path, feedback)
        # --- End Pre-Scan ---

        
        global_X = []
        global_y = []
        global_w = []
        global_coords = []
        global_feat_names = None
        
        # Keep track of outputs per year for Phase 4
        year_outputs = {}
        
        # ---------------------------------------------------------
        # LOOP 1: Phase 1 & 2 (Per-Image Preprocessing & Filtering)
        # ---------------------------------------------------------
        first_year = True
        master_osw_polygon = None
        
        for year, year_info in yearly_datasets.items():
            year_start = time.time()
            append_log(f"▶ Year {year} | Extraction Loop", log_path, feedback)
            append_log("  ──────────────────────────────────────────────────────────", log_path, feedback)
            if feedback.isCanceled(): return {}
            
            year_out_dir = os.path.join(self.master_output_folder, f"Year_{year}")
            p1_dir = os.path.join(year_out_dir, "Phase_01_Preprocessing")
            p2_dir = os.path.join(year_out_dir, "Phase_02_Filtering")
            
            os.makedirs(year_out_dir, exist_ok=True)
            os.makedirs(p1_dir, exist_ok=True)
            os.makedirs(p2_dir, exist_ok=True)
            
            # 2. Determine ICEsat layer
            icesat_path = None
            specific_file = year_info.get("icesat_file_path", None)
            
            if specific_file:
                # 2A. A specific file was found for this year (e.g. inside /2019/)
                icesat_path = specific_file
            else:
                # 2B. Fallback to the Global Layer, but FILTER it if a field was provided
                global_layer = year_info.get("icesat_layer", None)
                if not global_layer:
                    append_log(f"  ⚠ WARNING: No ICESat data found for year {year}. Skipping...", log_path, feedback)
                    continue
                    
                year_field = year_info.get("icesat_year_field", None)
                if year_field and year_field.strip():
                    try:
                        ext = processing.run("native:extractbyexpression", {
                            'INPUT': global_layer,
                            'EXPRESSION': f'"{year_field}" = {year} OR "{year_field}" = \'{year}\'',
                            'OUTPUT': 'TEMPORARY_OUTPUT'
                        }, context=context, feedback=feedback, is_child_algorithm=True)
                        icesat_path = ext['OUTPUT']
                    except Exception as e:
                        append_log(f"  ✗ ERROR: Failed to filter global ICESat for {year}: {e}", log_path, feedback)
                        continue
                else:
                    try:
                        icesat_path = global_layer.source()
                    except AttributeError:
                        icesat_path = global_layer
                        
            if not icesat_path:
                append_log(f"  ✗ ERROR: Failed to resolve ICESat path for year {year}. Skipping...", log_path, feedback)
                continue
            
            # 3. Setup run parameters
            run_params = masterflow_params.copy()
            
            # Map WATER_MASK_POLY to INPUT_WATER_POLY for Phase 1
            if "WATER_MASK_POLY" in run_params and "INPUT_WATER_POLY" not in run_params:
                run_params["INPUT_WATER_POLY"] = run_params["WATER_MASK_POLY"]
                
            run_params["INPUT_RASTER"] = year_info["image_path"]
            run_params["INPUT_TRAIN"] = icesat_path
            run_params["OUTPUT_FOLDER"] = p1_dir
            
            # 4. Phase 1
            enable_preproc = algorithm.parameterAsBool(masterflow_params, "ENABLE_PREPROCESSING", context) if (algorithm and hasattr(algorithm, "parameterAsBool")) else masterflow_params.get("ENABLE_PREPROCESSING", True)
            if enable_preproc:
                append_log("  [Phase 01] Pre-processing", log_path, feedback)
                if first_year:
                    append_log("      → Base Year Mode", log_path, feedback)
                    p1 = processing.run("sdb_tools:sdb_phase1_preprocessing", run_params, is_child_algorithm=True, context=context, feedback=feedback)
                    master_osw_polygon = p1.get("OUTPUT_OSW_POLY")
                    first_year = False
                else:
                    append_log("      → Enforcing Master OSW Polygon", log_path, feedback)
                    if master_osw_polygon:
                        run_params["APPLY_DEEPWATER"] = True
                        run_params["DEEPWATER_METHOD"] = 5  # Shallow Water Bound (OSW Polygon)
                        run_params["DEEPWATER_ROI"] = master_osw_polygon
                    p1 = processing.run("sdb_tools:sdb_phase1_preprocessing", run_params, is_child_algorithm=True, context=context, feedback=feedback)
                
                p1_feat = p1["OUTPUT_FEATURES"]
                append_log("  ✓ Phase 01 completed\n", log_path, feedback)
            else:
                append_log("  [Phase 01] Pre-processing", log_path, feedback)
                append_log("      → Skipped by User.\n", log_path, feedback)
                p1_feat = year_info["image_path"]
                p1 = {"OUTPUT_FEATURES": p1_feat, "OUTPUT_MASK": None, "OUTPUT_OSW_POLY": None}
            
            # 5. Phase 2
            has_points = False
            if icesat_path:
                try:
                    from qgis.core import QgsVectorLayer
                    tmp_layer = QgsVectorLayer(icesat_path, "tmp", "ogr")
                    if tmp_layer.isValid() and tmp_layer.featureCount() > 0:
                        has_points = True
                except Exception:
                    has_points = False
                    
            enable_p2 = algorithm.parameterAsBool(masterflow_params, "ENABLE_RANSAC", context) if (algorithm and hasattr(algorithm, "parameterAsBool")) else masterflow_params.get("ENABLE_RANSAC", True)
            
            p2_vec = None
            if not has_points:
                append_log(f"  ⚠ WARNING: No valid training points found for year {year}. Skipping Phase 02 and Training Extraction.", log_path, feedback)
            elif enable_p2:
                append_log("  [Phase 02] Filtering", log_path, feedback)
                p2_params = run_params.copy()
                p2_params["INPUT_STACK"] = p1_feat
                p2_params["INPUT_MASK"] = p1.get("OUTPUT_MASK")
                p2_params["INPUT_POINTS"] = icesat_path
                p2_params["BLUE_BAND"] = run_params.get("FILTER_NUMERATOR_BAND", run_params.get("BLUE_BAND"))
                p2_params["GREEN_BAND"] = run_params.get("FILTER_DENOMINATOR_BAND", run_params.get("GREEN_BAND"))
                p2_params["RESIDUAL_THRESHOLD"] = run_params.get("RANSAC_THRESHOLD", 3.0)
                p2_params["OUTPUT_FOLDER"] = p2_dir
                try:
                    p2 = processing.run("sdb_tools:sdb_02_filtering", p2_params, is_child_algorithm=True, context=context, feedback=feedback)
                    p2_vec = p2["OUTPUT_CLEAN_VEC"]
                    append_log("  ✓ Phase 02 completed\n", log_path, feedback)
                except Exception as e:
                    append_log(f"  ⚠ WARNING: Phase 02 Filtering failed: {e}", log_path, feedback)
            else:
                append_log("  [Phase 02] Filtering", log_path, feedback)
                append_log("      → Skipped by User.\n", log_path, feedback)
                p2_vec = icesat_path
            
            # 6. Extract Samples for Global Matrix
            if has_points and p2_vec:
                append_log("  [Sample Extraction]", log_path, feedback)
                append_log("      → Extracting Global Matrix Samples...", log_path, feedback)
                depth_fld = run_params["FIELD_DEPTH"]
                col_mode = run_params["COLLISION_HANDLING"]
                weight_fld = None
                
                try:
                    X_yr, y_yr, w_yr, coords_yr = extract_samples(p1_feat, p2_vec, depth_fld, weight_fld, col_mode)
                except Exception as e:
                    append_log(f"  ⚠ WARNING: Extraction failed: {e}", log_path, feedback)
                    X_yr, y_yr, w_yr, coords_yr = np.array([]), np.array([]), np.array([]), np.array([])
            else:
                X_yr, y_yr, w_yr, coords_yr = np.array([]), np.array([]), np.array([]), np.array([])
            
            if len(y_yr) < 10:
                append_log(f"  ⚠ WARNING: Insufficient or no training points ({len(y_yr)}) for year {year}. Skipping this year entirely from modeling and prediction.\n", log_path, feedback)
                continue
                
            # Add Year feature
            year_col = np.full((X_yr.shape[0], 1), int(year), dtype=X_yr.dtype)
            X_yr_extended = np.hstack((X_yr, year_col))
            
            global_X.append(X_yr_extended)
            global_y.append(y_yr)
            global_w.append(w_yr)
            global_coords.append(coords_yr)
            
            if global_feat_names is None:
                try:
                    import rasterio
                    with rasterio.open(p1_feat) as src:
                        f_names = [str(desc).strip() if desc and str(desc).strip() else f"Band_{i+1}" for i, desc in enumerate(src.descriptions)]
                        f_names.append("Year")
                        global_feat_names = f_names
                except Exception:
                    global_feat_names = [f"Band_{i+1}" for i in range(X_yr_extended.shape[1]-1)] + ["Year"]
            
            year_outputs[year] = {
                "P1_FEATURES": p1_feat,
                "P1_MASK": p1.get("OUTPUT_MASK"),
                "P2_VEC": p2_vec,
                "INFO": year_info,
                "X_yr": X_yr,
                "y_yr": y_yr
            }
            append_log(f"  ✓ Extracted {len(y_yr)} points for year {year}\n", log_path, feedback)
            
            scene_elapsed = time.time() - year_start
            m, s = divmod(int(scene_elapsed), 60)
            h, m = divmod(m, 60)
            append_log(f"✓ Year {year} extraction completed | {h:02d}:{m:02d}:{s:02d}\n", log_path, feedback)
            
        if not global_X:
            raise QgsProcessingException("No valid data extracted across all years. Cannot train global model.")
            
        # ---------------------------------------------------------
        # GLOBAL PHASE 3: Training on the Spatiotemporal Matrix
        # ---------------------------------------------------------
        append_log("════════════════════════════════════════════════════════════", log_path, feedback)
        append_log("Global Phase 03 | Spatiotemporal Modeling".center(60), log_path, feedback)
        append_log("════════════════════════════════════════════════════════════", log_path, feedback)
        global_out_dir = os.path.join(self.master_output_folder, "Global_Model")
        os.makedirs(global_out_dir, exist_ok=True)
        
        X_glob = np.vstack(global_X)
        y_glob = np.concatenate(global_y)
        w_glob = np.concatenate(global_w)
        coords_glob = np.vstack(global_coords)
        
        append_log(f"  → Global Matrix size: {X_glob.shape[0]} points, {X_glob.shape[1]} features.", log_path, feedback)
        
        pre_extracted_data = {
            "X": X_glob,
            "y": y_glob,
            "weights": w_glob,
            "coords": coords_glob,
            "feature_names": global_feat_names
        }
        
        run_params = masterflow_params.copy()
        run_params["OUTPUT_FOLDER"] = global_out_dir
        # We need a dummy stack path just so the parameter passes validation inside run_phase03
        run_params["INPUT_STACK"] = list(year_outputs.values())[0]["P1_FEATURES"]
        run_params["INPUT_POINTS"] = masterflow_params.get("INPUT_TRAIN")
        run_params["INPUT_MASK"] = list(year_outputs.values())[0].get("P1_MASK")
        run_params["SPATIAL_CV"] = algorithm.parameterAsBool(masterflow_params, "SPATIAL_CV_P3", context) if (algorithm and hasattr(algorithm, "parameterAsBool")) else masterflow_params.get("SPATIAL_CV_P3", masterflow_params.get("SPATIAL_CV", False))
        
        # Call Phase 3 directly!
        p3 = run_phase03_initial_modeling(algorithm, run_params, context, feedback, pre_extracted_data=pre_extracted_data)
        
        global_model_path = p3["OUTPUT_MODEL_PKL"]
        append_log("✓ Global Phase 03 completed\n", log_path, feedback)
        
        # Load global model to memory for fast predictions
        global_model = joblib.load(global_model_path)
        
        global_uncert_model_path = p3.get("OUTPUT_UNCERT_MODEL_PKL")
        global_uncert_model = None
        if global_uncert_model_path and os.path.exists(global_uncert_model_path):
            global_uncert_model = joblib.load(global_uncert_model_path)
            
        selected_indices = p3.get("SELECTED_INDICES")
        
        # ---------------------------------------------------------
        # LOOP 2: PREDICTION & PHASE 04 PER YEAR
        # ---------------------------------------------------------
        final_depth_maps = {}
        for year, outputs in year_outputs.items():
            year_start = time.time()
            append_log(f"▶ Year {year} | Phase 04 Adaptive Refinement", log_path, feedback)
            append_log("  ──────────────────────────────────────────────────────────", log_path, feedback)
            if feedback.isCanceled(): return {}
            
            year_out_dir = os.path.join(self.master_output_folder, f"Year_{year}")
            p3_dir = os.path.join(year_out_dir, "Phase_03_Initial_Modeling")
            p4_dir = os.path.join(year_out_dir, "Phase_04_Adaptive_Refinement")
            p5_dir = os.path.join(year_out_dir, "Phase_05_Scientific_Validation")
            os.makedirs(p3_dir, exist_ok=True)
            os.makedirs(p4_dir, exist_ok=True)
            os.makedirs(p5_dir, exist_ok=True)
            
            # Predict Global Model for this specific year
            p3_map = os.path.join(p3_dir, f"3_Initial_Global_Depth_{year}.tif")
            p1_feat = outputs["P1_FEATURES"]
            p1_mask = outputs["P1_MASK"]
            
            med_size = masterflow_params.get("MEDIAN_SIZE", 3)
            fmt_map = {0: "float32", 1: "float64", 2: "uint16"}
            fmt_idx = masterflow_params.get("OUTPUT_FORMAT", 0)
            output_format = fmt_map.get(fmt_idx, "float32")
            
            append_log(f"  [Prediction] Generating map using Global Model...", log_path, feedback)
            predict_map(global_model, p1_feat, p1_mask, p3_map, med_size, output_format, selected_indices=selected_indices, extra_features=[int(year)], feedback=feedback)
            
            # Apply Depth Variance Correction to this year's global prediction if enabled
            if algorithm.parameterAsBool(masterflow_params, "ENABLE_DEPTH_VARIANCE_CORR", context):
                append_log(f"  → Applying Depth Variance Correction...", log_path, feedback)
                try:
                    from sklearn.linear_model import HuberRegressor
                    import rasterio
                    
                    X_yr = outputs["X_yr"]
                    y_yr = outputs["y_yr"]
                    
                    if len(y_yr) < 10:
                        append_log(f"      → Not enough local training points ({len(y_yr)}) for Depth Variance Correction. Skipping...", log_path, feedback)
                    else:
                        # Add year feature to X_yr to match global model input
                        year_col = np.full((X_yr.shape[0], 1), int(year), dtype=X_yr.dtype)
                        X_yr_full = np.hstack((X_yr, year_col))
                        
                        if selected_indices is not None and len(selected_indices) > 0:
                            X_yr_selected = X_yr_full[:, selected_indices]
                        else:
                            X_yr_selected = X_yr_full
                            
                        # Predict using global model on this year's points
                        y_yr_pred = global_model.predict(X_yr_selected)
                        
                        # Calculate residuals
                        raw_residuals = y_yr - y_yr_pred
                        mean_bias = float(np.mean(raw_residuals))
                        append_log(f"      → Mean residual offset: {mean_bias:.4f}m", log_path, feedback)
                        
                        residuals = raw_residuals - mean_bias
                        
                        # Train year-specific Huber Regressor
                        huber_mod = HuberRegressor(epsilon=1.35)
                        huber_mod.fit(y_yr_pred.reshape(-1, 1), residuals)
                        
                        # Save the year-specific Huber model to disk
                        huber_pkl_path = os.path.join(p3_dir, f"3_Huber_Variance_Model_{year}.pkl")
                        joblib.dump(huber_mod, huber_pkl_path)
                        
                        # Apply to map
                        with rasterio.open(p3_map, "r+") as dst:
                            depth_arr = dst.read(1)
                            valid_mask = (depth_arr != -9999.0)
                            if np.any(valid_mask):
                                valid_depths = depth_arr[valid_mask]
                                residual_grid = huber_mod.predict(valid_depths.reshape(-1, 1))
                                corrected_depths = valid_depths + residual_grid + mean_bias
                                depth_arr[valid_mask] = corrected_depths
                                dst.write(depth_arr, 1)
                        append_log(f"  ✓ Depth Variance Correction applied", log_path, feedback)
                except Exception as e:
                    append_log(f"  ⚠ WARNING: Failed to apply Depth Variance Correction: {e}", log_path, feedback)
            
            
            p3_uncert_map = None
            if global_uncert_model:
                p3_uncert_map = os.path.join(p3_dir, f"3_Initial_Global_Uncertainty_{year}.tif")
                append_log(f"  → Generating uncertainty map...", log_path, feedback)
                try:
                    predict_map(global_uncert_model, p1_feat, p1_mask, p3_uncert_map, med_size, "float32", selected_indices=selected_indices, extra_features=[int(year)], feedback=feedback)
                except Exception as e:
                    append_log(f"  ⚠ WARNING: Failed to generate local uncertainty map: {e}", log_path, feedback)
                    p3_uncert_map = None
            
            # --- PHASE 03 CLEANUP (Now handled internally in trainers.py) ---
            pass

            if master_osw_polygon and os.path.exists(master_osw_polygon):
                append_log(f"  → Clipping Phase 03 Map for {year} with Master OSW Polygon...", log_path, feedback)
                p3_osw_clipped = os.path.join(p3_dir, f"Phase03_Depth_OSW_Clipped_{year}.tif")
                try:
                    processing.run(
                        "gdal:cliprasterbymasklayer",
                        {
                            "INPUT": p3_map,
                            "MASK": master_osw_polygon,
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
                        p3_map = p3_osw_clipped
                except Exception as e:
                    append_log(f"  ⚠ WARNING: Failed to clip Phase 03 for {year} with OSW Polygon: {e}", log_path, feedback)
            # -----------------------------------------------

            run_params = masterflow_params.copy()
            run_params["OUTPUT_FOLDER"] = p4_dir
            run_params["INPUT_ORIGINAL_FEAT"] = p1_feat
            run_params["INPUT_GLOBAL_RASTER"] = p3_map
            run_params["INPUT_TRAIN"] = outputs["P2_VEC"]
            run_params["SPATIAL_CV"] = algorithm.parameterAsBool(masterflow_params, "SPATIAL_CV_P4", context) if (algorithm and hasattr(algorithm, "parameterAsBool")) else masterflow_params.get("SPATIAL_CV_P4", False)
            run_params["ENABLE_DEPTH_VARIANCE_CORR"] = algorithm.parameterAsBool(masterflow_params, "ENABLE_DEPTH_VARIANCE_CORR_P4", context) if (algorithm and hasattr(algorithm, "parameterAsBool")) else masterflow_params.get("ENABLE_DEPTH_VARIANCE_CORR_P4", False)
            
            ui_stack = run_params.get("STACK_COMPONENTS_P4", [0, 1])
            run_params["STACK_COMPONENTS"] = [x + 1 for x in ui_stack]
            
            year_info = outputs["INFO"]
            ui_adaptive_depth = run_params.get("FIELD_ADAPTIVE_DEPTH", "")
            if not ui_adaptive_depth:
                ui_adaptive_depth = run_params.get("FIELD_DEPTH", "")
                
            ui_test_depth = run_params.get("FIELD_TEST_DEPTH", "")
            if not ui_test_depth:
                ui_test_depth = run_params.get("FIELD_DEPTH", "")
            
            # 1. Control points (Phase 04 Logic)
            ui_enable_adaptive = algorithm.parameterAsBool(masterflow_params, "ENABLE_ADAPTIVE", context) if (algorithm and hasattr(algorithm, "parameterAsBool")) else masterflow_params.get("ENABLE_ADAPTIVE", False)
            has_control_points = False
            if ui_enable_adaptive and year_info.get("control_path"):
                run_params["INPUT_TRAIN"] = year_info["control_path"]
                run_params["FIELD_TRAIN"] = ui_adaptive_depth
                run_params["ENABLE_ADAPTIVE"] = True
                has_control_points = True
                append_log("  [Phase 04] Adaptive Refinement", log_path, feedback)
                append_log("      → Control Points found. Executing...", log_path, feedback)
            else:
                run_params["ENABLE_ADAPTIVE"] = False
                append_log("  [Phase 04] Adaptive Refinement", log_path, feedback)
                if not ui_enable_adaptive:
                    append_log("      → Skipped by User.", log_path, feedback)
                else:
                    append_log("      → No Control Points found. SKIPPED.", log_path, feedback)
                
            # 2. Validation points (Unseen Data for Phase 05)
            ui_enable_val = algorithm.parameterAsBool(masterflow_params, "ENABLE_VALIDATION", context) if (algorithm and hasattr(algorithm, "parameterAsBool")) else masterflow_params.get("ENABLE_VALIDATION", False)
            year_unseen_path = None
            if ui_enable_val:
                if year_info.get("unseen_file_path") and os.path.exists(year_info["unseen_file_path"]):
                    year_unseen_path = year_info["unseen_file_path"]
                elif year_info.get("unseen_layer") and hasattr(year_info["unseen_layer"], "isValid") and year_info["unseen_layer"].isValid():
                    unseen_year_field = year_info.get("unseen_year_field", "")
                    unseen_layer = year_info["unseen_layer"]
                    if unseen_year_field and unseen_year_field.strip():
                        append_log(f"   [Temporal] Extracting year {year} from global Unseen layer...", log_path, feedback)
                        try:
                            ext_unseen = processing.run("native:extractbyexpression", {
                                'INPUT': unseen_layer,
                                'EXPRESSION': f'"{unseen_year_field}" = {year} OR "{unseen_year_field}" = \'{year}\'',
                                'OUTPUT': 'TEMPORARY_OUTPUT'
                            }, context=context, feedback=feedback, is_child_algorithm=True)
                            year_unseen_path = ext_unseen['OUTPUT']
                        except Exception as e:
                            append_log(f"  ⚠ WARNING: Failed to extract global Unseen for {year}: {e}", log_path, feedback)
                    else:
                        year_unseen_path = unseen_layer.source() if hasattr(unseen_layer, "source") else unseen_layer
            
            if ui_enable_val and year_unseen_path:
                run_params["ENABLE_VALIDATION"] = True
                run_params["INPUT_TEST"] = year_unseen_path
                run_params["FIELD_TEST_DEPTH"] = ui_test_depth
                append_log(f"  [Phase 05] Scientific Validation", log_path, feedback)
                append_log(f"      → Validation Points found. Executing...", log_path, feedback)
            else:
                run_params["ENABLE_VALIDATION"] = False
                append_log(f"  [Phase 05] Scientific Validation", log_path, feedback)
                if not ui_enable_val:
                    append_log(f"      → Skipped by User.", log_path, feedback)
                else:
                    append_log(f"      → No Validation Points found. SKIPPED.", log_path, feedback)
            
            # Phase 4 Execution
            if has_control_points:
                append_log(f"  [Phase 04] Running...", log_path, feedback)
                p4 = processing.run("sdb_tools:sdb_phase4_adaptive", run_params, is_child_algorithm=True, context=context, feedback=feedback)
                p4_map = p4["OUTPUT_FINAL"]
                
                # --- PHASE 04 CLEANUP (Now handled internally in evaluators.py) ---
                pass

                if master_osw_polygon and os.path.exists(master_osw_polygon):
                    append_log(f"  → Clipping Phase 04 Map for {year} with Master OSW Polygon...", log_path, feedback)
                    p4_osw_clipped = os.path.join(p4_dir, f"Phase04_Final_Depth_OSW_Clipped_{year}.tif")
                    try:
                        processing.run(
                            "gdal:cliprasterbymasklayer",
                            {
                                "INPUT": p4_map,
                                "MASK": master_osw_polygon,
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
                            p4_map = p4_osw_clipped
                    except Exception as e:
                        append_log(f"  ⚠ WARNING: Failed to clip Phase 04 for {year} with OSW Polygon: {e}", log_path, feedback)
                # -----------------------------------------------
            else:
                p4_map = p3_map  # Pass through cleaned Phase 03 map directly
            
            # Phase 5 Execution
            final_map = p4_map if os.path.exists(p4_map) else p3_map
            
            if run_params.get("ENABLE_VALIDATION") and run_params.get("INPUT_TEST"):
                append_log("  [Phase 05] Scientific Validation & Reporting...", log_path, feedback)
                p5_params = {
                    "INPUT_MAP_P3": p3_map,
                    "INPUT_MAP_P4": p4_map,
                    "INPUT_TRAIN": outputs["P2_VEC"],
                    "FIELD_TRAIN": run_params.get("FIELD_DEPTH", ""),
                    "OUTPUT_FOLDER": p5_dir,
                    "INPUT_VALIDATION": run_params["INPUT_TEST"],
                    "FIELD_VAL_DEPTH": ui_test_depth
                }
                    
                try:
                    p5 = processing.run("sdb_tools:sdb_05_reporting", p5_params, is_child_algorithm=True, context=context, feedback=feedback)
                    append_log("  ✓ Scientific Validation metrics generated", log_path, feedback)
                except Exception as e:
                    append_log(f"  ✗ ERROR: Phase 5 Failed for {year}: {e}", log_path, feedback)
            else:
                if not ui_enable_val:
                    append_log("  [Phase 05] Scientific Validation Skipped by User.", log_path, feedback)
                else:
                    append_log("  [Phase 05] No Validation Points found. Skipping Scientific Validation metrics.", log_path, feedback)

            # Generate Dashboard and 3D plot directly into the Phase 05 folder (Always)
            from Bathymetrix_AI.core.pipeline import generate_html_dashboard, generate_3d_seabed_png
            
            # Generate 3D plot directly into the Phase 05 folder
            if final_map and os.path.exists(final_map):
                out_3d_png = os.path.join(p5_dir, "5_Plot_3D_Seabed.png")
                generate_3d_seabed_png(final_map, out_3d_png, feedback)
            
            try:
                generate_html_dashboard(
                    out_dir=p5_dir,
                    p3_dir=global_out_dir,
                    p4_dir=p4_dir,
                    spatial_cv_p3=masterflow_params.get("SPATIAL_CV_P3", True),
                    spatial_cv_p4=masterflow_params.get("SPATIAL_CV_P4", True),
                    enable_ransac=masterflow_params.get("ENABLE_RANSAC", False),
                    filter_mode=masterflow_params.get("FILTER_MODE", 0),
                    field_depth=run_params.get("FIELD_DEPTH", ""),
                    field_weight=masterflow_params.get("FIELD_WEIGHT", ""),
                    collision_handling_idx=masterflow_params.get("COLLISION_HANDLING", 0),
                    log_path=log_path,
                    feedback=feedback,
                    raster_name=os.path.basename(outputs["INFO"]["image_path"]),
                    train_name=os.path.basename(outputs["P2_VEC"]),
                    test_name=os.path.basename(year_unseen_path) if year_unseen_path else None,
                    final_raster_path=final_map,
                    p2_dir=p2_dir
                )
            except Exception as e:
                append_log(f"  ⚠ WARNING: Dashboard generation failed for {year}: {e}", log_path, feedback)
            
            # Save final SDB depth map to Phase 5 and load it in QGIS
            import shutil
            from qgis.core import QgsProcessingContext, QgsProject
            
            if final_map and os.path.exists(final_map):
                final_sdb_path = os.path.join(year_out_dir, f"SDB {year}.tif")
                shutil.copy2(final_map, final_sdb_path)
                
                # Generate standardized QML style
                from Bathymetrix_AI.infrastructure.raster_io import write_qml_style, StylePostProcessor
                dst_qml = write_qml_style(final_sdb_path)
                
                append_log(f"  ✓ Saved final map: {os.path.basename(final_sdb_path)}", log_path, feedback)
                
                if context and hasattr(context, "addLayerToLoadOnCompletion") and hasattr(QgsProcessingContext, "LayerDetails"):
                    try:
                        details = QgsProcessingContext.LayerDetails(f"SDB {year}", QgsProject.instance(), "SDB")
                        if dst_qml and os.path.exists(dst_qml):
                            details.setPostProcessor(StylePostProcessor(dst_qml))
                        context.addLayerToLoadOnCompletion(final_sdb_path, details)
                    except Exception:
                        pass
            
            scene_elapsed = time.time() - year_start
            m, s = divmod(int(scene_elapsed), 60)
            h, m = divmod(m, 60)
            append_log(f"✓ Year {year} completed | {h:02d}:{m:02d}:{s:02d}\n", log_path, feedback)
            
        total_elapsed = time.time() - start_time
        tm, ts = divmod(int(total_elapsed), 60)
        th, tm = divmod(tm, 60)
        
        try:
            from Bathymetrix_AI.infrastructure.logging import log_module_completion
            primary_files = {
                "Multi-Year Analytics": os.path.join(self.master_output_folder, "MultiYear_Analytics"),
                "Final Execution Log": log_path
            }
            log_module_completion(
                module_title=f"SDB Spatiotemporal Masterflow ({len(yearly_datasets)} Years - Elapsed: {th:02d}:{tm:02d}:{ts:02d})",
                out_dir=self.master_output_folder,
                primary_files=primary_files,
                log_path=log_path,
                feedback=feedback
            )
        except Exception:
            append_log("════════════════════════════════════════════════════════════", log_path, feedback)
            append_log(f"✓ SDB Spatiotemporal Masterflow Completed".center(60), log_path, feedback)
            append_log(f"Total Time: {th:02d}:{tm:02d}:{ts:02d}", log_path, feedback)
            append_log("════════════════════════════════════════════════════════════", log_path, feedback)
        
        return {"OUTPUT_FOLDER": self.master_output_folder, "LOG_FILE": log_path}
