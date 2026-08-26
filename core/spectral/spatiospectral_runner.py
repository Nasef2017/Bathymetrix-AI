import os
import glob

try:
    from qgis.core import QgsProcessingException
    from qgis import processing
except ImportError:
    QgsProcessingException = Exception
    processing = None

try:
    from Bathymetrix_AI.infrastructure.logging import append_log
    from Bathymetrix_AI.core.spectral.aggregation import spatiospectral_aggregate, spatiospectral_mask_intersection
    from Bathymetrix_AI.infrastructure.raster_io import clean_depth_map, remove_positive_pixels, slope_filter_depth
except (ImportError, ValueError):
    from infrastructure.logging import append_log
    from core.spectral.aggregation import spatiospectral_aggregate, spatiospectral_mask_intersection
    from infrastructure.raster_io import clean_depth_map, remove_positive_pixels, slope_filter_depth


class SpatioSpectralSDBRunner:
    def __init__(self, master_output_folder):
        self.master_output_folder = master_output_folder

    def run_spatiospectral_flow(self, image_folder, masterflow_params, algorithm, context, feedback):
        import time
        start_time = time.time()
        start_str = time.strftime('%H:%M:%S', time.localtime(start_time))
        
        log_path = os.path.join(self.master_output_folder, "SpatioSpectral_Flow_Log.txt")
        append_log("════════════════════════════════════════════════════════════", log_path, feedback)
        append_log("SDB SpatioSpectral Masterflow".center(60), log_path, feedback)
        append_log("════════════════════════════════════════════════════════════", log_path, feedback)
        append_log(f"Started: {start_str}", log_path, feedback)
        
        # 1. Scan image folder for tif files
        tif_files = glob.glob(os.path.join(image_folder, "*.tif"))
        if not tif_files:
            raise QgsProcessingException(f"No .tif images found in {image_folder}")
            
        append_log(f"Scenes: {len(tif_files)}", log_path, feedback)
        append_log("════════════════════════════════════════════════════════════\n", log_path, feedback)
        
        training_layer = masterflow_params.get("INPUT_TRAIN")
        if not training_layer:
            raise QgsProcessingException("No training layer provided.")

        # --- Pre-Scan: Spatial Overlap Check ---
        append_log("→ Validating Spatial Overlap...", log_path, feedback)
        from qgis.core import QgsRasterLayer, QgsVectorLayer, QgsCoordinateTransform, QgsProject, QgsGeometry
        
        vl = training_layer if isinstance(training_layer, QgsVectorLayer) else QgsVectorLayer(training_layer, "training_points", "ogr")
        if vl and vl.isValid():
            v_crs = vl.crs()
            v_geom = QgsGeometry.fromRect(vl.extent())
            for tif_path in tif_files:
                rl = QgsRasterLayer(tif_path, "raster")
                if rl.isValid():
                    r_crs = rl.crs()
                    r_geom = QgsGeometry.fromRect(rl.extent())
                    if r_crs != v_crs:
                        try:
                            transform = QgsCoordinateTransform(r_crs, v_crs, QgsProject.instance())
                            r_geom.transform(transform)
                        except Exception:
                            pass
                    if not r_geom.intersects(v_geom):
                        scene_name = os.path.basename(tif_path)
                        err_msg = f"✗ ERROR: Scene '{scene_name}' failed spatial overlap."
                        append_log(err_msg, log_path, feedback)
                        raise QgsProcessingException(err_msg)
        append_log("✓ Spatial Overlap verified\n", log_path, feedback)
        # --- End Pre-Scan ---

        aggregated_dir = os.path.join(self.master_output_folder, "Aggregated_Results")
        os.makedirs(aggregated_dir, exist_ok=True)
        
        p3_depth_maps = []
        p1_masks = []
        p3_weights = []
        
        # ---------------------------------------------------------
        # LOOP 1: Phase 1, 2, 3 for Each Scene
        # ---------------------------------------------------------
        for i, tif_path in enumerate(tif_files):
            scene_start = time.time()
            scene_name = os.path.splitext(os.path.basename(tif_path))[0]
            append_log(f"▶ Scene {i+1:02d} / {len(tif_files):02d}", log_path, feedback)
            append_log(f"  Image: {scene_name}", log_path, feedback)
            append_log("  ──────────────────────────────────────────────────────────", log_path, feedback)
            if feedback.isCanceled(): return {}
            
            scene_out_dir = os.path.join(self.master_output_folder, f"Scene_{i+1}_{scene_name}")
            p1_dir = os.path.join(scene_out_dir, "Phase_01_Preprocessing")
            p2_dir = os.path.join(scene_out_dir, "Phase_02_Filtering")
            p3_dir = os.path.join(scene_out_dir, "Phase_03_Initial_Modeling")
            
            os.makedirs(p1_dir, exist_ok=True)
            os.makedirs(p2_dir, exist_ok=True)
            os.makedirs(p3_dir, exist_ok=True)
            
            run_params = masterflow_params.copy()
            
            # Map WATER_MASK_POLY to INPUT_WATER_POLY for Phase 1
            if "WATER_MASK_POLY" in run_params and "INPUT_WATER_POLY" not in run_params:
                run_params["INPUT_WATER_POLY"] = run_params["WATER_MASK_POLY"]
                
            run_params["INPUT_RASTER"] = tif_path
            run_params["INPUT_TRAIN"] = training_layer
            run_params["OUTPUT_FOLDER"] = p1_dir
            
            # Phase 1
            enable_preproc = algorithm.parameterAsBool(masterflow_params, "ENABLE_PREPROCESSING", context) if (algorithm and hasattr(algorithm, "parameterAsBool")) else masterflow_params.get("ENABLE_PREPROCESSING", True)
            if enable_preproc:
                append_log("  [Phase 01] Pre-processing", log_path, feedback)
                p1 = processing.run("sdb_tools:sdb_phase1_preprocessing", run_params, is_child_algorithm=True, context=context, feedback=feedback)
                p1_feat = p1["OUTPUT_FEATURES"]
                p1_mask = p1["OUTPUT_MASK"]
                p1_masks.append(p1_mask)
                append_log("  ✓ Phase 01 completed\n", log_path, feedback)
            else:
                append_log("  [Phase 01] Pre-processing", log_path, feedback)
                append_log("      → Skipped by User.\n", log_path, feedback)
                p1_feat = tif_path
                p1_mask = None

            # Phase 2
            enable_p2 = algorithm.parameterAsBool(masterflow_params, "ENABLE_RANSAC", context) if (algorithm and hasattr(algorithm, "parameterAsBool")) else masterflow_params.get("ENABLE_RANSAC", True)
            if enable_p2:
                append_log("  [Phase 02] Filtering", log_path, feedback)
                p2_params = run_params.copy()
                p2_params["INPUT_STACK"] = p1_feat
                p2_params["INPUT_POINTS"] = training_layer
                p2_params["BLUE_BAND"] = run_params.get("FILTER_NUMERATOR_BAND", run_params.get("BLUE_BAND"))
                p2_params["GREEN_BAND"] = run_params.get("FILTER_DENOMINATOR_BAND", run_params.get("GREEN_BAND"))
                p2_params["RESIDUAL_THRESHOLD"] = run_params.get("RANSAC_THRESHOLD", 3.0)
                p2_params["OUTPUT_FOLDER"] = p2_dir
                p2 = processing.run("sdb_tools:sdb_02_filtering", p2_params, is_child_algorithm=True, context=context, feedback=feedback)
                p2_vec = p2["OUTPUT_CLEAN_VEC"]
                append_log("  ✓ Phase 02 completed\n", log_path, feedback)
            else:
                append_log("  [Phase 02] Filtering", log_path, feedback)
                append_log("      → Skipped by User.\n", log_path, feedback)
                p2_vec = training_layer
            
            # Phase 3
            append_log("  [Phase 03] Global Modeling", log_path, feedback)
            p3_params = run_params.copy()
            p3_params["INPUT_STACK"] = p1_feat
            p3_params["INPUT_MASK"] = p1_mask
            p3_params["INPUT_POINTS"] = p2_vec
            p3_params["OUTPUT_FOLDER"] = p3_dir
            if "SPATIAL_CV_P3" in masterflow_params:
                p3_params["SPATIAL_CV"] = masterflow_params["SPATIAL_CV_P3"]
            
            p3 = processing.run("sdb_tools:sdb_03_initial_modeling", p3_params, is_child_algorithm=True, context=context, feedback=feedback)
            
            best_depth_path = p3.get("OUTPUT_DEPTH_MAP")
            
            # Extract R2 and RMSE to compute Weight
            r2 = p3.get("BEST_R2", 0.0)
            rmse = p3.get("BEST_RMSE", 1.0)
            weight = max(0.001, r2) / (rmse + 0.001)
            
            append_log(f"  → R2: {r2:.4f} | RMSE: {rmse:.4f} | Weight: {weight:.4f}", log_path, feedback)
            append_log("  ✓ Phase 03 completed\n", log_path, feedback)
            
            # ---------------------------------------------------------
            # CLEANUP: Clamp max depth and remove extreme nodata predictions
            # ---------------------------------------------------------
            if best_depth_path and os.path.exists(best_depth_path):
                max_depth = masterflow_params.get("MAX_DEPTH_THRESHOLD", -30.0)
                p3_clamped = os.path.join(p3_dir, "3_Initial_Global_Depth_Cleaned.tif")
                
                # We use clean_depth_map which also masks using the feature stack extent
                clean_depth_map(best_depth_path, p1_feat, max_depth, p3_clamped, context, feedback)
                current_p3 = p3_clamped
                
                apply_slope = algorithm.parameterAsBool(masterflow_params, "ENABLE_SLOPE_FILTER", context) if (algorithm and hasattr(algorithm, "parameterAsBool")) else masterflow_params.get("ENABLE_SLOPE_FILTER", True)
                slope_threshold_val = algorithm.parameterAsDouble(masterflow_params, "SLOPE_THRESHOLD", context) if (algorithm and hasattr(algorithm, "parameterAsDouble")) else masterflow_params.get("SLOPE_THRESHOLD", 35.0)
                
                if apply_slope:
                    p3_slope = os.path.join(p3_dir, "3_Initial_Global_Depth_SlopeFiltered.tif")
                    current_p3 = slope_filter_depth(
                        current_p3,
                        slope_threshold=slope_threshold_val,
                        out_path=p3_slope,
                        context=context,
                        feedback=feedback,
                    )
                
                remove_pos = algorithm.parameterAsBool(masterflow_params, "REMOVE_POSITIVES", context) if (algorithm and hasattr(algorithm, "parameterAsBool")) else masterflow_params.get("REMOVE_POSITIVES", True)
                if remove_pos:
                    p3_no_pos = os.path.join(p3_dir, "3_Initial_Global_Depth_NoPositives.tif")
                    remove_positive_pixels(current_p3, p3_no_pos, feedback)
                    final_p3_path = p3_no_pos
                else:
                    final_p3_path = current_p3
                    
                p3_depth_maps.append(final_p3_path)
                p3_weights.append(weight)
            else:
                append_log(f"✗ ERROR: Scene {i+1:02d} failed during model prediction.", log_path, feedback)
                
            scene_elapsed = time.time() - scene_start
            m, s = divmod(int(scene_elapsed), 60)
            h, m = divmod(m, 60)
            append_log(f"✓ Scene {i+1:02d} completed | {h:02d}:{m:02d}:{s:02d}\n", log_path, feedback)

        if not p3_depth_maps:
            raise QgsProcessingException("No valid Phase 3 Depth Maps generated. SpatioSpectral Flow failed.")

        # ---------------------------------------------------------
        # AGGREGATION: Pixel-wise Median/Mean/Max/Min
        # ---------------------------------------------------------
        agg_method_idx = masterflow_params.get("SPATIOSPECTRAL_AGGREGATION", 0)
        agg_methods = ["Median", "Mean", "Max (Deepest)", "Min (Shallowest)", "Weighted Median (R2/RMSE)", "Weighted Mean (R2/RMSE)", "Select Best Scene (Highest R2 / Lowest RMSE)"]
        agg_method = agg_methods[agg_method_idx] if 0 <= agg_method_idx < len(agg_methods) else "Median"
        
        append_log("════════════════════════════════════════════════════════════", log_path, feedback)
        append_log("SPATIOSPECTRAL AGGREGATION".center(60), log_path, feedback)
        append_log("════════════════════════════════════════════════════════════", log_path, feedback)
        
        safe_agg_method_name = agg_method.replace("/", "_").replace("\\", "_").replace(" ", "_")
        aggregated_depth_path = os.path.join(aggregated_dir, f"Aggregated_Depth_{safe_agg_method_name}.tif")
        aggregated_mask_path = os.path.join(aggregated_dir, "Aggregated_Intersection_Mask.tif")
        
        if agg_method == "Select Best Scene (Highest R2 / Lowest RMSE)":
            best_idx = p3_weights.index(max(p3_weights))
            best_depth_map = p3_depth_maps[best_idx]
            
            append_log(f"→ Selected Scene {best_idx+1} as the Best Scene (Weight: {p3_weights[best_idx]:.4f}).", log_path, feedback)
            
            import shutil
            shutil.copy2(best_depth_map, aggregated_depth_path)
            
            append_log("→ Copying mask for the best scene...", log_path, feedback)
            if p1_masks and best_idx < len(p1_masks):
                shutil.copy2(p1_masks[best_idx], aggregated_mask_path)
        else:
            append_log("→ Aggregating scene results...", log_path, feedback)
            
            agg_kwargs = {"method": agg_method, "feedback": feedback}
            if "Weighted" in agg_method:
                agg_kwargs["weights"] = p3_weights
                
            spatiospectral_aggregate(p3_depth_maps, aggregated_depth_path, **agg_kwargs)
            
            append_log("→ Generating aggregated intersection mask...", log_path, feedback)
            spatiospectral_mask_intersection(p1_masks, aggregated_mask_path, feedback=feedback)
            
        append_log("✓ Aggregation completed\n════════════════════════════════════════════════════════════\n", log_path, feedback)
        
        # ---------------------------------------------------------
        # LOOP 2: Phase 4 & 5 (On Aggregated Result)
        # ---------------------------------------------------------
        
        p4_dir = os.path.join(aggregated_dir, "Phase_04_Adaptive_Refinement")
        p5_dir = os.path.join(aggregated_dir, "Phase_05_Scientific_Validation")
        os.makedirs(p4_dir, exist_ok=True)
        os.makedirs(p5_dir, exist_ok=True)
        
        # Phase 4
        enable_adaptive = algorithm.parameterAsBool(masterflow_params, "ENABLE_ADAPTIVE", context) if (algorithm and hasattr(algorithm, "parameterAsBool")) else masterflow_params.get("ENABLE_ADAPTIVE", False)
        
        if enable_adaptive:
            append_log("[Phase 04] Adaptive Refinement", log_path, feedback)
            append_log("  → Post-Aggregation processing", log_path, feedback)
            append_log("  → Running once on final depth map", log_path, feedback)
            
            p4_params = masterflow_params.copy()
            p4_params["INPUT_GLOBAL_RASTER"] = aggregated_depth_path
            
            # We must provide *something* to INPUT_ORIGINAL_FEAT because it's required by the UI of SDB_04_Spatial_Retraining,
            # but we explicitly tell P4 to NOT use it via STACK_COMPONENTS (which we set to Depth + Residual Error Grid only).
            p4_params["INPUT_ORIGINAL_FEAT"] = aggregated_depth_path
            
            # 1 = Phase 03 Depth Map, 2 = Residual Error Grid. (0 = Feature Stack, which we exclude)
            ui_stack = masterflow_params.get("STACK_COMPONENTS_P4", [0, 1])
            p4_params["STACK_COMPONENTS"] = [x + 1 for x in ui_stack]
            
            p4_params["INPUT_MASK"] = aggregated_mask_path
            
            # Use Adaptive Training points if provided, else fallback to main training points
            adaptive_train = masterflow_params.get("INPUT_ADAPTIVE_TRAIN")
            if adaptive_train:
                p4_params["INPUT_TRAIN"] = adaptive_train
                p4_params["FIELD_TRAIN"] = masterflow_params.get("FIELD_ADAPTIVE_DEPTH", masterflow_params.get("FIELD_DEPTH"))
            else:
                p4_params["INPUT_TRAIN"] = training_layer
                p4_params["FIELD_TRAIN"] = masterflow_params.get("FIELD_DEPTH")
                
            p4_params["OUTPUT_FOLDER"] = p4_dir
            if "SPATIAL_CV_P4" in masterflow_params:
                p4_params["SPATIAL_CV"] = masterflow_params["SPATIAL_CV_P4"]
            p4_params["ENABLE_DEPTH_VARIANCE_CORR"] = masterflow_params.get("ENABLE_DEPTH_VARIANCE_CORR_P4", False)
            
            p4 = processing.run("sdb_tools:sdb_phase4_adaptive", p4_params, is_child_algorithm=True, context=context, feedback=feedback)
            
            append_log("✓ Phase 04 completed\n", log_path, feedback)
            
            raw_p4_depth = p4["OUTPUT_FINAL"]
            
            # ---------------------------------------------------------
            # CLEANUP Phase 4 Output
            # ---------------------------------------------------------
            if raw_p4_depth and os.path.exists(raw_p4_depth):
                max_depth = masterflow_params.get("MAX_DEPTH_THRESHOLD", -30.0)
                p4_clamped = os.path.join(p4_dir, "4_Phase04_Depth_Cleaned.tif")
                
                # Clean depth map (masking to the aggregated intersection mask)
                clean_depth_map(raw_p4_depth, aggregated_depth_path, max_depth, p4_clamped, context, feedback)
                current_p4 = p4_clamped
                
                apply_slope = algorithm.parameterAsBool(masterflow_params, "ENABLE_SLOPE_FILTER", context) if (algorithm and hasattr(algorithm, "parameterAsBool")) else masterflow_params.get("ENABLE_SLOPE_FILTER", True)
                slope_threshold_val = algorithm.parameterAsDouble(masterflow_params, "SLOPE_THRESHOLD", context) if (algorithm and hasattr(algorithm, "parameterAsDouble")) else masterflow_params.get("SLOPE_THRESHOLD", 35.0)
                
                if apply_slope:
                    p4_slope = os.path.join(p4_dir, "4_Phase04_Depth_SlopeFiltered.tif")
                    current_p4 = slope_filter_depth(
                        current_p4,
                        slope_threshold=slope_threshold_val,
                        out_path=p4_slope,
                        context=context,
                        feedback=feedback,
                    )
                
                remove_pos = algorithm.parameterAsBool(masterflow_params, "REMOVE_POSITIVES", context) if (algorithm and hasattr(algorithm, "parameterAsBool")) else masterflow_params.get("REMOVE_POSITIVES", True)
                if remove_pos:
                    p4_no_pos = os.path.join(p4_dir, "4_Phase04_Depth_NoPositives.tif")
                    remove_positive_pixels(current_p4, p4_no_pos, feedback)
                    p4_final_depth = p4_no_pos
                else:
                    p4_final_depth = current_p4
            else:
                p4_final_depth = raw_p4_depth
        else:
            append_log("[Phase 04] Adaptive Refinement", log_path, feedback)
            append_log("  → Skipped by User.\n", log_path, feedback)
            p4_final_depth = aggregated_depth_path
            p4_dir = None
        
        # Phase 5
        append_log("[Phase 05] Finalization", log_path, feedback)
        enable_val = algorithm.parameterAsBool(masterflow_params, "ENABLE_VALIDATION", context) if (algorithm and hasattr(algorithm, "parameterAsBool")) else masterflow_params.get("ENABLE_VALIDATION", False)
        
        # Only run if ENABLE_VALIDATION is True and INPUT_TEST is provided
        input_test_layer = masterflow_params.get("INPUT_TEST")
        if enable_val and input_test_layer:
            append_log("  → Generating final outputs...", log_path, feedback)
            p5_params = {
                "INPUT_MAP_P3": aggregated_depth_path,
                "INPUT_MAP_P4": p4_final_depth if p4_final_depth else aggregated_depth_path,
                "INPUT_TRAIN": training_layer,
                "FIELD_TRAIN": masterflow_params.get("FIELD_DEPTH"),
                "INPUT_VALIDATION": input_test_layer,
                "FIELD_VAL_DEPTH": masterflow_params.get("FIELD_TEST_DEPTH", masterflow_params.get("FIELD_DEPTH")),
                "OUTPUT_FOLDER": p5_dir
            }
            append_log("  → Saving validation results...", log_path, feedback)
            p5 = processing.run("sdb_tools:sdb_05_reporting", p5_params, is_child_algorithm=True, context=context, feedback=feedback)
            append_log("✓ Scientific Validation metrics generated", log_path, feedback)
        else:
            if not enable_val:
                append_log("  → Validation Skipped by User.", log_path, feedback)
            else:
                append_log("  ⚠ WARNING: No independent validation points provided. Skipping Scientific Validation metrics.", log_path, feedback)

        # Generate static 3D seabed PNG (Always)
        final_depth_for_3d = p4_final_depth if p4_final_depth else aggregated_depth_path
        if final_depth_for_3d and os.path.exists(final_depth_for_3d):
            try:
                from Bathymetrix_AI.core.pipeline import generate_3d_seabed_png
                out_3d_png = os.path.join(p5_dir, "5_Plot_3D_Seabed.png")
                generate_3d_seabed_png(final_depth_for_3d, out_3d_png, feedback)
            except Exception as e:
                append_log(f"  ⚠ WARNING: 3D Seabed Plot failed: {str(e)}", log_path, feedback)

        # Generate Interactive HTML Dashboard (Always)
        append_log("  → Generating Interactive HTML Dashboard...", log_path, feedback)
        try:
            from Bathymetrix_AI.core.pipeline import generate_html_dashboard
            generate_html_dashboard(
                out_dir=self.master_output_folder,
                p3_dir=self.master_output_folder,
                p4_dir=p4_dir,
                spatial_cv_p3=False,
                spatial_cv_p4=masterflow_params.get("SPATIAL_CV_P4", False),
                field_depth=masterflow_params.get("FIELD_DEPTH"),
                feedback=feedback,
                raster_name="SpatioSpectral Aggregated Scenes",
                train_name="Training Points",
                test_name="Validation Points" if input_test_layer else None,
                final_raster_path=final_depth_for_3d,
                is_spatiospectral=True,
                p2_dir=os.path.join(self.master_output_folder, "Phase_02_Filtering"),
                p5_dir=p5_dir
            )
        except Exception as e:
            append_log(f"  ⚠ WARNING: Dashboard generation failed: {str(e)}", log_path, feedback)
        
        append_log("✓ Phase 05 Finalization completed\n", log_path, feedback)

        # Generate standardized ocean bathymetry .qml styles & load layers to canvas
        try:
            from Bathymetrix_AI.infrastructure.raster_io import write_qml_style
            from Bathymetrix_AI.infrastructure.canvas import add_raster_to_canvas
            if aggregated_depth_path and os.path.exists(aggregated_depth_path):
                qml_agg = write_qml_style(aggregated_depth_path)
                add_raster_to_canvas(
                    aggregated_depth_path,
                    f"Aggregated Depth ({self.agg_method})",
                    context=context,
                    style_path=qml_agg,
                )
            if p4_final_depth and os.path.exists(p4_final_depth):
                qml_p4 = write_qml_style(p4_final_depth)
                add_raster_to_canvas(
                    p4_final_depth,
                    f"Phase 04 Final Refined Depth ({self.agg_method})",
                    context=context,
                    style_path=qml_p4,
                )
        except Exception:
            pass


        total_elapsed = time.time() - start_time
        tm, ts = divmod(int(total_elapsed), 60)
        th, tm = divmod(tm, 60)

        try:
            from Bathymetrix_AI.infrastructure.logging import log_module_completion
            dash_path = os.path.join(self.master_output_folder, "SDB_Validation_Dashboard.html")
            if not os.path.exists(dash_path):
                dash_path = os.path.join(p5_dir, "SDB_Validation_Dashboard.html")
            primary_files = {
                "Aggregated Depth Map": aggregated_depth_path,
                "Refined Depth Map": p4_final_depth,
                "HTML Dashboard": dash_path if os.path.exists(dash_path) else None,
                "3D Seabed Plot": os.path.join(p5_dir, "5_Plot_3D_Seabed.png") if os.path.exists(os.path.join(p5_dir, "5_Plot_3D_Seabed.png")) else None,
                "IHO S-44 Assessment CSV": os.path.join(p5_dir, "5_Stratified_Error_Analysis.csv") if os.path.exists(os.path.join(p5_dir, "5_Stratified_Error_Analysis.csv")) else None
            }
            log_module_completion(
                module_title=f"SDB SpatioSpectral Masterflow ({len(tif_files)} Scenes - Elapsed: {th:02d}:{tm:02d}:{ts:02d})",
                out_dir=self.master_output_folder,
                primary_files=primary_files,
                log_path=log_path,
                feedback=feedback
            )
        except Exception:
            append_log("════════════════════════════════════════════════════════════", log_path, feedback)
            append_log("✓ SDB SpatioSpectral Masterflow Completed".center(60), log_path, feedback)
            append_log("════════════════════════════════════════════════════════════", log_path, feedback)
            append_log(f"Scenes processed : {len(tif_files)} / {len(tif_files)}", log_path, feedback)
            append_log(f"Total elapsed    : {th:02d}:{tm:02d}:{ts:02d}", log_path, feedback)
            append_log("Status           : SUCCESS", log_path, feedback)
            append_log("════════════════════════════════════════════════════════════\n", log_path, feedback)
        
        return {
            "AGGREGATED_DEPTH": aggregated_depth_path,
            "FINAL_REFINED_DEPTH": p4_final_depth,
            "OUTPUT_FOLDER": self.master_output_folder
        }
