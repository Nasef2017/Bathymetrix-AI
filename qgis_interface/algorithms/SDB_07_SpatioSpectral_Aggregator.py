import os
import glob
import re
import warnings

from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterFile,
    QgsProcessingParameterEnum,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterDefinition,
    QgsProcessingException,
    QgsProcessingContext,
    QgsProject,
)
from qgis import processing

try:
    from Bathymetrix_AI.core.spectral.aggregation import (
        spatiospectral_aggregate,
        spatiospectral_mask_intersection,
    )
    from Bathymetrix_AI.infrastructure.raster_io import (
        clean_depth_map,
        remove_positive_pixels,
        slope_filter_depth,
        get_raster_min_max,
        write_qml_style,
        StylePostProcessor,
    )
    from Bathymetrix_AI.infrastructure.canvas import add_raster_to_canvas
    from Bathymetrix_AI.infrastructure.logging import log_module_completion, append_log
    from Bathymetrix_AI.core.pipeline import generate_html_dashboard, generate_3d_seabed_png
except (ImportError, ValueError):
    from core.spectral.aggregation import (
        spatiospectral_aggregate,
        spatiospectral_mask_intersection,
    )
    from infrastructure.raster_io import (
        clean_depth_map,
        remove_positive_pixels,
        slope_filter_depth,
        get_raster_min_max,
        write_qml_style,
        StylePostProcessor,
    )
    from infrastructure.canvas import add_raster_to_canvas
    from infrastructure.logging import log_module_completion, append_log
    from core.pipeline import generate_html_dashboard, generate_3d_seabed_png

warnings.filterwarnings("ignore")


class PostSpatioSpectralAggregator(QgsProcessingAlgorithm):
    """
    Post-SpatioSpectral Aggregator:
    Re-aggregates depth maps from a completed SDB SpatioSpectral Masterflow workspace,
    and runs downstream Phase 04 Adaptive Refinement, Phase 05 Validation, and Cleanup
    using the exact same shared parameters and options as SDB SpatioSpectral Masterflow.
    """

    # 1. Parameter Constants (Matching SDB SpatioSpectral Flow exactly)
    INPUT_WORKSPACE = "INPUT_WORKSPACE"
    SPATIOSPECTRAL_AGGREGATION = "SPATIOSPECTRAL_AGGREGATION"
    OUTPUT_FOLDER = "OUTPUT_FOLDER"

    # [4] Phase 04: Adaptive Refinement
    ENABLE_ADAPTIVE = "ENABLE_ADAPTIVE"
    STACK_COMPONENTS_P4 = "STACK_COMPONENTS_P4"
    FEATURE_CORR_METHOD_P4 = "FEATURE_CORR_METHOD_P4"
    FEATURE_CORR_THRESHOLD_P4 = "FEATURE_CORR_THRESHOLD_P4"
    RESIDUAL_INTERP_METHOD = "RESIDUAL_INTERP_METHOD"
    KNN_NEIGHBORS = "KNN_NEIGHBORS"
    MAX_GPR_SAMPLES = "MAX_GPR_SAMPLES"
    SPATIAL_CV_P4 = "SPATIAL_CV_P4"
    ENABLE_DEPTH_VARIANCE_CORR_P4 = "ENABLE_DEPTH_VARIANCE_CORR_P4"
    INPUT_ADAPTIVE_TRAIN = "INPUT_ADAPTIVE_TRAIN"
    FIELD_ADAPTIVE_DEPTH = "FIELD_ADAPTIVE_DEPTH"

    # [5] Phase 05: Validation & Reporting
    ENABLE_VALIDATION = "ENABLE_VALIDATION"
    INPUT_TEST = "INPUT_TEST"
    FIELD_TEST_DEPTH = "FIELD_TEST_DEPTH"

    # SDB Composite Score & Model Selection Strategy
    SCORE_SELECTION_STRATEGY = "SCORE_SELECTION_STRATEGY"
    SCORE_METRICS = "SCORE_METRICS"
    SCORE_CUSTOM_CONFIG = "SCORE_CUSTOM_CONFIG"

    SCORE_STRATEGY_OPTIONS = [
        "0 - Winner Stability (Monte Carlo Sensitivity Analysis) [Recommended]",
        "1 - Max SDB Composite Score (0-100 Benchmark Matrix)",
        "2 - Highest R2 (Variance Explained)",
        "3 - Lowest RMSE (Overall Vertical Error)",
        "4 - Lowest wMAPE (Shallow-Depth Relative Error)",
        "5 - Lowest |Bias| (Zero-Mean Residual Offset)",
        "6 - Lowest MAE (Mean Absolute Error)",
    ]

    SCORE_METRIC_OPTIONS = [
        "R2 Accuracy (Correlation & Fit)",
        "RMSE (Root Mean Squared Error in meters)",
        "wMAPE (Weighted Mean Absolute Percentage Error %)",
        "|Bias| (Zero-Mean Residual Offset in meters)",
        "MAE (Mean Absolute Error in meters)",
    ]

    # Cleanup & Thresholds
    MAX_DEPTH_THRESHOLD = "MAX_DEPTH_THRESHOLD"
    REMOVE_POSITIVES = "REMOVE_POSITIVES"
    ENABLE_SLOPE_FILTER = "ENABLE_SLOPE_FILTER"
    SLOPE_THRESHOLD = "SLOPE_THRESHOLD"

    # Dropdown Options (Matching SpatioSpectral Flow)
    AGGREGATION_METHODS = [
        "Median",
        "Mean",
        "Max (Deepest)",
        "Min (Shallowest)",
        "Weighted Median (R2/RMSE)",
        "Weighted Mean (R2/RMSE)",
        "Select Best Scene (High R2 / Low RMSE)",
    ]

    FEATURE_CORR_THRESHOLDS_P4 = [
        "Use Phase 03 (-1.0)",
        "0.0",
        "0.1",
        "0.2",
        "0.3",
        "0.4",
        "0.5",
        "0.6",
        "0.7",
        "0.8",
        "0.9",
        "1.0",
    ]


    def initAlgorithm(self, config=None):
        # -------------------------------------------------------------------
        # [0] General Settings (Input Workspace & Aggregation Method)
        # -------------------------------------------------------------------
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_WORKSPACE,
                "📁 [0.1] SpatioSpectral Master Output Folder (Contains Scene_* folders & Logs)",
                behavior=QgsProcessingParameterFile.Folder,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.SPATIOSPECTRAL_AGGREGATION,
                "🧩 [0.2] SpatioSpectral Aggregation Method",
                options=self.AGGREGATION_METHODS,
                defaultValue=4,  # Weighted Median default
            )
        )
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_FOLDER, "📁 [0.3] Output Folder"
            )
        )

        # -------------------------------------------------------------------
        # [4] Phase 04: Adaptive Refinement (Matching SpatioSpectral Flow)
        # -------------------------------------------------------------------
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_ADAPTIVE,
                "🎯 [4] Enable Phase 04 Adaptive Refinement",
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.STACK_COMPONENTS_P4,
                "🎯 [4] Features for Retraining",
                options=["Phase 03 Depth Map", "Residual Error Grid"],
                allowMultiple=True,
                defaultValue=[0, 1],
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.FEATURE_CORR_METHOD_P4,
                "🤖 [4] Feature Correlation Method",
                options=[
                    "Disabled",
                    "Pearson (Linear)",
                    "Spearman (Rank)",
                    "Automatic-RANSAC",
                    "Automatic-Random Forest",
                ],
                defaultValue=3,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.FEATURE_CORR_THRESHOLD_P4,
                "🤖 [4] Feature Correlation Threshold",
                options=self.FEATURE_CORR_THRESHOLDS_P4,
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.RESIDUAL_INTERP_METHOD,
                "📍 [4] Spatial Residual Interpolation Method",
                options=[
                    "Standard KNN",
                    "Robust KNN (Huber Weights)",
                    "Gaussian Process / Kriging",
                ],
                defaultValue=0,
            )
        )

        p_knn = QgsProcessingParameterNumber(
            self.KNN_NEIGHBORS,
            "📍 [Phase 04] KNN Nearest Neighbors (K) for Residuals",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=15,
            minValue=1,
            maxValue=100,
        )
        p_knn.setFlags(p_knn.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_knn)

        p_gpr = QgsProcessingParameterNumber(
            self.MAX_GPR_SAMPLES,
            "📍 [Phase 04] Max GPR Training Samples",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=1500,
        )
        p_gpr.setFlags(p_gpr.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_gpr)

        p_sp_p4 = QgsProcessingParameterBoolean(
            self.SPATIAL_CV_P4,
            "🌍 [Phase 04] Enable Spatial Block Cross-Validation",
            defaultValue=False,
        )
        p_sp_p4.setFlags(p_sp_p4.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_sp_p4)

        p_var_corr_p4 = QgsProcessingParameterBoolean(
            self.ENABLE_DEPTH_VARIANCE_CORR_P4,
            "🎛️ [Phase 04] Enable Depth Variance Correction",
            defaultValue=False,
        )
        p_var_corr_p4.setFlags(p_var_corr_p4.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_var_corr_p4)


        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_ADAPTIVE_TRAIN, "🎯 [4] Adaptive Points", optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_ADAPTIVE_DEPTH,
                "🎯 [4] Adaptive Depth Field",
                defaultValue="ortho_h",
                parentLayerParameterName=self.INPUT_ADAPTIVE_TRAIN,
                type=QgsProcessingParameterField.Numeric,
                optional=True,
            )
        )

        # --- SDB Composite Score & Model Selection Strategy ---
        p_strat = QgsProcessingParameterEnum(
            self.SCORE_SELECTION_STRATEGY,
            "🎯 [Auto-ML Ranking] Model Selection Strategy / Criterion",
            options=self.SCORE_STRATEGY_OPTIONS,
            defaultValue=0,
        )
        p_strat.setFlags(p_strat.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_strat)

        p_metrics = QgsProcessingParameterEnum(
            self.SCORE_METRICS,
            "⚖️ [Score Equation] Included Evaluation Metrics (Auto-Balanced)",
            options=self.SCORE_METRIC_OPTIONS,
            allowMultiple=True,
            defaultValue=[0, 1, 2, 3],
        )
        p_metrics.setFlags(p_metrics.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_metrics)

        p_custom_cfg = QgsProcessingParameterString(
            self.SCORE_CUSTOM_CONFIG,
            "🎛️ [Custom Score Matrix] Optional Weights (e.g. 'R2: 50, MAE: 50') & Simulation Settings",
            defaultValue="R2: 35, RMSE: 30, wMAPE: 20, Bias: 15, Rounds: 20, Variation: +/-35%",
            optional=True,
        )
        p_custom_cfg.setFlags(p_custom_cfg.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_custom_cfg)

        # -------------------------------------------------------------------
        # [5] Phase 05: Validation & Reporting (Matching SpatioSpectral Flow)
        # -------------------------------------------------------------------
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_VALIDATION,
                "📉 [5] Enable Phase 05 Validation & Reporting",
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_TEST, "📉 [5] Validation Points", optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_TEST_DEPTH,
                "📉 [5] Validation Depth Field",
                defaultValue="ortho_h",
                parentLayerParameterName=self.INPUT_TEST,
                type=QgsProcessingParameterField.Numeric,
                optional=True,
            )
        )

        # -------------------------------------------------------------------
        # Cleanup & Thresholds (Matching SpatioSpectral Flow)
        # -------------------------------------------------------------------
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_DEPTH_THRESHOLD,
                "🛑 Maximum Depth Threshold (e.g. -30)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=-30.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.REMOVE_POSITIVES,
                "🧽 [Cleanup] Remove Positive Depths (>= 0)",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_SLOPE_FILTER,
                "🧽 [Cleanup] Apply Slope Filter (Remove sharp jumps)",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SLOPE_THRESHOLD,
                "🧽 [Cleanup] Slope Filter Threshold (Degrees)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=35.0,
            )
        )

    def name(self):
        return "sdb_post_spatiospectral_aggregator"

    def displayName(self):
        return "Post-SpatioSpectral Aggregator"

    def group(self):
        return "SDB Research Tools"

    def groupId(self):
        return "sdb_tools"

    def createInstance(self):
        return PostSpatioSpectralAggregator()

    def shortHelpString(self):
        return """
        <div style="font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.5; color: #2C3E50;">
            <h2 style="margin-bottom: 5px; color: #D35400;">🧩 Bathymetrix-AI: Post-SpatioSpectral Aggregator</h2>
            <p style="margin-top: 0; margin-bottom: 15px; font-size: 13px;">
                Aggregates multiple Satellite-Derived Bathymetry (SDB) depth maps generated across different scenes into a single, highly accurate consensus model.
                Allows re-running or fine-tuning the aggregation step using a completed <b>SDB SpatioSpectral Masterflow</b> workspace without re-executing Phase 01–03.
            </p>
            
            <h3 style="color: #117A65; margin-bottom: 5px; border-bottom: 2px solid #117A65; padding-bottom: 3px;">📊 Aggregation Methods</h3>
            <ul style="font-size: 12px; margin-top: 5px; padding-left: 20px;">
                <li><b>Weighted Median (R2/RMSE) [Recommended]:</b> Combines all depth maps, giving higher weight to scenes with stronger training accuracy while remaining robust against local outliers.</li>
                <li><b>Weighted Mean (R2/RMSE):</b> Accuracy-weighted linear average with weights automatically extracted from the Masterflow log file.</li>
                <li><b>Select Best Scene (High R2 / Low RMSE):</b> Automatically selects the single best-performing scene based on training metrics.</li>
                <li><b>Median / Mean:</b> Combines all scenes equally using standard statistical central tendencies.</li>
                <li><b>Max (Deepest) / Min (Shallowest):</b> Extreme depth envelope extractions.</li>
            </ul>

            <h3 style="color: #8E44AD; margin-top: 15px; margin-bottom: 5px; border-bottom: 2px solid #8E44AD; padding-bottom: 3px;">📌 Downstream Phases</h3>
            <ul style="font-size: 12px; margin-top: 5px; padding-left: 20px;">
                <li><b>Phase 04 Adaptive Refinement:</b> Spatial residual calibration applied directly to the synthesized consensus map.</li>
                <li><b>Phase 05 Validation:</b> Independent accuracy assessment with automated <b>IHO S-44 (Order 1a/1b/2 TVU)</b> compliance reporting and HTML Dashboard.</li>
            </ul>
            <br><b>Developer:</b> Mohamed Aly Nasef
        </div>
        """

    def helpString(self):
        return self.shortHelpString()

    def processAlgorithm(self, parameters, context, feedback):
        import time
        start_time = time.time()

        workspace = self.parameterAsString(parameters, self.INPUT_WORKSPACE, context)
        agg_method_idx = self.parameterAsInt(
            parameters, self.SPATIOSPECTRAL_AGGREGATION, context
        )
        agg_method = self.AGGREGATION_METHODS[agg_method_idx]

        if not os.path.isdir(workspace):
            raise QgsProcessingException(f"Workspace directory not found: {workspace}")

        # 1. Discover depth maps and masks
        scene_folders = sorted([f for f in os.listdir(workspace) if f.startswith("Scene_")])
        if not scene_folders:
            raise QgsProcessingException(f"No Scene_* folders found in workspace: {workspace}")

        p3_depth_maps = []
        p1_masks = []

        feedback.pushInfo(f"Found {len(scene_folders)} scenes in workspace.")

        for scene_folder in scene_folders:
            scene_path = os.path.join(workspace, scene_folder)

            # Find Depth Map
            p3_dir = os.path.join(scene_path, "Phase_03_Initial_Modeling")
            depth_map = os.path.join(p3_dir, "3_Initial_Global_Depth_NoPositives.tif")
            if not os.path.exists(depth_map):
                depth_map = os.path.join(p3_dir, "3_Initial_Global_Depth_Cleaned.tif")
            if not os.path.exists(depth_map):
                depth_map = os.path.join(p3_dir, "3_Initial_Global_Depth.tif")

            if not os.path.exists(depth_map):
                feedback.pushInfo(f"Missing depth map for {scene_folder}. Skipping.")
                continue

            p3_depth_maps.append(depth_map)

            # Find Mask
            p1_dir = os.path.join(scene_path, "Phase_01_Preprocessing")
            mask_files = glob.glob(os.path.join(p1_dir, "*Mask*.tif"))
            if mask_files:
                p1_masks.append(mask_files[0])
            else:
                feedback.pushInfo(f"Missing mask for {scene_folder}.")

        if not p3_depth_maps:
            raise QgsProcessingException("No valid Phase 03 depth maps found in the workspace.")

        # 2. Extract Weights if needed
        p3_weights = []
        if "Weighted" in agg_method or "Best Scene" in agg_method:
            log_path = os.path.join(workspace, "SpatioSpectral_Master_Log.txt")
            if not os.path.exists(log_path):
                log_path = os.path.join(workspace, "SpatioSpectral_Flow_Log.txt")  # Fallback

            if not os.path.exists(log_path):
                raise QgsProcessingException(
                    f"Log file not found in {workspace}. Cannot extract weights for {agg_method}."
                )

            with open(log_path, "r", encoding="utf-8") as f:
                content = f.read()

            if "Started: " in content:
                content = content.split("Started: ")[-1]

            weights = re.findall(r"Weight:\s*([0-9.]+)", content)
            if len(weights) >= len(p3_depth_maps):
                weights = weights[: len(p3_depth_maps)]

            if not weights:
                r2s = re.findall(r"'BEST_R2':\s*(?:np\.float64\()?([0-9.]+)", content)
                rmses = re.findall(r"'BEST_RMSE':\s*(?:np\.float64\()?([0-9.]+)", content)

                if len(r2s) >= len(p3_depth_maps) and len(rmses) >= len(p3_depth_maps):
                    weights = []
                    for i in range(len(p3_depth_maps)):
                        r2 = float(r2s[i])
                        rmse = float(rmses[i])
                        w = max(0.001, r2) / (rmse + 0.001)
                        weights.append(str(w))

            if len(weights) != len(p3_depth_maps):
                raise QgsProcessingException(
                    f"Extracted {len(weights)} weights from log, but found {len(p3_depth_maps)} depth maps. Mismatch!"
                )

            p3_weights = [float(w) for w in weights]
            feedback.pushInfo(f"Extracted/Calculated Weights: {p3_weights}")

        # 3. Output directory
        out_folder = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        if not out_folder:
            out_folder = os.path.join(workspace, "Aggregated_Results")

        os.makedirs(out_folder, exist_ok=True)
        agg_log_path = os.path.join(out_folder, "Post_Aggregator_Log.txt")

        safe_agg_method_name = (
            agg_method.replace("/", "_").replace("\\", "_").replace(" ", "_")
        )
        aggregated_depth_path = os.path.join(
            out_folder, f"Aggregated_Depth_{safe_agg_method_name}.tif"
        )
        aggregated_mask_path = os.path.join(out_folder, "Aggregated_Intersection_Mask.tif")

        # 4. Run Aggregation
        if "Best Scene" in agg_method:
            best_idx = p3_weights.index(max(p3_weights))
            best_depth_map = p3_depth_maps[best_idx]
            feedback.pushInfo(
                f"Selecting Scene {best_idx+1} as the best scene based on R2/RMSE weight ({p3_weights[best_idx]:.4f})."
            )

            import shutil
            shutil.copy2(best_depth_map, aggregated_depth_path)

            feedback.pushInfo("Copying mask for the best scene...")
            if p1_masks and best_idx < len(p1_masks):
                shutil.copy2(p1_masks[best_idx], aggregated_mask_path)
        else:
            feedback.pushInfo(f"Aggregating using method: {agg_method}...")
            agg_kwargs = {"method": agg_method, "feedback": feedback}
            if "Weighted" in agg_method:
                agg_kwargs["weights"] = p3_weights

            spatiospectral_aggregate(p3_depth_maps, aggregated_depth_path, **agg_kwargs)

            feedback.pushInfo("Generating aggregated intersection mask...")
            if p1_masks:
                spatiospectral_mask_intersection(p1_masks, aggregated_mask_path, feedback=feedback)

        feedback.pushInfo(f"Aggregation complete. Output saved to: {aggregated_depth_path}")
        write_qml_style(aggregated_depth_path)

        # 5. Optional Phase 04: Adaptive Refinement (Matching SpatioSpectral Flow)
        enable_adaptive = self.parameterAsBool(parameters, self.ENABLE_ADAPTIVE, context)
        p4_final_depth = None
        p4_dir = None

        if enable_adaptive:
            adaptive_train_layer = self.parameterAsVectorLayer(
                parameters, self.INPUT_ADAPTIVE_TRAIN, context
            )
            if not adaptive_train_layer:
                raise QgsProcessingException(
                    "Adaptive Points layer (INPUT_ADAPTIVE_TRAIN) is required to run Phase 04."
                )

            feedback.pushInfo("Running Phase 04 (Adaptive Refinement)...")
            p4_dir = os.path.join(out_folder, "Phase_04_Adaptive_Refinement")
            os.makedirs(p4_dir, exist_ok=True)

            stack_indices = self.parameterAsEnums(
                parameters, self.STACK_COMPONENTS_P4, context
            )
            # Map [0, 1] to 1-based band indexes [1, 2]
            stack_components = [idx + 1 for idx in stack_indices] if stack_indices else [1, 2]

            p4_thresh_idx = self.parameterAsInt(parameters, self.FEATURE_CORR_THRESHOLD_P4, context)
            mapped_thresh = max(0, p4_thresh_idx - 1) if p4_thresh_idx > 0 else 0

            p4_params = {
                "INPUT_GLOBAL_RASTER": aggregated_depth_path,
                "INPUT_ORIGINAL_FEAT": aggregated_depth_path,
                "STACK_COMPONENTS": stack_components,
                "INPUT_MASK": aggregated_mask_path if os.path.exists(aggregated_mask_path) else None,
                "INPUT_TRAIN": adaptive_train_layer,
                "FIELD_TRAIN": self.parameterAsString(parameters, self.FIELD_ADAPTIVE_DEPTH, context),
                "FEATURE_CORR_METHOD": self.parameterAsInt(parameters, self.FEATURE_CORR_METHOD_P4, context),
                "FEATURE_CORR_THRESHOLD": mapped_thresh,
                "RESIDUAL_INTERP_METHOD": self.parameterAsInt(parameters, self.RESIDUAL_INTERP_METHOD, context),
                "KNN_NEIGHBORS": self.parameterAsInt(parameters, self.KNN_NEIGHBORS, context),
                "MAX_GPR_SAMPLES": self.parameterAsInt(parameters, self.MAX_GPR_SAMPLES, context),
                "SPATIAL_CV": self.parameterAsBool(parameters, self.SPATIAL_CV_P4, context),
                "ENABLE_DEPTH_VARIANCE_CORR": self.parameterAsBool(parameters, self.ENABLE_DEPTH_VARIANCE_CORR_P4, context),
                "OUTPUT_FOLDER": p4_dir,
            }



            p4 = processing.run(
                "sdb_tools:sdb_phase4_adaptive",
                p4_params,
                is_child_algorithm=True,
                context=context,
                feedback=feedback,
            )
            raw_p4_depth = p4["OUTPUT_FINAL"]

            if raw_p4_depth and os.path.exists(raw_p4_depth):
                max_depth = self.parameterAsDouble(parameters, self.MAX_DEPTH_THRESHOLD, context)
                p4_clamped = os.path.join(p4_dir, "4_Phase04_Depth_Cleaned.tif")
                clean_depth_map(
                    raw_p4_depth, aggregated_depth_path, max_depth, p4_clamped, context, feedback
                )
                current_p4 = p4_clamped

                if self.parameterAsBool(parameters, self.ENABLE_SLOPE_FILTER, context):
                    slope_threshold = self.parameterAsDouble(
                        parameters, self.SLOPE_THRESHOLD, context
                    )
                    p4_slope = os.path.join(p4_dir, "4_Phase04_Depth_SlopeFiltered.tif")
                    current_p4 = slope_filter_depth(
                        current_p4,
                        slope_threshold=slope_threshold,
                        out_path=p4_slope,
                        context=context,
                        feedback=feedback,
                    )

                if self.parameterAsBool(parameters, self.REMOVE_POSITIVES, context):
                    p4_no_pos = os.path.join(p4_dir, "4_Phase04_Depth_NoPositives.tif")
                    remove_positive_pixels(current_p4, p4_no_pos, feedback)
                    p4_final_depth = p4_no_pos
                else:
                    p4_final_depth = current_p4

                write_qml_style(p4_final_depth)
            else:
                p4_final_depth = raw_p4_depth

            feedback.pushInfo(f"Phase 04 complete. Output saved to: {p4_final_depth}")

        # 6. Optional Phase 05: Validation & Reporting (Matching SpatioSpectral Flow)
        enable_val = self.parameterAsBool(parameters, self.ENABLE_VALIDATION, context)
        p5_dir = None

        if enable_val:
            test_layer = self.parameterAsVectorLayer(parameters, self.INPUT_TEST, context)
            train_ref = self.parameterAsVectorLayer(
                parameters, self.INPUT_ADAPTIVE_TRAIN, context
            )

            feedback.pushInfo("Running Phase 05 (Validation & Reporting)...")
            p5_dir = os.path.join(out_folder, "Phase_05_Scientific_Validation")
            os.makedirs(p5_dir, exist_ok=True)

            p5_params = {
                "INPUT_MAP_P3": aggregated_depth_path,
                "INPUT_MAP_P4": p4_final_depth if p4_final_depth else aggregated_depth_path,
                "INPUT_TRAIN": train_ref,
                "FIELD_TRAIN": self.parameterAsString(parameters, self.FIELD_ADAPTIVE_DEPTH, context) if train_ref else "ortho_h",
                "INPUT_VALIDATION": test_layer,
                "FIELD_VAL_DEPTH": self.parameterAsString(parameters, self.FIELD_TEST_DEPTH, context) if test_layer else "ortho_h",
                "OUTPUT_FOLDER": p5_dir,
            }

            processing.run(
                "sdb_tools:sdb_05_reporting",
                p5_params,
                is_child_algorithm=True,
                context=context,
                feedback=feedback,
            )
            feedback.pushInfo(f"Phase 05 complete. Reports & IHO S-44 analysis saved to: {p5_dir}")

            # Generate static 3D Seabed PNG
            final_depth_for_3d = p4_final_depth if (enable_adaptive and p4_final_depth) else aggregated_depth_path
            if final_depth_for_3d and os.path.exists(final_depth_for_3d):
                try:
                    out_3d_png = os.path.join(p5_dir, "5_Plot_3D_Seabed.png")
                    generate_3d_seabed_png(final_depth_for_3d, out_3d_png, feedback)
                except Exception as e:
                    feedback.pushWarning(f"3D Seabed plot failed: {str(e)}")

            feedback.pushInfo("Generating Interactive HTML Dashboard & S-44 Report...")
            try:
                generate_html_dashboard(
                    out_dir=out_folder,
                    p3_dir=out_folder,
                    p4_dir=p4_dir if enable_adaptive else None,
                    spatial_cv_p3=False,
                    spatial_cv_p4=self.parameterAsBool(parameters, self.SPATIAL_CV_P4, context),
                    field_depth=self.parameterAsString(parameters, self.FIELD_ADAPTIVE_DEPTH, context) if train_ref else None,
                    feedback=feedback,
                    raster_name="SpatioSpectral Aggregated Scenes",
                    train_name=train_ref.name() if train_ref else "Adaptive Points",
                    test_name=test_layer.name() if test_layer else "Validation Points",
                    final_raster_path=p4_final_depth if (enable_adaptive and p4_final_depth) else aggregated_depth_path,
                    is_spatiospectral=True,
                    p5_dir=p5_dir,
                )
            except Exception as e:
                feedback.pushWarning(f"Dashboard generation failed: {str(e)}")

        # 7. Layer loading with unified ocean styling on map canvas
        if aggregated_depth_path and os.path.exists(aggregated_depth_path):
            qml_agg = write_qml_style(aggregated_depth_path)
            add_raster_to_canvas(
                aggregated_depth_path,
                f"Aggregated Depth ({safe_agg_method_name})",
                context=context,
                style_path=qml_agg,
            )

        if enable_adaptive and p4_final_depth and os.path.exists(p4_final_depth):
            qml_p4 = write_qml_style(p4_final_depth)
            add_raster_to_canvas(
                p4_final_depth,
                f"Phase 4 Final Depth ({safe_agg_method_name})",
                context=context,
                style_path=qml_p4,
            )

        if aggregated_mask_path and os.path.exists(aggregated_mask_path):
            add_raster_to_canvas(aggregated_mask_path, "Aggregated Intersection Mask", context=context)


        # 8. Log module completion banner with clickable URLs
        elapsed = time.time() - start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)

        dash_file = os.path.join(out_folder, "SDB_Validation_Dashboard.html")
        if not os.path.exists(dash_file) and p5_dir:
            dash_file = os.path.join(p5_dir, "SDB_Validation_Dashboard.html")

        strat_file = os.path.join(p5_dir, "5_Stratified_Error_Analysis.csv") if p5_dir else None

        primary_files = {
            "Aggregated Depth Map": aggregated_depth_path,
            "Refined Depth Map": p4_final_depth,
            "Intersection Mask": aggregated_mask_path,
            "HTML Dashboard": dash_file if os.path.exists(dash_file) else None,
            "IHO S-44 Assessment CSV": strat_file if strat_file and os.path.exists(strat_file) else None,
        }

        try:
            log_module_completion(
                module_title=f"Post-SpatioSpectral Aggregator ({agg_method} - Elapsed: {mins}m {secs}s)",
                out_dir=out_folder,
                primary_files=primary_files,
                log_path=agg_log_path,
                feedback=feedback,
            )
        except Exception:
            pass

        return {
            "OUTPUT_DEPTH": p4_final_depth if (enable_adaptive and p4_final_depth) else aggregated_depth_path,
            "OUTPUT_FOLDER": out_folder,
        }
