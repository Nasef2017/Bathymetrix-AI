import os
import glob
import re

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterFile,
    QgsProcessingParameterEnum,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFolderDestination,
    QgsProcessingException,
)
from qgis.core import QgsProcessing
from qgis import processing

from Bathymetrix_AI.core.spectral.aggregation import spatiospectral_aggregate, spatiospectral_mask_intersection
from Bathymetrix_AI.infrastructure.raster_io import clean_depth_map, remove_positive_pixels, slope_filter_depth
from Bathymetrix_AI.infrastructure.canvas import add_raster_to_canvas
from Bathymetrix_AI.core.pipeline import generate_html_dashboard

class PostSpatioSpectralAggregator(QgsProcessingAlgorithm):

    INPUT_WORKSPACE = "INPUT_WORKSPACE"
    AGGREGATION_METHOD = "AGGREGATION_METHOD"
    OUTPUT_FOLDER = "OUTPUT_FOLDER"

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_WORKSPACE, 
                "📁 SpatioSpectral Master Output Folder (Contains Scene_* folders)", 
                behavior=QgsProcessingParameterFile.Folder
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.AGGREGATION_METHOD,
                "🧩 Aggregation Method",
                options=["Median", "Mean", "Max (Deepest)", "Min (Shallowest)", "Weighted Median (R2/RMSE)", "Weighted Mean (R2/RMSE)", "Select Best Scene (Highest R2 / Lowest RMSE)"],
                defaultValue=4, # Weighted Median default
            )
        )
        
        self.addParameter(QgsProcessingParameterBoolean("RUN_PHASE_04", "Run Phase 04 (Adaptive Refinement)", defaultValue=False))
        self.addParameter(
            QgsProcessingParameterEnum(
                "RESIDUAL_INTERP_METHOD",
                "📍 Phase 04 Interpolation Method",
                options=["Standard KNN", "Robust KNN (Huber Weights)", "Gaussian Process / Kriging"],
                defaultValue=1,
                optional=True
            )
        )
        self.addParameter(QgsProcessingParameterBoolean("RUN_PHASE_05", "Run Phase 05 (Scientific Validation)", defaultValue=False))
        
        self.addParameter(QgsProcessingParameterVectorLayer("INPUT_TRAIN", "Training Points (Required if P4 or P5 checked)", optional=True))
        self.addParameter(QgsProcessingParameterField("FIELD_DEPTH", "Depth Field (Training)", defaultValue="depth", parentLayerParameterName="INPUT_TRAIN", type=QgsProcessingParameterField.Numeric, optional=True))
        self.addParameter(QgsProcessingParameterNumber("MAX_DEPTH_THRESHOLD", "Max Depth Threshold (e.g. -30) for P4", type=QgsProcessingParameterNumber.Double, defaultValue=-30.0, optional=True))
        self.addParameter(QgsProcessingParameterBoolean("REMOVE_POSITIVES", "🧽 [Cleanup] Remove positive depths (Land) for P4", defaultValue=True, optional=True))
        self.addParameter(QgsProcessingParameterBoolean("ENABLE_SLOPE_FILTER", "🧽 [Cleanup] Apply Slope Filter for P4 (Remove sharp jumps)", defaultValue=True, optional=True))
        self.addParameter(QgsProcessingParameterNumber("SLOPE_THRESHOLD", "🧽 [Cleanup] Slope Filter Threshold (Degrees) for P4", type=QgsProcessingParameterNumber.Double, defaultValue=35.0, optional=True))
        
        self.addParameter(QgsProcessingParameterVectorLayer("INPUT_VALIDATION", "Validation Points (Optional for P5)", optional=True))
        self.addParameter(QgsProcessingParameterField("FIELD_VAL_DEPTH", "Depth Field (Validation)", defaultValue="depth", parentLayerParameterName="INPUT_VALIDATION", type=QgsProcessingParameterField.Numeric, optional=True))
        
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, "📁 Output Folder"))

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

            <h3 style="color: #8E44AD; margin-top: 15px; margin-bottom: 5px; border-bottom: 2px solid #8E44AD; padding-bottom: 3px;">📌 Usage</h3>
            <p style="font-size: 12px; margin-top: 5px;">
                Select the master output folder generated by a previous SpatioSpectral Masterflow run (which contains <code>Scene_*</code> subfolders and <code>SpatioSpectral_Flow_Log.txt</code>).
            </p>
            <br><b>Developer:</b> Mohamed Aly Nasef
        </div>
        """

    def helpString(self):
        return self.shortHelpString()

    def processAlgorithm(self, parameters, context, feedback):
        workspace = self.parameterAsString(parameters, self.INPUT_WORKSPACE, context)
        agg_method_idx = self.parameterAsInt(parameters, self.AGGREGATION_METHOD, context)
        
        agg_methods = ["Median", "Mean", "Max (Deepest)", "Min (Shallowest)", "Weighted Median (R2/RMSE)", "Weighted Mean (R2/RMSE)", "Select Best Scene (Highest R2 / Lowest RMSE)"]
        agg_method = agg_methods[agg_method_idx]
        
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
                log_path = os.path.join(workspace, "SpatioSpectral_Flow_Log.txt") # Fallback to old name
                
            if not os.path.exists(log_path):
                raise QgsProcessingException(f"Log file not found in {workspace}. Cannot extract weights for {agg_method}.")
                
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Master Log is appended to. Only look at the latest run!
            if "Started: " in content:
                content = content.split("Started: ")[-1]
                
            # Extract weights using regex
            weights = re.findall(r"Weight:\s*([0-9.]+)", content)
            
            # Phase 4 might also print a weight at the end of the log, so we take exactly the first N matches
            # corresponding to Phase 3 of the scenes.
            if len(weights) >= len(p3_depth_maps):
                weights = weights[:len(p3_depth_maps)]
            
            if not weights:
                # Fallback: calculate from BEST_R2 and BEST_RMSE in the Results dictionary
                r2s = re.findall(r"'BEST_R2':\s*(?:np\.float64\()?([0-9.]+)", content)
                rmses = re.findall(r"'BEST_RMSE':\s*(?:np\.float64\()?([0-9.]+)", content)
                
                # Phase 4 also outputs BEST_R2/RMSE at the end of the log.
                # We only need the first N matches corresponding to the N scenes.
                if len(r2s) >= len(p3_depth_maps) and len(rmses) >= len(p3_depth_maps):
                    weights = []
                    for i in range(len(p3_depth_maps)):
                        r2 = float(r2s[i])
                        rmse = float(rmses[i])
                        w = max(0.001, r2) / (rmse + 0.001)
                        weights.append(str(w))
            
            if len(weights) != len(p3_depth_maps):
                raise QgsProcessingException(f"Extracted {len(weights)} weights (or R2/RMSE pairs) from log, but found {len(p3_depth_maps)} depth maps. Mismatch!")
                
            p3_weights = [float(w) for w in weights]
            feedback.pushInfo(f"Extracted/Calculated Weights: {p3_weights}")
            
        # 3. Create output directory
        out_folder = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        if not out_folder:
            out_folder = os.path.join(workspace, "Aggregated_Results")
            
        os.makedirs(out_folder, exist_ok=True)
        
        safe_agg_method_name = agg_method.replace("/", "_").replace("\\", "_").replace(" ", "_")
        aggregated_depth_path = os.path.join(out_folder, f"Aggregated_Depth_{safe_agg_method_name}.tif")
        aggregated_mask_path = os.path.join(out_folder, "Aggregated_Intersection_Mask.tif")
        
        # 4. Run Aggregation
        if agg_method == "Select Best Scene (Highest R2 / Lowest RMSE)":
            best_idx = p3_weights.index(max(p3_weights))
            best_depth_map = p3_depth_maps[best_idx]
            feedback.pushInfo(f"Selecting Scene {best_idx+1} as the best scene based on R2/RMSE weight ({p3_weights[best_idx]:.4f}).")
            
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
        
        # 5. Optional Phase 04
        run_p4 = self.parameterAsBool(parameters, "RUN_PHASE_04", context)
        p4_final_depth = None
        
        if run_p4:
            train_layer = self.parameterAsVectorLayer(parameters, "INPUT_TRAIN", context)
            if not train_layer:
                raise QgsProcessingException("Training Points layer is required to run Phase 04.")
                
            feedback.pushInfo("Running Phase 04 (Adaptive Refinement)...")
            p4_dir = os.path.join(out_folder, "Phase_04_Adaptive_Refinement")
            os.makedirs(p4_dir, exist_ok=True)
            
            p4_params = {
                "INPUT_GLOBAL_RASTER": aggregated_depth_path,
                "INPUT_ORIGINAL_FEAT": aggregated_depth_path, # Dummy for UI requirement
                "STACK_COMPONENTS": [1, 2],
                "INPUT_MASK": aggregated_mask_path,
                "INPUT_TRAIN": self.parameterAsVectorLayer(parameters, "INPUT_TRAIN", context),
                "FIELD_TRAIN": self.parameterAsString(parameters, "FIELD_DEPTH", context),
                "RESIDUAL_INTERP_METHOD": self.parameterAsInt(parameters, "RESIDUAL_INTERP_METHOD", context),
                "OUTPUT_FOLDER": p4_dir
            }
            
            p4 = processing.run("sdb_tools:sdb_phase4_adaptive", p4_params, is_child_algorithm=True, context=context, feedback=feedback)
            raw_p4_depth = p4["OUTPUT_FINAL"]
            
            if raw_p4_depth and os.path.exists(raw_p4_depth):
                max_depth = self.parameterAsDouble(parameters, "MAX_DEPTH_THRESHOLD", context)
                p4_clamped = os.path.join(p4_dir, "4_Phase04_Depth_Cleaned.tif")
                clean_depth_map(raw_p4_depth, aggregated_depth_path, max_depth, p4_clamped, context, feedback)
                current_p4 = p4_clamped
                
                if self.parameterAsBool(parameters, "ENABLE_SLOPE_FILTER", context):
                    slope_threshold = self.parameterAsDouble(parameters, "SLOPE_THRESHOLD", context)
                    p4_slope = os.path.join(p4_dir, "4_Phase04_Depth_SlopeFiltered.tif")
                    current_p4 = slope_filter_depth(
                        current_p4,
                        slope_threshold=slope_threshold,
                        out_path=p4_slope,
                        context=context,
                        feedback=feedback,
                    )
                
                if self.parameterAsBool(parameters, "REMOVE_POSITIVES", context):
                    p4_no_pos = os.path.join(p4_dir, "4_Phase04_Depth_NoPositives.tif")
                    remove_positive_pixels(current_p4, p4_no_pos, feedback)
                    p4_final_depth = p4_no_pos
                else:
                    p4_final_depth = current_p4
            else:
                p4_final_depth = raw_p4_depth
                
            feedback.pushInfo(f"Phase 04 complete. Output saved to: {p4_final_depth}")
            
        # 6. Optional Phase 05
        run_p5 = self.parameterAsBool(parameters, "RUN_PHASE_05", context)
        if run_p5:
            train_layer = self.parameterAsVectorLayer(parameters, "INPUT_TRAIN", context)
            if not train_layer:
                raise QgsProcessingException("Training Points layer is required to run Phase 05.")
                
            feedback.pushInfo("Running Phase 05 (Scientific Validation)...")
            p5_dir = os.path.join(out_folder, "Phase_05_Scientific_Validation")
            os.makedirs(p5_dir, exist_ok=True)
            
            p5_params = {
                "INPUT_MAP_P3": aggregated_depth_path,
                "INPUT_MAP_P4": p4_final_depth if p4_final_depth else aggregated_depth_path,
                "INPUT_TRAIN": self.parameterAsVectorLayer(parameters, "INPUT_TRAIN", context),
                "FIELD_TRAIN": self.parameterAsString(parameters, "FIELD_DEPTH", context),
                "INPUT_VALIDATION": self.parameterAsVectorLayer(parameters, "INPUT_VALIDATION", context),
                "FIELD_VAL_DEPTH": self.parameterAsString(parameters, "FIELD_VAL_DEPTH", context),
                "OUTPUT_FOLDER": p5_dir
            }
            
            processing.run("sdb_tools:sdb_05_reporting", p5_params, is_child_algorithm=True, context=context, feedback=feedback)
            feedback.pushInfo(f"Phase 05 complete. Reports saved to: {p5_dir}")
            
            feedback.pushInfo("Generating Interactive HTML Dashboard...")
            try:
                generate_html_dashboard(
                    out_dir=out_folder,
                    p3_dir=out_folder,  # Dummy p3_dir for SpatioSpectral since Phase 3 happens per-scene
                    p4_dir=p4_dir if run_p4 else None,
                    spatial_cv_p3=False,
                    spatial_cv_p4=self.parameterAsBool(parameters, "SPATIAL_CV_P4", context) if hasattr(self, "SPATIAL_CV_P4") else False,
                    field_depth=self.parameterAsString(parameters, "FIELD_DEPTH", context),
                    feedback=feedback,
                    raster_name="SpatioSpectral Aggregated Scenes",
                    train_name="Training Points",
                    test_name="Validation Points" if self.parameterAsVectorLayer(parameters, "INPUT_VALIDATION", context) else "Training Points (No Validation)",
                    final_raster_path=p4_final_depth if (run_p4 and p4_final_depth) else aggregated_depth_path
                )
            except Exception as e:
                feedback.pushWarning(f"Dashboard generation failed: {str(e)}")
            
        # Load final layers to map canvas
        add_raster_to_canvas(aggregated_depth_path, f"Aggregated Depth ({safe_agg_method_name})", context)
        add_raster_to_canvas(aggregated_mask_path, "Aggregated Intersection Mask", context)
        
        if run_p4 and p4_final_depth and os.path.exists(p4_final_depth):
            add_raster_to_canvas(p4_final_depth, f"Phase 4 Final Depth ({safe_agg_method_name})", context)
        
        return {}
