# SDB_Master_Orchestrator.py
# ---------------------------------------------------------------------------
# SDB MASTER ORCHESTRATOR (FULL LOGGING & AUTO-REPROJECT EDITION)
# Features:
# - Orchestrates Modules 01 to 05.
# - Automatically detects and reprojects input vectors if their CRS
#   does not match the input raster, preventing common errors.
# - Generates a "FULL_PROJECT_LOG.txt".
# ---------------------------------------------------------------------------

import os
import time
import datetime
import warnings
from qgis.core import (
    QgsProcessing, QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer, QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField, QgsProcessingParameterNumber,
    QgsProcessingParameterBand, QgsProcessingParameterFolderDestination,
    QgsProcessingParameterBoolean, QgsProject,
    QgsProcessingParameterEnum, QgsProcessingException, QgsCoordinateReferenceSystem
)
import processing

warnings.filterwarnings("ignore")

class SDBMasterOrchestrator(QgsProcessingAlgorithm):
    # --- INPUT CONSTANTS ---
    INPUT_RASTER = 'INPUT_RASTER'
    OUTPUT_FOLDER = 'OUTPUT_FOLDER'
    
    # Phase 1
    COASTAL_BAND = 'COASTAL_BAND'
    BLUE_BAND = 'BLUE_BAND'; GREEN_BAND = 'GREEN_BAND'
    RED_BAND = 'RED_BAND'; NIR_BAND = 'NIR_BAND'
    APPLY_SUNGLINT = 'APPLY_SUNGLINT'; NIR_BAND_SUNGLINT = 'NIR_BAND_SUNGLINT'
    DEEP_WATER_POLY = 'DEEP_WATER_POLY'
    APPLY_WATER_MASK = 'APPLY_WATER_MASK'

    # Phase 2
    RANSAC_THRESHOLD = 'RANSAC_THRESHOLD'

    # Phase 3 & 4
    INPUT_TRAIN = 'INPUT_TRAIN'
    FIELD_DEPTH = 'FIELD_DEPTH'
    FIELD_WEIGHT = 'FIELD_WEIGHT'
    
    SELECTED_ALGOS = 'SELECTED_ALGOS'
    N_ITERATIONS = 'N_ITERATIONS'
    MEDIAN_SIZE = 'MEDIAN_SIZE'

    # Phase 5
    INPUT_TEST = 'INPUT_TEST'
    FIELD_TEST_DEPTH = 'FIELD_TEST_DEPTH'

    # Algorithm List (Mapped to indices)
    MODEL_LIST = [
        'Linear Regression',    # 0
        'Random Forest',        # 1
        'Gradient Boosting',    # 2
        'Extra Trees',          # 3
        'Ridge',                # 4
        'Lasso',                # 5
        'ElasticNet',           # 6
        'KNN',                  # 7
        'Decision Tree',        # 8
        'MLP (Neural Net)',     # 9
        'SVR'                   # 10
    ]

    def initAlgorithm(self, config=None):
        # 1. Main
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_RASTER, 'Input Satellite Image (Sentinel-2)'))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, 'Main Output Folder'))
        
        # 2. Bands
        self.addParameter(QgsProcessingParameterBand(self.COASTAL_BAND, 'Coastal Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=1))
        self.addParameter(QgsProcessingParameterBand(self.BLUE_BAND, 'Blue Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=2))
        self.addParameter(QgsProcessingParameterBand(self.GREEN_BAND, 'Green Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=3))
        self.addParameter(QgsProcessingParameterBand(self.RED_BAND, 'Red Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=4))
        self.addParameter(QgsProcessingParameterBand(self.NIR_BAND, 'NIR Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=8))

        # 3. Pre-processing
        self.addParameter(QgsProcessingParameterBoolean(self.APPLY_SUNGLINT, 'Apply Sunglint Correction', defaultValue=True))
        self.addParameter(QgsProcessingParameterBand(self.NIR_BAND_SUNGLINT, 'Sunglint NIR Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=8))
        self.addParameter(QgsProcessingParameterVectorLayer(self.DEEP_WATER_POLY, 'Deep Water ROI (Optional)', optional=True))
        self.addParameter(QgsProcessingParameterBoolean(self.APPLY_WATER_MASK, 'Apply Water Mask', defaultValue=True))
        
        # 4. Training
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_TRAIN, 'Training Points (ICESat-2)'))
        self.addParameter(QgsProcessingParameterField(self.FIELD_DEPTH, 'Depth Field', parentLayerParameterName=self.INPUT_TRAIN, type=QgsProcessingParameterField.Numeric))
        self.addParameter(QgsProcessingParameterField(self.FIELD_WEIGHT, 'Confidence/Weight Field (Optional)', parentLayerParameterName=self.INPUT_TRAIN, type=QgsProcessingParameterField.Numeric, optional=True))
        self.addParameter(QgsProcessingParameterNumber(self.RANSAC_THRESHOLD, 'RANSAC Outlier Threshold (0 = Auto)', type=QgsProcessingParameterNumber.Double, defaultValue=0.0))

        # 5. Modeling
        self.addParameter(QgsProcessingParameterEnum(self.SELECTED_ALGOS, 'Select Algorithms (For Training)', 
                                                     options=self.MODEL_LIST, allowMultiple=True, defaultValue=[0, 1, 2, 3]))
        self.addParameter(QgsProcessingParameterNumber(self.N_ITERATIONS, 'Optimization Iterations (0=Default)', 
                                                       type=QgsProcessingParameterNumber.Integer, defaultValue=10, minValue=0))
        self.addParameter(QgsProcessingParameterNumber(self.MEDIAN_SIZE, 'Median Filter Size (Smoothing)', 
                                                       type=QgsProcessingParameterNumber.Integer, defaultValue=3, minValue=1))
        
        # 6. Validation
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_TEST, 'Unseen Validation Points'))
        self.addParameter(QgsProcessingParameterField(self.FIELD_TEST_DEPTH, 'Validation Depth Field', parentLayerParameterName=self.INPUT_TEST, type=QgsProcessingParameterField.Numeric))

    def name(self): return 'sdb_master_orchestrator'
    def displayName(self): return 'SDB Master Workflow (Full Pipeline & Log)'
    def shortHelpString(self):
        return """
        <div style="color: #333333; font-family: 'Segoe UI', Arial, sans-serif;">
            <h2>🌊 SDB Master Workflow 🚀</h2>
            <p>
                This tool fully automates the Satellite Derived Bathymetry (SDB) process.
                It takes a satellite image and depth points to produce a final bathymetric map.
                <br><b>It now automatically handles CRS differences between inputs!</b>
            </p>
            <h3>✨ The 5-Phase Journey:</h3>
            <ol>
                <li><b>Phase 1: Pre-processing</b> 🛰️</li>
                <li><b>Phase 2: Data Filtering (RANSAC)</b> 📊</li>
                <li><b>Phase 3: Global Modeling (Machine Learning)</b> 🧠</li>
                <li><b>Phase 4: Spatial Stacking & Refinement</b> 📈</li>
                <li><b>Phase 5: Final Reporting & Validation</b> 📝</li>
            </ol>
            <h3>🔬 Scientific Foundation:</h3>
            <p>
                Based on physics-based radiative transfer and ML regression, inspired by Stumpf et al. (2003) and modern machine learning approaches.
            </p>
        </div>
        """
        
    def createInstance(self): return SDBMasterOrchestrator()

    def reproject_layer_if_needed(self, vector_layer, target_crs, temp_output_path, context, feedback):
        """
        Checks if a vector layer's CRS matches the target CRS. If not, it reprojects it.
        Returns the path to the layer that should be used for processing.
        """
        if not vector_layer:
            return None
            
        source_crs = vector_layer.crs()
        if source_crs == target_crs:
            feedback.pushInfo(f"✔ CRS for '{vector_layer.name()}' matches the raster. No reprojection needed.")
            return vector_layer.source()  # Return the original path
        else:
            feedback.pushWarning(f"CRS MISMATCH DETECTED for layer '{vector_layer.name()}'.")
            feedback.pushInfo(f"  > Points CRS: {source_crs.authid()} | Raster CRS: {target_crs.authid()}")
            feedback.pushInfo("  > Automatically reprojecting layer to a temporary file...")
            
            reproject_params = {
                'INPUT': vector_layer,
                'TARGET_CRS': target_crs,
                'OUTPUT': temp_output_path
            }
            result = processing.run("native:reprojectlayer", reproject_params, context=context, feedback=feedback, is_child_algorithm=True)
            feedback.pushInfo("  > Reprojection successful.")
            return result['OUTPUT']


    def processAlgorithm(self, parameters, context, feedback):
        total_start_time = time.time()
        start_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        out_dir = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        
        config_info = self.get_config_info(parameters, context)
        timing_log = {}
        
        # =====================================================================
        # CRS CHECK AND AUTOMATIC REPROJECTION
        # =====================================================================
        feedback.pushInfo(">>> ORCHESTRATOR: Checking Coordinate Reference Systems...")
        input_raster = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        target_crs = input_raster.crs()

        train_layer = self.parameterAsVectorLayer(parameters, self.INPUT_TRAIN, context)
        test_layer = self.parameterAsVectorLayer(parameters, self.INPUT_TEST, context)

        temp_train_path = os.path.join(out_dir, 'temp_reprojected_train.gpkg')
        temp_test_path = os.path.join(out_dir, 'temp_reprojected_test.gpkg')

        # Get the correct paths (either original or reprojected)
        final_train_path = self.reproject_layer_if_needed(train_layer, target_crs, temp_train_path, context, feedback)
        final_test_path = self.reproject_layer_if_needed(test_layer, target_crs, temp_test_path, context, feedback)
        
        # =====================================================================
        # STEP 1: PRE-PROCESSING
        # =====================================================================
        feedback.pushInfo("\n>>> ORCHESTRATOR: [Step 1/5] Pre-processing...")
        t0 = time.time()
        
        p1_params = {
            'INPUT_RASTER': input_raster,
            'COASTAL_BAND': self.parameterAsInt(parameters, self.COASTAL_BAND, context),
            'BLUE_BAND': self.parameterAsInt(parameters, self.BLUE_BAND, context),
            'GREEN_BAND': self.parameterAsInt(parameters, self.GREEN_BAND, context),
            'RED_BAND': self.parameterAsInt(parameters, self.RED_BAND, context),
            'NIR_BAND': self.parameterAsInt(parameters, self.NIR_BAND, context),
            'APPLY_SUNGLINT': self.parameterAsBool(parameters, self.APPLY_SUNGLINT, context),
            'NIR_BAND_SUNGLINT': self.parameterAsInt(parameters, self.NIR_BAND_SUNGLINT, context),
            'DEEP_WATER_POLY': self.parameterAsVectorLayer(parameters, self.DEEP_WATER_POLY, context),
            'APPLY_WATER_MASK': self.parameterAsBool(parameters, self.APPLY_WATER_MASK, context),
            'OUTPUT_FOLDER': out_dir
        }
        p1 = processing.run("sdb_tools:sdb_phase1_preprocessing", p1_params, context=context, feedback=feedback, is_child_algorithm=True)
        timing_log['Phase 1 (Pre-processing)'] = time.time() - t0
        
        path_features = p1['OUTPUT_FEATURES']
        path_mask = p1['OUTPUT_MASK']
        
        # =====================================================================
        # STEP 2: DATA FILTERING
        # =====================================================================
        feedback.pushInfo("\n>>> ORCHESTRATOR: [Step 2/5] Data Filtering (RANSAC)...")
        t0 = time.time()
        
        p2_params = {
            'INPUT_STACK': path_features,
            'INPUT_POINTS': final_train_path,  # USE THE REPROJECTED PATH
            'FIELD_DEPTH': self.parameterAsString(parameters, self.FIELD_DEPTH, context),
            'RATIO_BAND_INDEX': 11, 
            'RESIDUAL_THRESHOLD': self.parameterAsDouble(parameters, self.RANSAC_THRESHOLD, context),
            'OUTPUT_FOLDER': out_dir
        }
        p2 = processing.run("sdb_tools:sdb_02_filtering", p2_params, context=context, feedback=feedback, is_child_algorithm=True)
        timing_log['Phase 2 (Data Filtering)'] = time.time() - t0
        
        path_clean_points = p2['OUTPUT_CLEAN_VEC']

        # =====================================================================
        # STEP 3: INITIAL MODELING
        # =====================================================================
        feedback.pushInfo("\n>>> ORCHESTRATOR: [Step 3/5] Initial Global Modeling...")
        t0 = time.time()
        
        p3_params = {
            'INPUT_STACK': path_features,
            'INPUT_MASK': path_mask,
            'INPUT_POINTS': path_clean_points,
            'FIELD_DEPTH': self.parameterAsString(parameters, self.FIELD_DEPTH, context),
            'FIELD_WEIGHT': self.parameterAsString(parameters, self.FIELD_WEIGHT, context),
            'SELECTED_ALGOS': self.parameterAsEnums(parameters, self.SELECTED_ALGOS, context),
            'N_ITERATIONS': self.parameterAsInt(parameters, self.N_ITERATIONS, context),
            'MEDIAN_SIZE': self.parameterAsInt(parameters, self.MEDIAN_SIZE, context),
            'OUTPUT_FOLDER': out_dir
        }
        p3 = processing.run("sdb_tools:sdb_03_initial_modeling", p3_params, context=context, feedback=feedback, is_child_algorithm=True)
        timing_log['Phase 3 (Global Modeling)'] = time.time() - t0
        
        path_initial_depth = p3['OUTPUT_DEPTH_MAP']

        # =====================================================================
        # STEP 4: SPATIAL STACKING
        # =====================================================================
        feedback.pushInfo("\n>>> ORCHESTRATOR: [Step 4/5] Spatial Stacking & Refinement...")
        t0 = time.time()
        
        p4_params = {
            'INPUT_GLOBAL_RASTER': path_initial_depth,
            'INPUT_ORIGINAL_FEAT': path_features,
            'INPUT_MASK': path_mask,
            'INPUT_TRAIN': path_clean_points,
            'FIELD_TRAIN': self.parameterAsString(parameters, self.FIELD_DEPTH, context),
            'INPUT_VALIDATION': final_test_path, # USE THE REPROJECTED PATH
            'FIELD_VALIDATION': self.parameterAsString(parameters, self.FIELD_TEST_DEPTH, context),
            'SELECTED_ALGOS': self.parameterAsEnums(parameters, self.SELECTED_ALGOS, context),
            'N_ITERATIONS': self.parameterAsInt(parameters, self.N_ITERATIONS, context),
            'MEDIAN_SIZE': self.parameterAsInt(parameters, self.MEDIAN_SIZE, context),
            'OUTPUT_FOLDER': out_dir
        }
        p4 = processing.run("sdb_tools:sdb_phase4_adaptive", p4_params, context=context, feedback=feedback, is_child_algorithm=True)
        timing_log['Phase 4 (Spatial Refinement)'] = time.time() - t0
        
        path_refined_depth = p4['OUTPUT_FINAL']

        # =====================================================================
        # STEP 5: FINAL REPORTING
        # =====================================================================
        feedback.pushInfo("\n>>> ORCHESTRATOR: [Step 5/5] Final Comparative Reporting...")
        t0 = time.time()
        
        p5_params = {
            'INPUT_MAP_P3': path_initial_depth,
            'INPUT_MAP_P4': path_refined_depth,
            'INPUT_TRAIN': path_clean_points,
            'FIELD_TRAIN': self.parameterAsString(parameters, self.FIELD_DEPTH, context),
            'INPUT_VALIDATION': final_test_path, # USE THE REPROJECTED PATH
            'FIELD_VAL_DEPTH': self.parameterAsString(parameters, self.FIELD_TEST_DEPTH, context),
            'OUTPUT_FOLDER': out_dir
        }
        processing.run("sdb_tools:sdb_05_reporting", p5_params, context=context, feedback=feedback, is_child_algorithm=True)
        timing_log['Phase 5 (Reporting)'] = time.time() - t0

        # =====================================================================
        # FINAL LOG GENERATION
        # =====================================================================
        total_end_time = time.time()
        self.write_full_project_log(out_dir, start_dt, config_info, timing_log, total_end_time - total_start_time)
        
        feedback.pushInfo("\n>>> ORCHESTRATOR: WORKFLOW COMPLETED SUCCESSFULLY.")
        return {'FINAL_DEPTH_MAP': path_refined_depth}

    def get_config_info(self, params, context):
        indices = self.parameterAsEnums(params, self.SELECTED_ALGOS, context)
        names = [self.MODEL_LIST[i] for i in indices]
        
        return {
            'Algorithms': ", ".join(names),
            'Iterations': self.parameterAsInt(params, self.N_ITERATIONS, context),
            'Median Size': self.parameterAsInt(params, self.MEDIAN_SIZE, context),
            'RANSAC Threshold': self.parameterAsDouble(params, self.RANSAC_THRESHOLD, context),
            'Apply Sunglint': self.parameterAsBool(params, self.APPLY_SUNGLINT, context)
        }

    def write_full_project_log(self, out_dir, start_dt, config, timing, total_duration):
        log_path = os.path.join(out_dir, "FULL_PROJECT_LOG.txt")
        
        p5_summary = ""
        try:
            with open(os.path.join(out_dir, '5_FINAL_SUMMARY.txt'), 'r') as f:
                p5_summary = f.read()
        except:
            p5_summary = "Final summary file not found."

        with open(log_path, 'w') as f:
            f.write("======================================================\n")
            f.write("           SDB PROJECT EXECUTION LOG                 \n")
            f.write("======================================================\n\n")
            f.write(f"Date/Time Started: {start_dt}\n")
            f.write(f"Total Execution Time: {total_duration/60:.2f} minutes\n\n")
            
            f.write("--- CONFIGURATION ---\n")
            for k, v in config.items():
                f.write(f"{k:<20}: {v}\n")
            f.write("\n")
            
            f.write("--- TIMING BREAKDOWN ---\n")
            for k, v in timing.items():
                f.write(f"{k:<30}: {v:.2f} sec\n")
            f.write("\n")
            
            f.write("--- FINAL RESULTS SUMMARY ---\n")
            f.write(p5_summary)
            f.write("\n======================================================\n")