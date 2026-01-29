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
    QgsProcessingParameterEnum, QgsProcessingException, QgsCoordinateReferenceSystem,
    QgsRasterLayer, QgsProcessingParameterFile, QgsProcessingParameterString, QgsProcessingParameterDefinition
)
import processing
warnings.filterwarnings("ignore")

class SDBMasterOrchestrator(QgsProcessingAlgorithm):
    # --- CONSTANTS ---
    INPUT_RASTER = 'INPUT_RASTER'
    OUTPUT_FOLDER = 'OUTPUT_FOLDER'
    
    # Phase 1: Bands
    COASTAL_BAND = 'COASTAL_BAND'
    BLUE_BAND = 'BLUE_BAND'; GREEN_BAND = 'GREEN_BAND'
    RED_BAND = 'RED_BAND'; NIR_BAND = 'NIR_BAND'
    
    # Phase 1: Pre-processing Settings
    APPLY_SUNGLINT = 'APPLY_SUNGLINT'; NIR_BAND_SUNGLINT = 'NIR_BAND_SUNGLINT'
    SUNGLINT_PERCENTILE = 'SUNGLINT_PERCENTILE' # New
    
    DEEP_WATER_POLY = 'DEEP_WATER_POLY'
    
    MASKING_METHOD = 'MASKING_METHOD'
    MANUAL_THRESHOLD = 'MANUAL_THRESHOLD'
    OTSU_ADJUSTMENT = 'OTSU_ADJUSTMENT'         # New
    MASK_KERNEL_SIZE = 'MASK_KERNEL_SIZE'       # New
    NUM_THREADS = 'NUM_THREADS'                 # New

    # Phase 2: RANSAC Settings
    ENABLE_RANSAC = 'ENABLE_RANSAC'
    RANSAC_THRESHOLD = 'RANSAC_THRESHOLD'
    RANSAC_MIN_SAMPLES = 'RANSAC_MIN_SAMPLES' # New
    RANSAC_MAX_TRIALS = 'RANSAC_MAX_TRIALS'   # New

    # Phase 3: Training Data
    INPUT_TRAIN = 'INPUT_TRAIN'
    FIELD_DEPTH = 'FIELD_DEPTH'
    FIELD_WEIGHT = 'FIELD_WEIGHT'
    SELECTED_ALGOS = 'SELECTED_ALGOS'

    # Modeling Strategy
    OPTIMIZER_METHOD = 'OPTIMIZER_METHOD' 
    COLLISION_HANDLING = 'COLLISION_HANDLING'
    N_ITERATIONS = 'N_ITERATIONS'
    MEDIAN_SIZE = 'MEDIAN_SIZE'

    # Hyperparameters Strings (Advanced)
    PARAM_RF = 'PARAM_RF'
    PARAM_GB = 'PARAM_GB'
    PARAM_ET = 'PARAM_ET'
    PARAM_SVR = 'PARAM_SVR'
    PARAM_MLP = 'PARAM_MLP'

    # Phase 4: Adaptive
    ENABLE_ADAPTIVE = 'ENABLE_ADAPTIVE'
    INPUT_ADAPTIVE_TRAIN = 'INPUT_ADAPTIVE_TRAIN'
    FIELD_ADAPTIVE_DEPTH = 'FIELD_ADAPTIVE_DEPTH'

    # Phase 5: Validation
    ENABLE_VALIDATION = 'ENABLE_VALIDATION' 
    INPUT_TEST = 'INPUT_TEST'
    FIELD_TEST_DEPTH = 'FIELD_TEST_DEPTH'

    # Lists for Log Display & GUI
    MODEL_LIST_NAMES = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'Extra Trees', 'Ridge', 'Lasso', 'ElasticNet', 'KNN', 'Decision Tree', 'MLP', 'SVR']
    OPTIMIZER_LIST_NAMES = ['Random Search', 'Grid Search', 'Bayesian Search']
    COLLISION_LIST_NAMES = ['Keep All Points(weighted Average)', 'Highest Confidence', 'Closest to Pixel Center', 'Hybrid (Half-Pixel)', 'Strict Center Only']
    MASK_METHODS_NAMES = ['Otsu (Automatic)', 'Manual Threshold']

    def initAlgorithm(self, config=None):
        # 1. Input & Output
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_RASTER, 'Input Satellite Image'))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, 'Main Output Folder'))
        
        # 2. Band Selection
        self.addParameter(QgsProcessingParameterBand(self.COASTAL_BAND, 'Coastal Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=1))
        self.addParameter(QgsProcessingParameterBand(self.BLUE_BAND, 'Blue Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=2))
        self.addParameter(QgsProcessingParameterBand(self.GREEN_BAND, 'Green Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=3))
        self.addParameter(QgsProcessingParameterBand(self.RED_BAND, 'Red Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=4))
        self.addParameter(QgsProcessingParameterBand(self.NIR_BAND, 'NIR Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=8))

        # 3. Pre-processing Config
        self.addParameter(QgsProcessingParameterBoolean(self.APPLY_SUNGLINT, 'Apply Sunglint Correction', defaultValue=True))
        self.addParameter(QgsProcessingParameterBand(self.NIR_BAND_SUNGLINT, 'Sunglint NIR Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=8))
        # -- Advanced Sunglint --
        sg_perc = QgsProcessingParameterNumber(self.SUNGLINT_PERCENTILE, 'Sunglint Deep Water %', type=QgsProcessingParameterNumber.Double, defaultValue=1.0)
        sg_perc.setFlags(sg_perc.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(sg_perc)

        self.addParameter(QgsProcessingParameterVectorLayer(self.DEEP_WATER_POLY, 'Deep Water ROI (Optional)', optional=True))
        
        # -- Masking --
        self.addParameter(QgsProcessingParameterEnum(self.MASKING_METHOD, 'Water Masking Method', options=self.MASK_METHODS_NAMES, defaultValue=0))
        self.addParameter(QgsProcessingParameterNumber(self.MANUAL_THRESHOLD, 'Manual Threshold (if Manual selected)', type=QgsProcessingParameterNumber.Double, defaultValue=0.0, optional=True))
        
        # -- Advanced Masking --
        otsu_adj = QgsProcessingParameterNumber(self.OTSU_ADJUSTMENT, 'Otsu Threshold Adjustment (+/-)', type=QgsProcessingParameterNumber.Double, defaultValue=0.0)
        otsu_adj.setFlags(otsu_adj.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(otsu_adj)
        
        kern = QgsProcessingParameterNumber(self.MASK_KERNEL_SIZE, 'Mask Cleanup Kernel Size', type=QgsProcessingParameterNumber.Integer, defaultValue=3)
        kern.setFlags(kern.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(kern)
        
        # -- System --
        threads = QgsProcessingParameterNumber(self.NUM_THREADS, 'Processing Threads', type=QgsProcessingParameterNumber.Integer, defaultValue=4)
        threads.setFlags(threads.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(threads)

        # 4. Training Data
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_TRAIN, 'Main Training Points'))
        self.addParameter(QgsProcessingParameterField(self.FIELD_DEPTH, 'Depth Field', parentLayerParameterName=self.INPUT_TRAIN, type=QgsProcessingParameterField.Numeric))
        self.addParameter(QgsProcessingParameterField(self.FIELD_WEIGHT, 'Weight Field', parentLayerParameterName=self.INPUT_TRAIN, type=QgsProcessingParameterField.Numeric, optional=True))
        
        # 5. RANSAC
        self.addParameter(QgsProcessingParameterBoolean(self.ENABLE_RANSAC, 'Enable RANSAC Filtering', defaultValue=True))
        self.addParameter(QgsProcessingParameterNumber(self.RANSAC_THRESHOLD, 'RANSAC Residual Threshold (0=Auto)', type=QgsProcessingParameterNumber.Double, defaultValue=0.0))
        # -- Advanced RANSAC --
        ran_trials = QgsProcessingParameterNumber(self.RANSAC_MAX_TRIALS, 'RANSAC Max Trials', type=QgsProcessingParameterNumber.Integer, defaultValue=100)
        ran_trials.setFlags(ran_trials.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(ran_trials)

        # 6. Modeling Strategy
        self.addParameter(QgsProcessingParameterEnum(self.SELECTED_ALGOS, 'Algorithms', options=self.MODEL_LIST_NAMES, allowMultiple=True, defaultValue=[0, 1, 2, 3]))
        self.addParameter(QgsProcessingParameterEnum(self.OPTIMIZER_METHOD, 'Optimizer', options=self.OPTIMIZER_LIST_NAMES, defaultValue=0))
        self.addParameter(QgsProcessingParameterEnum(self.COLLISION_HANDLING, 'Collision Handling', options=self.COLLISION_LIST_NAMES, defaultValue=0))
        
        self.addParameter(QgsProcessingParameterNumber(self.N_ITERATIONS, 'Optimization Iterations', type=QgsProcessingParameterNumber.Integer, defaultValue=10))
        self.addParameter(QgsProcessingParameterNumber(self.MEDIAN_SIZE, 'Output Median Filter Size', type=QgsProcessingParameterNumber.Integer, defaultValue=3))
        
        # 7. Advanced Hyperparameters (Strings)
        rf_p = QgsProcessingParameterString(self.PARAM_RF, 'Random Forest Params', defaultValue="'n_estimators': [100, 300, 500], 'max_depth': [10, 30, 50], 'min_samples_split': [2, 5]")
        rf_p.setFlags(rf_p.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(rf_p)
        
        gb_p = QgsProcessingParameterString(self.PARAM_GB, 'Gradient Boosting Params', defaultValue="'n_estimators': [100, 300, 500], 'learning_rate': [0.01, 0.05, 0.1, 0.2], 'max_depth': [3, 5, 10]")
        gb_p.setFlags(gb_p.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(gb_p)
        
        et_p = QgsProcessingParameterString(self.PARAM_ET, 'Extra Trees Params', defaultValue="'n_estimators': [100, 300, 500], 'max_depth': [10, 30, 50], 'min_samples_split': [2, 5]")
        et_p.setFlags(et_p.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(et_p)
        
        svr_p = QgsProcessingParameterString(self.PARAM_SVR, 'SVR Params', defaultValue="'C': [1, 10, 100, 500], 'kernel': ['rbf', 'linear'], 'epsilon': [0.01, 0.1, 0.5]")
        svr_p.setFlags(svr_p.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(svr_p)
        
        mlp_p = QgsProcessingParameterString(self.PARAM_MLP, 'MLP Params', defaultValue="'hidden_layer_sizes': [(100,), (100, 50), (50, 50)], 'activation': ['relu', 'tanh'], 'learning_rate_init': [0.001, 0.01]")
        mlp_p.setFlags(mlp_p.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(mlp_p)

        # 8. Adaptive Phase
        self.addParameter(QgsProcessingParameterBoolean(self.ENABLE_ADAPTIVE, 'Enable Phase 4 (Adaptive Refinement)', defaultValue=True))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_ADAPTIVE_TRAIN, 'Adaptive Points (Optional)', optional=True))
        self.addParameter(QgsProcessingParameterField(self.FIELD_ADAPTIVE_DEPTH, 'Adaptive Depth Field', parentLayerParameterName=self.INPUT_ADAPTIVE_TRAIN, type=QgsProcessingParameterField.Numeric, optional=True))

        # 9. Validation
        self.addParameter(QgsProcessingParameterBoolean(self.ENABLE_VALIDATION, 'Enable Unseen Validation', defaultValue=True))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_TEST, 'Validation Points (Optional)', optional=True))
        self.addParameter(QgsProcessingParameterField(self.FIELD_TEST_DEPTH, 'Validation Field', parentLayerParameterName=self.INPUT_TEST, type=QgsProcessingParameterField.Numeric, optional=True))

    def name(self): return 'sdb_master_orchestrator'
    def displayName(self): return 'SDB Master Workflow (Full Pipeline)'
    def group(self): return ''
    def groupId(self): return ''
    def createInstance(self): return SDBMasterOrchestrator()

    def shortHelpString(self):
        help_text = """
        <div style="font-family: Arial, sans-serif; line-height: 1.2;">
            <h2 style="margin-bottom: 5px;">🛰️ <span style="color: #2E86C1;">Bathymetrix-AI</span>: Master SDB Workflow</h2>
            <p style="margin-top: 0; margin-bottom: 10px;">An advanced 5-phase pipeline for high-precision Satellite-Derived Bathymetry with Auto-ML.</p>

            <b style="display: block; margin-bottom: 2px;">🌊 Phase 01: Advanced Pre-processing</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li>Sun-glint correction <i>(Hedley et al., 2005)</i>.</li>
                <li>Adaptive Water Masking (Otsu/Manual) <i>(Otsu, 1979)</i>.</li>
                <li>Multi-band Log-Ratio features <i>(Stumpf et al., 2003)</i>.</li>
            </ul>

            <b style="display: block; margin-bottom: 2px;">🎯 Phase 02: Robust Filtering</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li>Noise removal from training data using <b>RANSAC</b> <i>(Fischler & Bolles, 1981)</i>.</li>
            </ul>

            <b style="display: block; margin-bottom: 2px;">🤖 Phase 03: Global Auto-ML</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li>Benchmarks 11 algorithms (RF, GBM, MLP, SVR, etc.).</li>
                <li>Optimization via <b>Random Search</b>, Grid Search, or Bayesian <i>(Bergstra & Bengio, 2012)</i>.</li>
                <li>Fully <b>Customizable Hyperparameters</b> for fine-tuning.</li>
            </ul>

            <b style="display: block; margin-bottom: 2px;">📍 Phase 04: Adaptive Refinement</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li>Spatially localized corrections & <b>Residual Analysis</b> <i>(Alevizos, 2020)</i>.</li>
            </ul>

            <b style="display: block; margin-bottom: 2px;">📉 Phase 05: Validation & Reporting</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li>Independent accuracy assessment on unseen test data.</li>
            </ul>

            <p style="margin-top: 10px; border-top: 1px solid #ccc; padding-top: 5px;">
                <b>Developer:</b> Mohamed Aly Nasef
            </p>
        </div>
        """
        return help_text

    def append_log(self, msg, log_path, feedback):
        feedback.pushInfo(msg)
        if log_path:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(msg + "\n")

    def reproject_layer_if_needed(self, vector_layer, target_crs, temp_output_path, context, feedback):
        if not vector_layer: return None
        if vector_layer.crs() == target_crs: return vector_layer.source()
        feedback.pushWarning(f"Reprojecting '{vector_layer.name()}'...")
        result = processing.run("native:reprojectlayer", {'INPUT': vector_layer, 'TARGET_CRS': target_crs, 'OUTPUT': temp_output_path}, context=context, feedback=feedback, is_child_algorithm=True)
        return result['OUTPUT']

    def processAlgorithm(self, parameters, context, feedback):
        out_dir = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        os.makedirs(out_dir, exist_ok=True)
        
        # --- INITIALIZE UNIFIED LOG ---
        log_path = os.path.join(out_dir, "SDB_Full_Log.txt")
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("===================================================\n")
            f.write(f"   SDB MASTER WORKFLOW LOG - {datetime.datetime.now()}\n")
            f.write("===================================================\n\n")
            
            f.write("--- [1. IMAGE SETTINGS] ---\n")
            f.write(f"Raster: {self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context).name()}\n")
            f.write(f"Bands: C={self.parameterAsInt(parameters, self.COASTAL_BAND, context)}, B={self.parameterAsInt(parameters, self.BLUE_BAND, context)}, G={self.parameterAsInt(parameters, self.GREEN_BAND, context)}, R={self.parameterAsInt(parameters, self.RED_BAND, context)}, NIR={self.parameterAsInt(parameters, self.NIR_BAND, context)}\n")
            f.write(f"Sunglint: {self.parameterAsBool(parameters, self.APPLY_SUNGLINT, context)} (DeepWater %: {self.parameterAsDouble(parameters, self.SUNGLINT_PERCENTILE, context)})\n")
            
            mask_method_idx = self.parameterAsInt(parameters, self.MASKING_METHOD, context)
            f.write(f"Masking: {self.MASK_METHODS_NAMES[mask_method_idx]} (Otsu Adj: {self.parameterAsDouble(parameters, self.OTSU_ADJUSTMENT, context)})\n")
            f.write(f"Threads: {self.parameterAsInt(parameters, self.NUM_THREADS, context)}\n\n")
            
            f.write("--- [2. MODELING STRATEGY] ---\n")
            f.write(f"RANSAC: {self.parameterAsBool(parameters, self.ENABLE_RANSAC, context)} (Thresh: {self.parameterAsDouble(parameters, self.RANSAC_THRESHOLD, context)})\n")
            opt_idx = self.parameterAsInt(parameters, self.OPTIMIZER_METHOD, context)
            col_idx = self.parameterAsInt(parameters, self.COLLISION_HANDLING, context)
            sel_algos = self.parameterAsEnums(parameters, self.SELECTED_ALGOS, context)
            f.write(f"Optimizer: {self.OPTIMIZER_LIST_NAMES[opt_idx]} (Iter: {self.parameterAsInt(parameters, self.N_ITERATIONS, context)})\n")
            f.write(f"Collision: {self.COLLISION_LIST_NAMES[col_idx]}\n")
            f.write(f"Algos: {[self.MODEL_LIST_NAMES[i] for i in sel_algos]}\n")
            f.write("===================================================\n\n")

        self.append_log(">>> Workflow Started...", log_path, feedback)

        input_raster = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        target_crs = input_raster.crs()
        
        temp_train = os.path.join(out_dir, 'temp_reprojected_train.gpkg')
        temp_test = os.path.join(out_dir, 'temp_reprojected_test.gpkg')
        temp_adapt = os.path.join(out_dir, 'temp_reprojected_adaptive.gpkg')

        final_train = self.reproject_layer_if_needed(self.parameterAsVectorLayer(parameters, self.INPUT_TRAIN, context), target_crs, temp_train, context, feedback)
        
        enable_val = self.parameterAsBool(parameters, self.ENABLE_VALIDATION, context)
        final_test = None
        if enable_val:
            t_layer = self.parameterAsVectorLayer(parameters, self.INPUT_TEST, context)
            if t_layer: final_test = self.reproject_layer_if_needed(t_layer, target_crs, temp_test, context, feedback)
            else: enable_val = False

        # Phase 1
        self.append_log("\n>>> Phase 1: Pre-processing...", log_path, feedback)
        p1 = processing.run("sdb_tools:sdb_phase1_preprocessing", {
            'INPUT_RASTER': input_raster,
            'COASTAL_BAND': self.parameterAsInt(parameters, self.COASTAL_BAND, context),
            'BLUE_BAND': self.parameterAsInt(parameters, self.BLUE_BAND, context),
            'GREEN_BAND': self.parameterAsInt(parameters, self.GREEN_BAND, context),
            'RED_BAND': self.parameterAsInt(parameters, self.RED_BAND, context),
            'NIR_BAND': self.parameterAsInt(parameters, self.NIR_BAND, context),
            'APPLY_SUNGLINT': self.parameterAsBool(parameters, self.APPLY_SUNGLINT, context),
            'NIR_BAND_SUNGLINT': self.parameterAsInt(parameters, self.NIR_BAND_SUNGLINT, context),
            'SUNGLINT_PERCENTILE': self.parameterAsDouble(parameters, self.SUNGLINT_PERCENTILE, context), # NEW
            'DEEP_WATER_POLY': self.parameterAsVectorLayer(parameters, self.DEEP_WATER_POLY, context),
            'MASKING_METHOD': self.parameterAsInt(parameters, self.MASKING_METHOD, context), # NEW
            'MANUAL_THRESHOLD': self.parameterAsDouble(parameters, self.MANUAL_THRESHOLD, context), # NEW
            'OTSU_ADJUSTMENT': self.parameterAsDouble(parameters, self.OTSU_ADJUSTMENT, context), # NEW
            'MASK_KERNEL_SIZE': self.parameterAsInt(parameters, self.MASK_KERNEL_SIZE, context), # NEW
            'NUM_THREADS': self.parameterAsInt(parameters, self.NUM_THREADS, context), # NEW
            'OUTPUT_FOLDER': out_dir
        }, context=context, feedback=feedback, is_child_algorithm=True)
        
        # Phase 2
        path_clean = final_train
        if self.parameterAsBool(parameters, self.ENABLE_RANSAC, context):
            self.append_log("\n>>> Phase 2: RANSAC...", log_path, feedback)
            p2 = processing.run("sdb_tools:sdb_02_filtering", {
                'INPUT_STACK': p1['OUTPUT_FEATURES'], 'INPUT_POINTS': final_train,
                'FIELD_DEPTH': self.parameterAsString(parameters, self.FIELD_DEPTH, context),
                'BLUE_BAND': self.parameterAsInt(parameters, self.BLUE_BAND, context),
                'GREEN_BAND': self.parameterAsInt(parameters, self.GREEN_BAND, context),
                'RESIDUAL_THRESHOLD': self.parameterAsDouble(parameters, self.RANSAC_THRESHOLD, context),
                'OUTPUT_FOLDER': out_dir
            }, context=context, feedback=feedback, is_child_algorithm=True)
            path_clean = p2['OUTPUT_CLEAN_VEC']

        # Phase 3
        self.append_log("\n>>> Phase 3: Global Modeling...", log_path, feedback)
        p3_params = {
            'INPUT_STACK': p1['OUTPUT_FEATURES'], 'INPUT_MASK': p1['OUTPUT_MASK'],
            'INPUT_POINTS': path_clean,
            'FIELD_DEPTH': self.parameterAsString(parameters, self.FIELD_DEPTH, context),
            'FIELD_WEIGHT': self.parameterAsString(parameters, self.FIELD_WEIGHT, context),
            'SELECTED_ALGOS': self.parameterAsEnums(parameters, self.SELECTED_ALGOS, context),
            'OPTIMIZER_METHOD': self.parameterAsInt(parameters, self.OPTIMIZER_METHOD, context),
            'COLLISION_HANDLING': self.parameterAsInt(parameters, self.COLLISION_HANDLING, context),
            'N_ITERATIONS': self.parameterAsInt(parameters, self.N_ITERATIONS, context),
            'MEDIAN_SIZE': self.parameterAsInt(parameters, self.MEDIAN_SIZE, context),
            'OUTPUT_FOLDER': out_dir,
            'LOG_FILE': log_path,
            'PARAM_RF': self.parameterAsString(parameters, self.PARAM_RF, context),
            'PARAM_GB': self.parameterAsString(parameters, self.PARAM_GB, context),
            'PARAM_ET': self.parameterAsString(parameters, self.PARAM_ET, context),
            'PARAM_SVR': self.parameterAsString(parameters, self.PARAM_SVR, context),
            'PARAM_MLP': self.parameterAsString(parameters, self.PARAM_MLP, context)
        }
        p3 = processing.run("sdb_tools:sdb_03_initial_modeling", p3_params, context=context, feedback=feedback, is_child_algorithm=True)
        
        # --- LOG METRICS FOR MODULE 03 ---
        if 'BEST_R2' in p3:
            msg_p3 = f"   [Phase 3 RESULT] R2: {p3['BEST_R2']:.4f} | RMSE: {p3['BEST_RMSE']:.4f} | wMAPE: {p3['BEST_WMAPE']:.2f}%"
            self.append_log(msg_p3, log_path, feedback)

        # Phase 4
        path_refined = p3['OUTPUT_DEPTH_MAP']
        if self.parameterAsBool(parameters, self.ENABLE_ADAPTIVE, context):
            self.append_log("\n>>> Phase 4: Adaptive Refinement...", log_path, feedback)
            ad_layer = self.parameterAsVectorLayer(parameters, self.INPUT_ADAPTIVE_TRAIN, context)
            final_ad = self.reproject_layer_if_needed(ad_layer, target_crs, temp_adapt, context, feedback)
            p4_params = {
                'INPUT_GLOBAL_RASTER': p3['OUTPUT_DEPTH_MAP'], 'INPUT_ORIGINAL_FEAT': p1['OUTPUT_FEATURES'],
                'INPUT_MASK': p1['OUTPUT_MASK'], 'INPUT_TRAIN': final_ad,
                'FIELD_TRAIN': self.parameterAsString(parameters, self.FIELD_ADAPTIVE_DEPTH, context),
                'SELECTED_ALGOS': self.parameterAsEnums(parameters, self.SELECTED_ALGOS, context),
                'OPTIMIZER_METHOD': self.parameterAsInt(parameters, self.OPTIMIZER_METHOD, context),
                'COLLISION_HANDLING': self.parameterAsInt(parameters, self.COLLISION_HANDLING, context),
                'N_ITERATIONS': self.parameterAsInt(parameters, self.N_ITERATIONS, context),
                'MEDIAN_SIZE': self.parameterAsInt(parameters, self.MEDIAN_SIZE, context),
                'OUTPUT_FOLDER': out_dir,
                'LOG_FILE': log_path,
                'PARAM_RF': self.parameterAsString(parameters, self.PARAM_RF, context),
                'PARAM_GB': self.parameterAsString(parameters, self.PARAM_GB, context),
                'PARAM_ET': self.parameterAsString(parameters, self.PARAM_ET, context),
                'PARAM_SVR': self.parameterAsString(parameters, self.PARAM_SVR, context),
                'PARAM_MLP': self.parameterAsString(parameters, self.PARAM_MLP, context)
            }
            p4 = processing.run("sdb_tools:sdb_phase4_adaptive", p4_params, context=context, feedback=feedback, is_child_algorithm=True)
            path_refined = p4['OUTPUT_FINAL']
            
            # --- LOG METRICS FOR MODULE 04 ---
            if 'BEST_R2' in p4:
                msg_p4 = f"   [Phase 4 RESULT] R2: {p4['BEST_R2']:.4f} | RMSE: {p4['BEST_RMSE']:.4f} | wMAPE: {p4['BEST_WMAPE']:.2f}%"
                self.append_log(msg_p4, log_path, feedback)
            
            if os.path.exists(path_refined):
                self.append_log(f"   [+] Phase 4 Map Generated: {path_refined}", log_path, feedback)
                QgsProject.instance().addMapLayer(QgsRasterLayer(path_refined, "4-Refined Model"))

        # Phase 5
        if enable_val and final_test:
            self.append_log("\n>>> Phase 5: Validation...", log_path, feedback)
            processing.run("sdb_tools:sdb_05_reporting", {
                'INPUT_MAP_P3': p3['OUTPUT_DEPTH_MAP'], 'INPUT_MAP_P4': path_refined,
                'INPUT_TRAIN': path_clean,
                'FIELD_TRAIN': self.parameterAsString(parameters, self.FIELD_DEPTH, context),
                'INPUT_VALIDATION': final_test,
                'FIELD_VAL_DEPTH': self.parameterAsString(parameters, self.FIELD_TEST_DEPTH, context),
                'OUTPUT_FOLDER': out_dir
            }, context=context, feedback=feedback, is_child_algorithm=True)

        return {'FINAL_DEPTH_MAP': path_refined}