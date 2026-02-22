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
    QgsProcessingParameterEnum, QgsProcessingParameterString, QgsRasterLayer
)
import processing
warnings.filterwarnings("ignore")

class SDBMasterOrchestrator(QgsProcessingAlgorithm):
    # ... (IDs كما هي في الكود السابق)
    INPUT_RASTER = 'INPUT_RASTER'; OUTPUT_FOLDER = 'OUTPUT_FOLDER'
    COASTAL_BAND = 'COASTAL_BAND'; BLUE_BAND = 'BLUE_BAND'; GREEN_BAND = 'GREEN_BAND'
    RED_BAND = 'RED_BAND'; NIR_BAND = 'NIR_BAND'
    APPLY_SUNGLINT = 'APPLY_SUNGLINT'; NIR_BAND_SUNGLINT = 'NIR_BAND_SUNGLINT'
    SUNGLINT_PERCENTILE = 'SUNGLINT_PERCENTILE'; DEEP_WATER_POLY = 'DEEP_WATER_POLY'
    
    ENABLE_MASKING = 'ENABLE_MASKING'
    MASKING_METHOD = 'MASKING_METHOD'; MANUAL_THRESHOLD = 'MANUAL_THRESHOLD'
    OTSU_ADJUSTMENT = 'OTSU_ADJUSTMENT'; MASK_KERNEL_SIZE = 'MASK_KERNEL_SIZE'
    FEATURE_SELECTION = 'FEATURE_SELECTION'
    ENABLE_BAND_CALC = 'ENABLE_BAND_CALC'
    BAND_MATH_FORMULA = 'BAND_MATH_FORMULA'
    
    NUM_THREADS = 'NUM_THREADS'
    
    ENABLE_RANSAC = 'ENABLE_RANSAC'; FILTER_MODE = 'FILTER_MODE'
    RANSAC_THRESHOLD = 'RANSAC_THRESHOLD'; RANSAC_MAX_TRIALS = 'RANSAC_MAX_TRIALS'
    
    INPUT_TRAIN = 'INPUT_TRAIN'; FIELD_DEPTH = 'FIELD_DEPTH'; FIELD_WEIGHT = 'FIELD_WEIGHT'
    SELECTED_ALGOS = 'SELECTED_ALGOS'; OPTIMIZER_METHOD = 'OPTIMIZER_METHOD' 
    COLLISION_HANDLING = 'COLLISION_HANDLING'; N_ITERATIONS = 'N_ITERATIONS'
    
    MEDIAN_SIZE = 'MEDIAN_SIZE'; PARAM_RF = 'PARAM_RF'; PARAM_GB = 'PARAM_GB'
    PARAM_ET = 'PARAM_ET'; PARAM_SVR = 'PARAM_SVR'; PARAM_MLP = 'PARAM_MLP'
    ENABLE_ADAPTIVE = 'ENABLE_ADAPTIVE'; INPUT_ADAPTIVE_TRAIN = 'INPUT_ADAPTIVE_TRAIN'
    FIELD_ADAPTIVE_DEPTH = 'FIELD_ADAPTIVE_DEPTH'; ENABLE_VALIDATION = 'ENABLE_VALIDATION' 
    INPUT_TEST = 'INPUT_TEST'; FIELD_TEST_DEPTH = 'FIELD_TEST_DEPTH'

    # Lists
    FILTER_MODES_NAMES = ['Linear RANSAC', 'LS Variance Fit', 'Huber Variance Fit']
    MODEL_LIST_NAMES = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'Extra Trees', 'Ridge', 'Lasso', 'ElasticNet', 'KNN', 'Decision Tree', 'MLP', 'SVR']
    OPTIMIZER_LIST_NAMES = ['Random Search', 'Grid Search', 'Bayesian Search']
    COLLISION_LIST_NAMES = ['Keep All Points', 'Highest Confidence', 'Closest to Pixel Center', 'Hybrid', 'Strict Center']
    MASK_METHODS_NAMES = ['Otsu (Automatic)', 'Manual Threshold']
    FEATURE_OPTIONS_NAMES = [
        '[All Raw] All Bands from Input Image', '[Log] Log(Coastal)', '[Log] Log(Blue)', '[Log] Log(Green)',
        '[Log] Log(Red)', '[Log] Log(NIR)', '[Ratio] Log(Blue) / Log(Green)', '[Ratio] Log(Blue) / Log(Red)',
        '[Ratio] Log(Coastal) / Log(Green)', '[Custom] Band Math Calculator'
    ]

    def initAlgorithm(self, config=None):
        # Phase 1
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_RASTER, 'Input Satellite Image'))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, 'Main Output Folder'))
        self.addParameter(QgsProcessingParameterBand(self.COASTAL_BAND, 'Coastal Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=1))
        self.addParameter(QgsProcessingParameterBand(self.BLUE_BAND, 'Blue Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=2))
        self.addParameter(QgsProcessingParameterBand(self.GREEN_BAND, 'Green Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=3))
        self.addParameter(QgsProcessingParameterBand(self.RED_BAND, 'Red Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=4))
        self.addParameter(QgsProcessingParameterBand(self.NIR_BAND, 'NIR Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=8))
        self.addParameter(QgsProcessingParameterBoolean(self.APPLY_SUNGLINT, 'Apply Sunglint Correction', defaultValue=True))
        self.addParameter(QgsProcessingParameterBand(self.NIR_BAND_SUNGLINT, 'Sunglint NIR Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=8))
        self.addParameter(QgsProcessingParameterNumber(self.SUNGLINT_PERCENTILE, 'Sunglint Deep Water %', type=QgsProcessingParameterNumber.Double, defaultValue=1.0))
        self.addParameter(QgsProcessingParameterVectorLayer(self.DEEP_WATER_POLY, 'Deep Water ROI (Optional)', optional=True))
        self.addParameter(QgsProcessingParameterBoolean(self.ENABLE_MASKING, 'Enable Water Masking', defaultValue=True))
        self.addParameter(QgsProcessingParameterEnum(self.MASKING_METHOD, 'Water Masking Method', options=self.MASK_METHODS_NAMES, defaultValue=0))
        self.addParameter(QgsProcessingParameterNumber(self.MANUAL_THRESHOLD, 'Manual Threshold', type=QgsProcessingParameterNumber.Double, defaultValue=0.0, optional=True))
        self.addParameter(QgsProcessingParameterNumber(self.OTSU_ADJUSTMENT, 'Otsu Threshold Adjustment', type=QgsProcessingParameterNumber.Double, defaultValue=0.0))
        self.addParameter(QgsProcessingParameterNumber(self.MASK_KERNEL_SIZE, 'Mask Cleanup Kernel Size', type=QgsProcessingParameterNumber.Integer, defaultValue=3))
        default_feats = list(range(len(self.FEATURE_OPTIONS_NAMES)))
        self.addParameter(QgsProcessingParameterEnum(self.FEATURE_SELECTION, 'Output Feature Stack', options=self.FEATURE_OPTIONS_NAMES, allowMultiple=True, defaultValue=default_feats))
        self.addParameter(QgsProcessingParameterBoolean(self.ENABLE_BAND_CALC, 'Enable Custom Band Math', defaultValue=False))
        self.addParameter(QgsProcessingParameterString(self.BAND_MATH_FORMULA, 'Band Math Formula', defaultValue="", optional=True))
        self.addParameter(QgsProcessingParameterNumber(self.NUM_THREADS, 'Processing Threads', type=QgsProcessingParameterNumber.Integer, defaultValue=4))

        # Phase 2 & 3 (Training Data Inputs - REORDERED)
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_TRAIN, 'Main Training Points'))
        self.addParameter(QgsProcessingParameterField(self.FIELD_DEPTH, 'Depth Field', parentLayerParameterName=self.INPUT_TRAIN))
        # --- MOVED WEIGHT FIELD HERE ---
        self.addParameter(QgsProcessingParameterField(self.FIELD_WEIGHT, 'Weight Field (Optional)', parentLayerParameterName=self.INPUT_TRAIN, optional=True))
        
        # Phase 2 Filters
        self.addParameter(QgsProcessingParameterBoolean(self.ENABLE_RANSAC, 'Enable Data Filtering', defaultValue=True))
        self.addParameter(QgsProcessingParameterEnum(self.FILTER_MODE, 'Filtering Strategy', options=self.FILTER_MODES_NAMES, defaultValue=2)) 
        self.addParameter(QgsProcessingParameterNumber(self.RANSAC_THRESHOLD, 'Threshold / Sigma Multiplier', type=QgsProcessingParameterNumber.Double, defaultValue=3.0))
        self.addParameter(QgsProcessingParameterNumber(self.RANSAC_MAX_TRIALS, 'RANSAC Trials', type=QgsProcessingParameterNumber.Integer, defaultValue=100))
        
        # Phase 3 Modeling Params
        self.addParameter(QgsProcessingParameterEnum(self.SELECTED_ALGOS, 'Algorithms', options=self.MODEL_LIST_NAMES, allowMultiple=True, defaultValue=[0, 1, 2, 3]))
        self.addParameter(QgsProcessingParameterEnum(self.OPTIMIZER_METHOD, 'Optimizer', options=self.OPTIMIZER_LIST_NAMES, defaultValue=0))
        self.addParameter(QgsProcessingParameterEnum(self.COLLISION_HANDLING, 'Collision Handling', options=self.COLLISION_LIST_NAMES, defaultValue=0))
        self.addParameter(QgsProcessingParameterNumber(self.N_ITERATIONS, 'Optimization Iterations', type=QgsProcessingParameterNumber.Integer, defaultValue=10))
        self.addParameter(QgsProcessingParameterNumber(self.MEDIAN_SIZE, 'Output Median Filter Size', type=QgsProcessingParameterNumber.Integer, defaultValue=3))
        self.addParameter(QgsProcessingParameterString(self.PARAM_RF, 'Random Forest Params', defaultValue="'n_estimators': [100, 300], 'max_depth': [10, 30]"))
        self.addParameter(QgsProcessingParameterString(self.PARAM_GB, 'Gradient Boosting Params', defaultValue="'n_estimators': [100, 300], 'learning_rate': [0.05, 0.1]"))
        self.addParameter(QgsProcessingParameterString(self.PARAM_ET, 'Extra Trees Params', defaultValue="'n_estimators': [100, 300], 'max_depth': [10, 30]"))
        self.addParameter(QgsProcessingParameterString(self.PARAM_SVR, 'SVR Params', defaultValue="'C': [1, 10, 100], 'kernel': ['rbf']"))
        self.addParameter(QgsProcessingParameterString(self.PARAM_MLP, 'MLP Params', defaultValue="'hidden_layer_sizes': [(100,), (50, 50)]"))
        
        # Phase 4 & 5
        self.addParameter(QgsProcessingParameterBoolean(self.ENABLE_ADAPTIVE, 'Enable Phase 4 (Adaptive)', defaultValue=True))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_ADAPTIVE_TRAIN, 'Adaptive Points (Optional)', optional=True))
        self.addParameter(QgsProcessingParameterField(self.FIELD_ADAPTIVE_DEPTH, 'Adaptive Depth Field', parentLayerParameterName=self.INPUT_ADAPTIVE_TRAIN, optional=True))
        self.addParameter(QgsProcessingParameterBoolean(self.ENABLE_VALIDATION, 'Enable Validation', defaultValue=True))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_TEST, 'Validation Points (Optional)', optional=True))
        self.addParameter(QgsProcessingParameterField(self.FIELD_TEST_DEPTH, 'Validation Field', parentLayerParameterName=self.INPUT_TEST, optional=True))

    def name(self): return 'sdb_master_orchestrator'
    def displayName(self): return 'SDB Master Workflow (Full Pipeline)'
    def group(self): return ''
    def groupId(self): return ''
    def createInstance(self): return SDBMasterOrchestrator()

    def shortHelpString(self):
        return """
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
                <li>Noise removal using <b>Linear RANSAC</b>, <b>LS Variance Fit</b>, or <b>Huber Variance Fit</b> <i>(Zhang et al., 2006)</i>.</li>
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

    def helpString(self): return self.shortHelpString()

    def append_log(self, msg, log_path, feedback):
        feedback.pushInfo(msg)
        if log_path:
            with open(log_path, 'a', encoding='utf-8') as f: f.write(msg + "\n")

    def reproject_layer_if_needed(self, vector_layer, target_crs, temp_output_path, context, feedback):
        if not vector_layer: return None
        if vector_layer.crs() == target_crs: return vector_layer.source()
        feedback.pushWarning(f"Reprojecting '{vector_layer.name()}'...")
        result = processing.run("native:reprojectlayer", {'INPUT': vector_layer, 'TARGET_CRS': target_crs, 'OUTPUT': temp_output_path}, context=context, feedback=feedback, is_child_algorithm=True)
        return result['OUTPUT']

    def processAlgorithm(self, parameters, context, feedback):
        out_dir = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "SDB_Full_Log.txt")
        with open(log_path, 'w', encoding='utf-8') as f: f.write(f"SDB LOG - {datetime.datetime.now()}\n\n")

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

        self.append_log("\n>>> Phase 1: Pre-processing...", log_path, feedback)
        p1 = processing.run("sdb_tools:sdb_phase1_preprocessing", {
            'INPUT_RASTER': input_raster,
            'COASTAL_BAND': parameters[self.COASTAL_BAND], 'BLUE_BAND': parameters[self.BLUE_BAND],
            'GREEN_BAND': parameters[self.GREEN_BAND], 'RED_BAND': parameters[self.RED_BAND], 'NIR_BAND': parameters[self.NIR_BAND],
            'APPLY_SUNGLINT': parameters[self.APPLY_SUNGLINT], 'NIR_BAND_SUNGLINT': parameters[self.NIR_BAND_SUNGLINT],
            'SUNGLINT_PERCENTILE': parameters[self.SUNGLINT_PERCENTILE], 'DEEP_WATER_POLY': parameters[self.DEEP_WATER_POLY],
            'ENABLE_MASKING': parameters[self.ENABLE_MASKING],
            'MASKING_METHOD': parameters[self.MASKING_METHOD], 'MANUAL_THRESHOLD': parameters[self.MANUAL_THRESHOLD],
            'OTSU_ADJUSTMENT': parameters[self.OTSU_ADJUSTMENT], 'MASK_KERNEL_SIZE': parameters[self.MASK_KERNEL_SIZE],
            'FEATURE_SELECTION': parameters[self.FEATURE_SELECTION],
            'ENABLE_BAND_CALC': parameters[self.ENABLE_BAND_CALC],
            'BAND_MATH_FORMULA': parameters[self.BAND_MATH_FORMULA],
            'NUM_THREADS': parameters[self.NUM_THREADS], 'OUTPUT_FOLDER': out_dir
        }, context=context, feedback=feedback, is_child_algorithm=True)
        
        path_clean = final_train
        if self.parameterAsBool(parameters, self.ENABLE_RANSAC, context):
            self.append_log("\n>>> Phase 2: Filtering & Uncertainty...", log_path, feedback)
            p2 = processing.run("sdb_tools:sdb_02_filtering", {
                'INPUT_STACK': p1['OUTPUT_FEATURES'], 'INPUT_POINTS': final_train,
                'FIELD_DEPTH': self.parameterAsString(parameters, self.FIELD_DEPTH, context),
                'BLUE_BAND': parameters[self.BLUE_BAND], 'GREEN_BAND': parameters[self.GREEN_BAND],
                'FILTER_MODE': parameters[self.FILTER_MODE],
                'RESIDUAL_THRESHOLD': parameters[self.RANSAC_THRESHOLD],
                'RANSAC_MAX_TRIALS': parameters[self.RANSAC_MAX_TRIALS],
                'OUTPUT_FOLDER': out_dir
            }, context=context, feedback=feedback, is_child_algorithm=True)
            path_clean = p2['OUTPUT_CLEAN_VEC']

        self.append_log("\n>>> Phase 3: Global Modeling...", log_path, feedback)
        p3_params = {
            'INPUT_STACK': p1['OUTPUT_FEATURES'], 'INPUT_MASK': p1['OUTPUT_MASK'],
            'INPUT_POINTS': path_clean,
            'FIELD_DEPTH': self.parameterAsString(parameters, self.FIELD_DEPTH, context),
            'FIELD_WEIGHT': self.parameterAsString(parameters, self.FIELD_WEIGHT, context),
            'SELECTED_ALGOS': parameters[self.SELECTED_ALGOS], 'OPTIMIZER_METHOD': parameters[self.OPTIMIZER_METHOD],
            'COLLISION_HANDLING': parameters[self.COLLISION_HANDLING], 'N_ITERATIONS': parameters[self.N_ITERATIONS],
            'MEDIAN_SIZE': parameters[self.MEDIAN_SIZE], 'OUTPUT_FOLDER': out_dir, 'LOG_FILE': log_path,
            'PARAM_RF': parameters[self.PARAM_RF], 'PARAM_GB': parameters[self.PARAM_GB],
            'PARAM_ET': parameters[self.PARAM_ET], 'PARAM_SVR': parameters[self.PARAM_SVR], 'PARAM_MLP': parameters[self.PARAM_MLP]
        }
        p3 = processing.run("sdb_tools:sdb_03_initial_modeling", p3_params, context=context, feedback=feedback, is_child_algorithm=True)
        if 'BEST_R2' in p3: self.append_log(f"   [Phase 3] R2: {p3['BEST_R2']:.4f}", log_path, feedback)

        path_refined = p3['OUTPUT_DEPTH_MAP']
        if self.parameterAsBool(parameters, self.ENABLE_ADAPTIVE, context):
            self.append_log("\n>>> Phase 4: Adaptive Refinement...", log_path, feedback)
            ad_layer = self.parameterAsVectorLayer(parameters, self.INPUT_ADAPTIVE_TRAIN, context)
            final_ad = self.reproject_layer_if_needed(ad_layer, target_crs, temp_adapt, context, feedback)
            
            p4_params = {
                'INPUT_GLOBAL_RASTER': p3['OUTPUT_DEPTH_MAP'], 'INPUT_ORIGINAL_FEAT': p1['OUTPUT_FEATURES'],
                'INPUT_UNCERTAINTY': None,
                'INPUT_MASK': p1['OUTPUT_MASK'], 'INPUT_TRAIN': final_ad,
                'FIELD_TRAIN': self.parameterAsString(parameters, self.FIELD_ADAPTIVE_DEPTH, context),
                'SELECTED_ALGOS': parameters[self.SELECTED_ALGOS], 'OPTIMIZER_METHOD': parameters[self.OPTIMIZER_METHOD],
                'COLLISION_HANDLING': parameters[self.COLLISION_HANDLING], 'N_ITERATIONS': parameters[self.N_ITERATIONS],
                'MEDIAN_SIZE': parameters[self.MEDIAN_SIZE], 'OUTPUT_FOLDER': out_dir, 'LOG_FILE': log_path,
                'PARAM_RF': parameters[self.PARAM_RF], 'PARAM_GB': parameters[self.PARAM_GB],
                'PARAM_ET': parameters[self.PARAM_ET], 'PARAM_SVR': parameters[self.PARAM_SVR], 'PARAM_MLP': parameters[self.PARAM_MLP]
            }
            p4 = processing.run("sdb_tools:sdb_phase4_adaptive", p4_params, context=context, feedback=feedback, is_child_algorithm=True)
            path_refined = p4['OUTPUT_FINAL']
            if 'BEST_R2' in p4: self.append_log(f"   [Phase 4] R2: {p4['BEST_R2']:.4f}", log_path, feedback)
            if path_refined and os.path.exists(path_refined): QgsProject.instance().addMapLayer(QgsRasterLayer(path_refined, "4-Refined Model"))

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