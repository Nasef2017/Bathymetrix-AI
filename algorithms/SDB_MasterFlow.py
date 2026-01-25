# SDB_Master_Orchestrator.py
# ---------------------------------------------------------------------------
# SDB MASTER ORCHESTRATOR (FULL LOGGING & AUTO-REPROJECT & OPTIONAL PHASES)
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

    # Phase 2 (Optional)
    ENABLE_RANSAC = 'ENABLE_RANSAC'
    RANSAC_THRESHOLD = 'RANSAC_THRESHOLD'

    # Phase 3 (Global Model)
    INPUT_TRAIN = 'INPUT_TRAIN'
    FIELD_DEPTH = 'FIELD_DEPTH'
    FIELD_WEIGHT = 'FIELD_WEIGHT'
    SELECTED_ALGOS = 'SELECTED_ALGOS'
    N_ITERATIONS = 'N_ITERATIONS'
    MEDIAN_SIZE = 'MEDIAN_SIZE'

    # Phase 4 (Adaptive - Optional & Separate Input)
    ENABLE_ADAPTIVE = 'ENABLE_ADAPTIVE'
    INPUT_ADAPTIVE_TRAIN = 'INPUT_ADAPTIVE_TRAIN'
    FIELD_ADAPTIVE_DEPTH = 'FIELD_ADAPTIVE_DEPTH'

    # Phase 5
    ENABLE_VALIDATION = 'ENABLE_VALIDATION' 
    INPUT_TEST = 'INPUT_TEST'
    FIELD_TEST_DEPTH = 'FIELD_TEST_DEPTH'

    # Algorithm List
    MODEL_LIST = [
        'Linear Regression', 'Random Forest', 'Gradient Boosting', 'Extra Trees',
        'Ridge', 'Lasso', 'ElasticNet', 'KNN', 'Decision Tree', 'MLP (Neural Net)', 'SVR'
    ]

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_RASTER, 'Input Satellite Image'))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, 'Main Output Folder'))
        
        self.addParameter(QgsProcessingParameterBand(self.COASTAL_BAND, 'Coastal Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=1))
        self.addParameter(QgsProcessingParameterBand(self.BLUE_BAND, 'Blue Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=2))
        self.addParameter(QgsProcessingParameterBand(self.GREEN_BAND, 'Green Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=3))
        self.addParameter(QgsProcessingParameterBand(self.RED_BAND, 'Red Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=4))
        self.addParameter(QgsProcessingParameterBand(self.NIR_BAND, 'NIR Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=8))

        self.addParameter(QgsProcessingParameterBoolean(self.APPLY_SUNGLINT, 'Apply Sunglint Correction', defaultValue=True))
        self.addParameter(QgsProcessingParameterBand(self.NIR_BAND_SUNGLINT, 'Sunglint NIR Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=8))
        self.addParameter(QgsProcessingParameterVectorLayer(self.DEEP_WATER_POLY, 'Deep Water ROI', optional=True))
        self.addParameter(QgsProcessingParameterBoolean(self.APPLY_WATER_MASK, 'Apply Water Mask (uses Otsu)', defaultValue=True))
        
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_TRAIN, 'Main Training Points (ICESat-2)'))
        self.addParameter(QgsProcessingParameterField(self.FIELD_DEPTH, 'Depth Field (Main)', parentLayerParameterName=self.INPUT_TRAIN, type=QgsProcessingParameterField.Numeric))
        self.addParameter(QgsProcessingParameterField(self.FIELD_WEIGHT, 'Confidence/Weight Field', parentLayerParameterName=self.INPUT_TRAIN, type=QgsProcessingParameterField.Numeric, optional=True))
        
        self.addParameter(QgsProcessingParameterBoolean(self.ENABLE_RANSAC, 'Enable RANSAC Filtering (Phase 2)', defaultValue=True))
        self.addParameter(QgsProcessingParameterNumber(self.RANSAC_THRESHOLD, 'RANSAC Outlier Threshold (Auto=0)', type=QgsProcessingParameterNumber.Double, defaultValue=0.0))

        self.addParameter(QgsProcessingParameterEnum(self.SELECTED_ALGOS, 'Select Algorithms (For Global Model)', options=self.MODEL_LIST, allowMultiple=True, defaultValue=[0, 1, 2, 3]))
        self.addParameter(QgsProcessingParameterNumber(self.N_ITERATIONS, 'Optimization Iterations', type=QgsProcessingParameterNumber.Integer, defaultValue=10, minValue=0))
        self.addParameter(QgsProcessingParameterNumber(self.MEDIAN_SIZE, 'Median Filter Size', type=QgsProcessingParameterNumber.Integer, defaultValue=3, minValue=1))
        
        self.addParameter(QgsProcessingParameterBoolean(self.ENABLE_ADAPTIVE, 'Enable Adaptive Re-training (Phase 4)', defaultValue=True))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_ADAPTIVE_TRAIN, 'Adaptive Correction Points', optional=True))
        self.addParameter(QgsProcessingParameterField(self.FIELD_ADAPTIVE_DEPTH, 'Depth Field (Adaptive)', parentLayerParameterName=self.INPUT_ADAPTIVE_TRAIN, type=QgsProcessingParameterField.Numeric, optional=True))

        self.addParameter(QgsProcessingParameterBoolean(self.ENABLE_VALIDATION, 'Enable Unseen Validation', defaultValue=True))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_TEST, 'Unseen Validation Points', optional=True))
        self.addParameter(QgsProcessingParameterField(self.FIELD_TEST_DEPTH, 'Validation Depth Field', parentLayerParameterName=self.INPUT_TEST, type=QgsProcessingParameterField.Numeric, optional=True))

    def name(self): return 'sdb_master_orchestrator'
    def displayName(self): return 'SDB Master Workflow (Full Pipeline)'
    
    def shortHelpString(self):
        help_text = """
        <div style="font-family: Arial, sans-serif; line-height: 1.2;">
            <h2 style="margin-bottom: 5px;">🛰️ <span style="color: #2E86C1;">Bathymetrix-AI</span>: Master SDB Workflow</h2>
            <p style="margin-top: 0; margin-bottom: 10px;">A comprehensive 4-phase pipeline for high-precision Satellite-Derived Bathymetry.</p>

            <b style="display: block; margin-bottom: 2px;">🌊 Phase 01: Pre-processing & Features</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li>Sun-glint correction <i>(Hedley et al., 2005)</i>.</li>
                <li>Automated Water Masking <i>(Otsu, 1979)</i>.</li>
                <li>Log-Ratio feature generation <i>(Stumpf et al., 2003)</i>.</li>
            </ul>

            <b style="display: block; margin-bottom: 2px;">🎯 Phase 02: Altimetry Filtering</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li>Outlier removal from ICESat-2 ATL24 data using <b>RANSAC</b> <i>(Fischler & Bolles, 1981)</i>.</li>
            </ul>

            <b style="display: block; margin-bottom: 2px;">🤖 Phase 03: Global Auto-ML</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li>Benchmarks 11 algorithms with <b>Randomized Tuning</b> <i>(Bergstra & Bengio, 2012)</i>.</li>
            </ul>

            <b style="display: block; margin-bottom: 2px;">📍 Phase 04: Spatial Refinement</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li>Corrects local biases using <b>Residual Analysis</b> <i>(Alevizos, 2020)</i>.</li>
            </ul>

            <p style="margin-top: 10px; border-top: 1px solid #ccc; padding-top: 5px;">
                <b>Developer:</b> Mohamed Aly Nasef
            </p>
        </div>
        """
        return help_text
        
    def createInstance(self): return SDBMasterOrchestrator()

    def reproject_layer_if_needed(self, vector_layer, target_crs, temp_output_path, context, feedback):
        if not vector_layer: return None
        if vector_layer.crs() == target_crs:
            return vector_layer.source()
        else:
            feedback.pushWarning(f"Reprojecting '{vector_layer.name()}'...")
            result = processing.run("native:reprojectlayer", {'INPUT': vector_layer, 'TARGET_CRS': target_crs, 'OUTPUT': temp_output_path}, context=context, feedback=feedback, is_child_algorithm=True)
            return result['OUTPUT']

    def processAlgorithm(self, parameters, context, feedback):
        total_start_time = time.time()
        start_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        out_dir = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        os.makedirs(out_dir, exist_ok=True)
        
        input_raster = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        target_crs = input_raster.crs()
        
        temp_train_path = os.path.join(out_dir, 'temp_reprojected_train.gpkg')
        temp_test_path = os.path.join(out_dir, 'temp_reprojected_test.gpkg')
        temp_adaptive_path = os.path.join(out_dir, 'temp_reprojected_adaptive.gpkg')

        final_train_path = self.reproject_layer_if_needed(self.parameterAsVectorLayer(parameters, self.INPUT_TRAIN, context), target_crs, temp_train_path, context, feedback)
        
        enable_validation = self.parameterAsBool(parameters, self.ENABLE_VALIDATION, context)
        final_test_path = None
        if enable_validation:
            test_layer = self.parameterAsVectorLayer(parameters, self.INPUT_TEST, context)
            if test_layer:
                final_test_path = self.reproject_layer_if_needed(test_layer, target_crs, temp_test_path, context, feedback)
            else:
                enable_validation = False
        
        # Step 1
        feedback.pushInfo(">>> Phase 1: Pre-processing...")
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
            'MASKING_METHOD': 0 if self.parameterAsBool(parameters, self.APPLY_WATER_MASK, context) else 2, 
            'OUTPUT_FOLDER': out_dir
        }
        p1 = processing.run("sdb_tools:sdb_phase1_preprocessing", p1_params, context=context, feedback=feedback, is_child_algorithm=True)
        
        # Step 2
        path_clean_points = final_train_path
        if self.parameterAsBool(parameters, self.ENABLE_RANSAC, context):
            feedback.pushInfo(">>> Phase 2: RANSAC Filtering...")
            p2_params = {
                'INPUT_STACK': p1['OUTPUT_FEATURES'], 'INPUT_POINTS': final_train_path,
                'FIELD_DEPTH': self.parameterAsString(parameters, self.FIELD_DEPTH, context),
                'BLUE_BAND': self.parameterAsInt(parameters, self.BLUE_BAND, context),
                'GREEN_BAND': self.parameterAsInt(parameters, self.GREEN_BAND, context),
                'RESIDUAL_THRESHOLD': self.parameterAsDouble(parameters, self.RANSAC_THRESHOLD, context),
                'OUTPUT_FOLDER': out_dir
            }
            p2 = processing.run("sdb_tools:sdb_02_filtering", p2_params, context=context, feedback=feedback, is_child_algorithm=True)
            path_clean_points = p2['OUTPUT_CLEAN_VEC']

        # Step 3
        feedback.pushInfo(">>> Phase 3: Global Modeling...")
        p3_params = {
            'INPUT_STACK': p1['OUTPUT_FEATURES'], 'INPUT_MASK': p1['OUTPUT_MASK'],
            'INPUT_POINTS': path_clean_points,
            'FIELD_DEPTH': self.parameterAsString(parameters, self.FIELD_DEPTH, context),
            'FIELD_WEIGHT': self.parameterAsString(parameters, self.FIELD_WEIGHT, context),
            'SELECTED_ALGOS': self.parameterAsEnums(parameters, self.SELECTED_ALGOS, context),
            'N_ITERATIONS': self.parameterAsInt(parameters, self.N_ITERATIONS, context),
            'MEDIAN_SIZE': self.parameterAsInt(parameters, self.MEDIAN_SIZE, context),
            'OUTPUT_FOLDER': out_dir
        }
        p3 = processing.run("sdb_tools:sdb_03_initial_modeling", p3_params, context=context, feedback=feedback, is_child_algorithm=True)

        # Step 4
        path_refined_depth = p3['OUTPUT_DEPTH_MAP']
        if self.parameterAsBool(parameters, self.ENABLE_ADAPTIVE, context):
            feedback.pushInfo(">>> Phase 4: Adaptive Refinement...")
            adaptive_layer = self.parameterAsVectorLayer(parameters, self.INPUT_ADAPTIVE_TRAIN, context)
            final_adaptive_path = self.reproject_layer_if_needed(adaptive_layer, target_crs, temp_adaptive_path, context, feedback)
            p4_params = {
                'INPUT_GLOBAL_RASTER': p3['OUTPUT_DEPTH_MAP'], 'INPUT_ORIGINAL_FEAT': p1['OUTPUT_FEATURES'],
                'INPUT_MASK': p1['OUTPUT_MASK'], 'INPUT_TRAIN': final_adaptive_path,
                'FIELD_TRAIN': self.parameterAsString(parameters, self.FIELD_ADAPTIVE_DEPTH, context),
                'SELECTED_ALGOS': self.parameterAsEnums(parameters, self.SELECTED_ALGOS, context),
                'N_ITERATIONS': self.parameterAsInt(parameters, self.N_ITERATIONS, context),
                'MEDIAN_SIZE': self.parameterAsInt(parameters, self.MEDIAN_SIZE, context),
                'OUTPUT_FOLDER': out_dir
            }
            p4 = processing.run("sdb_tools:sdb_phase4_adaptive", p4_params, context=context, feedback=feedback, is_child_algorithm=True)
            path_refined_depth = p4['OUTPUT_FINAL']

        # Step 5
        if enable_validation and final_test_path:
            p5_params = {
                'INPUT_MAP_P3': p3['OUTPUT_DEPTH_MAP'], 'INPUT_MAP_P4': path_refined_depth,
                'INPUT_TRAIN': path_clean_points,
                'FIELD_TRAIN': self.parameterAsString(parameters, self.FIELD_DEPTH, context),
                'INPUT_VALIDATION': final_test_path,
                'FIELD_VAL_DEPTH': self.parameterAsString(parameters, self.FIELD_TEST_DEPTH, context),
                'OUTPUT_FOLDER': out_dir
            }
            processing.run("sdb_tools:sdb_05_reporting", p5_params, context=context, feedback=feedback, is_child_algorithm=True)

        return {'FINAL_DEPTH_MAP': path_refined_depth}