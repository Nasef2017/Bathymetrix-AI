import os
import time
import datetime
import warnings
import numpy as np
import rasterio

from qgis.core import (
    QgsProcessing, QgsProcessingAlgorithm, QgsProcessingException,
    QgsProcessingParameterRasterLayer, QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField, QgsProcessingParameterNumber,
    QgsProcessingParameterBand, QgsProcessingParameterFolderDestination,
    QgsProcessingParameterBoolean, QgsProject, QgsProcessingContext,
    QgsProcessingParameterEnum, QgsProcessingParameterString, QgsRasterLayer
)
import processing
warnings.filterwarnings("ignore")

class SDBMasterOrchestrator(QgsProcessingAlgorithm):

    # =======================================================================
    # 1. PARAMETER CONSTANTS
    # =======================================================================
    
    # [0] General I/O
    INPUT_RASTER         = 'INPUT_RASTER'
    OUTPUT_FOLDER        = 'OUTPUT_FOLDER'
    NUM_THREADS          = 'NUM_THREADS'
    
    # [1] Pre-processing (Bands, Sunglint, Masking, Features)
    COASTAL_BAND         = 'COASTAL_BAND'
    BLUE_BAND            = 'BLUE_BAND'
    GREEN_BAND           = 'GREEN_BAND'
    RED_BAND             = 'RED_BAND'
    NIR_BAND             = 'NIR_BAND'
    
    APPLY_SUNGLINT       = 'APPLY_SUNGLINT'
    NIR_BAND_SUNGLINT    = 'NIR_BAND_SUNGLINT'
    SUNGLINT_PERCENTILE  = 'SUNGLINT_PERCENTILE'
    
    WATER_MASK_POLY      = 'WATER_MASK_POLY'
    SHRINK_EDGE_DIST     = 'SHRINK_EDGE_DIST'
    ENABLE_MASKING       = 'ENABLE_MASKING'
    MASKING_METHOD       = 'MASKING_METHOD'
    MANUAL_THRESHOLD     = 'MANUAL_THRESHOLD'
    OTSU_ADJUSTMENT      = 'OTSU_ADJUSTMENT'
    MASK_KERNEL_SIZE     = 'MASK_KERNEL_SIZE'
    
    FEATURE_SELECTION    = 'FEATURE_SELECTION'
    ENABLE_BAND_CALC     = 'ENABLE_BAND_CALC'
    BAND_MATH_FORMULA    = 'BAND_MATH_FORMULA'

    # [2] Filtering & Training Data
    INPUT_TRAIN          = 'INPUT_TRAIN'
    FIELD_DEPTH          = 'FIELD_DEPTH'
    FIELD_WEIGHT         = 'FIELD_WEIGHT'
    MAX_DEPTH_THRESHOLD  = 'MAX_DEPTH_THRESHOLD'
    
    ENABLE_RANSAC        = 'ENABLE_RANSAC'
    FILTER_MODE          = 'FILTER_MODE'
    RANSAC_THRESHOLD     = 'RANSAC_THRESHOLD'
    RANSAC_MAX_TRIALS    = 'RANSAC_MAX_TRIALS'

    # [3] Global Modeling (Auto-ML)
    SELECTED_ALGOS       = 'SELECTED_ALGOS'
    OPTIMIZER_METHOD     = 'OPTIMIZER_METHOD'
    COLLISION_HANDLING   = 'COLLISION_HANDLING'
    N_ITERATIONS         = 'N_ITERATIONS'
    MEDIAN_SIZE          = 'MEDIAN_SIZE'
    
    PARAM_RF             = 'PARAM_RF'
    PARAM_GB             = 'PARAM_GB'
    PARAM_ET             = 'PARAM_ET'
    PARAM_SVR            = 'PARAM_SVR'
    PARAM_MLP            = 'PARAM_MLP'

    # [4] Adaptive Refinement
    ENABLE_ADAPTIVE      = 'ENABLE_ADAPTIVE'
    INPUT_ADAPTIVE_TRAIN = 'INPUT_ADAPTIVE_TRAIN'
    FIELD_ADAPTIVE_DEPTH = 'FIELD_ADAPTIVE_DEPTH'
    
    # [5] Validation & Output Cleanup
    ENABLE_VALIDATION    = 'ENABLE_VALIDATION'
    INPUT_TEST           = 'INPUT_TEST'
    FIELD_TEST_DEPTH     = 'FIELD_TEST_DEPTH'
    
    ENABLE_SLOPE_FILTER  = 'ENABLE_SLOPE_FILTER'
    SLOPE_THRESHOLD      = 'SLOPE_THRESHOLD'
    REMOVE_POSITIVES     = 'REMOVE_POSITIVES'


    # =======================================================================
    # 2. OPTION LISTS (DROPDOWNS)
    # =======================================================================
    FILTER_MODES_NAMES   = ['Linear RANSAC', 'LS Variance Fit', 'Huber Variance Fit']
    MODEL_LIST_NAMES     = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'Extra Trees',
                            'Ridge', 'Lasso', 'ElasticNet', 'KNN', 'Decision Tree', 'MLP', 'SVR']
    OPTIMIZER_LIST_NAMES = ['Random Search', 'Grid Search', 'Bayesian Search']
    COLLISION_LIST_NAMES = ['Keep All Points', 'Highest Confidence', 'Closest to Pixel Center',
                            'Hybrid', 'Strict Center']
    MASK_METHODS_NAMES   = ['Otsu (Automatic NDWI)', 'Manual NDWI Threshold']
    FEATURE_OPTIONS_NAMES= [
        '[All Raw] All Bands from Input Image', '[Log] Log(Coastal)', '[Log] Log(Blue)',
        '[Log] Log(Green)', '[Log] Log(Red)', '[Log] Log(NIR)',
        '[Ratio] Log(Blue) / Log(Green)', '[Ratio] Log(Blue) / Log(Red)',
        '[Ratio] Log(Coastal) / Log(Green)', '[Custom] Band Math Calculator'
    ]


    # =======================================================================
    # 3. ALGORITHM METADATA & HELP STRINGS
    # =======================================================================
    def name(self):        
        return 'sdb_master_orchestrator'
        
    def displayName(self): 
        return 'SDB Master Workflow (Full Pipeline)'
        
    def group(self):       
        return ''
        
    def groupId(self):     
        return ''
        
    def createInstance(self): 
        return SDBMasterOrchestrator()
        
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
                <li>Noise removal using <b>Linear RANSAC</b>, <b>LS Variance Fit</b>, or <b>Huber Variance Fit</b> <i>(Zhang et al., 2021)</i>.</li>
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
        
    def helpString(self):
        return """<b>SDB Master Workflow (Full Pipeline)</b><br><br>
        This tool executes a complete Auto-ML pipeline for Satellite-Derived Bathymetry (SDB).<br><br>
        <b>Outputs Explained:</b><br>
        • <b>Initial SDB Map[Phase 3]:</b> The base depth map produced after the global machine learning modeling.<br>
        • <b>Refined SDB Map [Phase 4]:</b> The final, highly accurate depth map after applying adaptive localized corrections.<br><br>
        <i>* Note: All output files are automatically saved to your specified 'Main Output Folder' and loaded cleanly into the map canvas upon completion.</i>
        """


    # =======================================================================
    # 4. ALGORITHM INIT (QGIS FRONT-END UI SETUP)
    # =======================================================================
    def initAlgorithm(self, config=None):
        
        # -------------------------------------------------------------------
        # [0] General Settings
        # -------------------------------------------------------------------
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_RASTER, '📁 [0] Input Satellite Image'))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, '📁 [0] Main Output Folder'))
        self.addParameter(QgsProcessingParameterNumber(self.NUM_THREADS, '⚙️ [0] Processing Threads', type=QgsProcessingParameterNumber.Integer, defaultValue=4))

        # -------------------------------------------------------------------
        # [1] Phase 1: Pre-processing (Bands & Sunglint)
        # -------------------------------------------------------------------
        self.addParameter(QgsProcessingParameterBand(self.COASTAL_BAND, '📡 [1.1] Coastal Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=1))
        self.addParameter(QgsProcessingParameterBand(self.BLUE_BAND,    '📡 [1.1] Blue Band',    parentLayerParameterName=self.INPUT_RASTER, defaultValue=2))
        self.addParameter(QgsProcessingParameterBand(self.GREEN_BAND,   '📡 [1.1] Green Band',   parentLayerParameterName=self.INPUT_RASTER, defaultValue=3))
        self.addParameter(QgsProcessingParameterBand(self.RED_BAND,     '📡 [1.1] Red Band',     parentLayerParameterName=self.INPUT_RASTER, defaultValue=4))
        self.addParameter(QgsProcessingParameterBand(self.NIR_BAND,     '📡 [1.1] NIR Band',     parentLayerParameterName=self.INPUT_RASTER, defaultValue=8))
        
        self.addParameter(QgsProcessingParameterBoolean(self.APPLY_SUNGLINT, '☀️ [1.2] Apply Sunglint Correction', defaultValue=True))
        self.addParameter(QgsProcessingParameterBand(self.NIR_BAND_SUNGLINT, '☀️ [1.2] Sunglint NIR Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=8))
        self.addParameter(QgsProcessingParameterNumber(self.SUNGLINT_PERCENTILE, '☀️ [1.2] Sunglint Deep Water %', type=QgsProcessingParameterNumber.Double, defaultValue=1.0))

        # -------------------------------------------------------------------
        # [1] Phase 1: Masking & Features
        # -------------------------------------------------------------------
        self.addParameter(QgsProcessingParameterVectorLayer(self.WATER_MASK_POLY, '🗺️ [1.3] Ready-made Water Mask Polygon', types=[QgsProcessing.TypeVectorPolygon], optional=True))
        self.addParameter(QgsProcessingParameterNumber(self.SHRINK_EDGE_DIST, '🗺️ [1.3] Water Edge Shrink (Map Units, e.g. -10)', type=QgsProcessingParameterNumber.Double, defaultValue=0.0))
        self.addParameter(QgsProcessingParameterBoolean(self.ENABLE_MASKING, '🏖️ [1.3] Enable Automated Water Masking', defaultValue=False))
        self.addParameter(QgsProcessingParameterEnum(self.MASKING_METHOD, '🏖️ [1.3] Water Masking Method', options=self.MASK_METHODS_NAMES, defaultValue=0))
        self.addParameter(QgsProcessingParameterNumber(self.MANUAL_THRESHOLD, '🏖️ [1.3] Manual Threshold', type=QgsProcessingParameterNumber.Double, defaultValue=0.0, optional=True))
        self.addParameter(QgsProcessingParameterNumber(self.OTSU_ADJUSTMENT, '🏖️ [1.3] Otsu Threshold Adjustment', type=QgsProcessingParameterNumber.Double, defaultValue=0.0))
        self.addParameter(QgsProcessingParameterNumber(self.MASK_KERNEL_SIZE, '🏖️ [1.3] Mask Cleanup Kernel Size', type=QgsProcessingParameterNumber.Integer, defaultValue=3))

        default_feats = list(range(len(self.FEATURE_OPTIONS_NAMES)))
        self.addParameter(QgsProcessingParameterEnum(self.FEATURE_SELECTION, '📊 [1.4] Output Feature Stack', options=self.FEATURE_OPTIONS_NAMES, allowMultiple=True, defaultValue=default_feats))
        self.addParameter(QgsProcessingParameterBoolean(self.ENABLE_BAND_CALC, '🧮 [1.4] Enable Custom Band Math', defaultValue=False))
        self.addParameter(QgsProcessingParameterString(self.BAND_MATH_FORMULA, '🧮 [1.4] Band Math Formula', defaultValue="", optional=True))

        # -------------------------------------------------------------------
        # [2] Phase 2: Training Data & Filtering
        # -------------------------------------------------------------------
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_TRAIN, '📍 [2.1] Main Training Points'))
        self.addParameter(QgsProcessingParameterField(self.FIELD_DEPTH, '📏 [2.1] Depth Field', parentLayerParameterName=self.INPUT_TRAIN))
        self.addParameter(QgsProcessingParameterField(self.FIELD_WEIGHT, '⚖️ [2.1] Weight Field', parentLayerParameterName=self.INPUT_TRAIN, optional=True))
        self.addParameter(QgsProcessingParameterNumber(self.MAX_DEPTH_THRESHOLD, '🛑 [2.1] Maximum Depth Threshold (e.g. -30)', type=QgsProcessingParameterNumber.Double, defaultValue=-30.0))

        self.addParameter(QgsProcessingParameterBoolean(self.ENABLE_RANSAC, '🧹 [2.2] Enable Data Filtering (Noise Removal)', defaultValue=True))
        self.addParameter(QgsProcessingParameterEnum(self.FILTER_MODE, '🧹 [2.2] Filtering Strategy', options=self.FILTER_MODES_NAMES, defaultValue=2))
        self.addParameter(QgsProcessingParameterNumber(self.RANSAC_THRESHOLD, '🧹 [2.2] Threshold / Sigma Multiplier', type=QgsProcessingParameterNumber.Double, defaultValue=3.0))
        self.addParameter(QgsProcessingParameterNumber(self.RANSAC_MAX_TRIALS, '🧹 [2.2] RANSAC Trials', type=QgsProcessingParameterNumber.Integer, defaultValue=100))

        # -------------------------------------------------------------------
        # [3] Phase 3: Global Modeling (Auto-ML)
        # -------------------------------------------------------------------
        self.addParameter(QgsProcessingParameterEnum(self.SELECTED_ALGOS, '🤖 [3] Algorithms to Benchmark', options=self.MODEL_LIST_NAMES, allowMultiple=True, defaultValue=[0, 1, 2, 3]))
        self.addParameter(QgsProcessingParameterEnum(self.OPTIMIZER_METHOD, '🤖 [3] Optimizer Method', options=self.OPTIMIZER_LIST_NAMES, defaultValue=0))
        self.addParameter(QgsProcessingParameterEnum(self.COLLISION_HANDLING, '🤖 [3] Collision Handling', options=self.COLLISION_LIST_NAMES, defaultValue=0))
        self.addParameter(QgsProcessingParameterNumber(self.N_ITERATIONS, '🤖 [3] Optimization Iterations', type=QgsProcessingParameterNumber.Integer, defaultValue=10))
        self.addParameter(QgsProcessingParameterNumber(self.MEDIAN_SIZE, '🤖 [3] Output Median Filter Size', type=QgsProcessingParameterNumber.Integer, defaultValue=5))
        
        # Hyperparameters
        self.addParameter(QgsProcessingParameterString(self.PARAM_RF,  '🎛️ [3] Random Forest Params',      defaultValue="'n_estimators':[100, 300], 'max_depth':[10, 30]"))
        self.addParameter(QgsProcessingParameterString(self.PARAM_GB,  '🎛️ [3] Gradient Boosting Params',  defaultValue="'n_estimators':[100, 300], 'learning_rate':[0.05, 0.1]"))
        self.addParameter(QgsProcessingParameterString(self.PARAM_ET,  '🎛️ [3] Extra Trees Params',        defaultValue="'n_estimators':[100, 300], 'max_depth':[10, 30]"))
        self.addParameter(QgsProcessingParameterString(self.PARAM_SVR, '🎛️ [3] SVR Params',                defaultValue="'C':[1, 10, 100], 'kernel':['rbf']"))
        self.addParameter(QgsProcessingParameterString(self.PARAM_MLP, '🎛️ [3] MLP Params',                defaultValue="'hidden_layer_sizes':[(100,), (50, 50)]"))

        # -------------------------------------------------------------------
        # [4] Phase 4: Adaptive Refinement
        # -------------------------------------------------------------------
        self.addParameter(QgsProcessingParameterBoolean(self.ENABLE_ADAPTIVE, '🎯 [4] Enable Adaptive Refinement', defaultValue=True))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_ADAPTIVE_TRAIN, '🎯 [4] Adaptive Points', optional=True))
        self.addParameter(QgsProcessingParameterField(self.FIELD_ADAPTIVE_DEPTH, '🎯 [4] Adaptive Depth Field', parentLayerParameterName=self.INPUT_ADAPTIVE_TRAIN, optional=True))
        
        # -------------------------------------------------------------------
        # [5] Phase 5: Validation & Output Cleanup
        # -------------------------------------------------------------------
        self.addParameter(QgsProcessingParameterBoolean(self.ENABLE_VALIDATION, '📉 [5] Enable Validation', defaultValue=True))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_TEST, '📉 [5] Validation Points', optional=True))
        self.addParameter(QgsProcessingParameterField(self.FIELD_TEST_DEPTH, '📉 [5] Validation Depth Field', parentLayerParameterName=self.INPUT_TEST, optional=True))
        
        self.addParameter(QgsProcessingParameterBoolean(self.REMOVE_POSITIVES, '🧽 [Cleanup] Remove Positive Depths (>= 0)', defaultValue=True))
        self.addParameter(QgsProcessingParameterBoolean(self.ENABLE_SLOPE_FILTER, '🧽 [Cleanup] Apply Slope Filter (Remove sharp jumps)', defaultValue=True))
        self.addParameter(QgsProcessingParameterNumber(self.SLOPE_THRESHOLD, '🧽 [Cleanup] Slope Filter Threshold (Degrees)', type=QgsProcessingParameterNumber.Double, defaultValue=35.0))


    # =======================================================================
    # 5. HELPER METHODS
    # =======================================================================
    
    def append_log(self, msg, log_path, feedback):
        """Writes messages to the processing log and text file."""
        feedback.pushInfo(msg)
        if log_path:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(msg + "\n")

    def reproject_layer_if_needed(self, vector_layer, target_crs, temp_output_path, context, feedback):
        """Reprojects vector layer to match raster CRS if necessary."""
        if not vector_layer:
            return None
        if vector_layer.crs() == target_crs:
            return vector_layer.source()
        
        feedback.pushWarning(f"Reprojecting '{vector_layer.name()}'...")
        result = processing.run(
            "native:reprojectlayer",
            {'INPUT': vector_layer, 'TARGET_CRS': target_crs, 'OUTPUT': temp_output_path},
            context=context, feedback=feedback, is_child_algorithm=True
        )
        return result['OUTPUT']

    def filter_by_depth(self, layer_source, depth_field, max_depth, context, feedback):
        """Filters input points by the maximum allowable depth."""
        if not layer_source or not depth_field:
            return layer_source
        
        expr = f'"{depth_field}" >= {max_depth}' if max_depth < 0 else f'"{depth_field}" <= {max_depth}'
        feedback.pushInfo(f"Filtering points... keeping: {expr}")
        
        result = processing.run(
            "native:extractbyexpression",
            {'INPUT': layer_source, 'EXPRESSION': expr, 'OUTPUT': 'memory:'},
            context=context, feedback=feedback, is_child_algorithm=True
        )
        return result['OUTPUT']

    def clean_depth_map(self, depth_raster, feature_stack_raster, max_depth, out_path, context, feedback):
        """Clamps edges and handles deep extrapolation anomalies."""
        feedback.pushInfo(">>> Cleaning Depth Map: Clamping edges and deep anomalies...")
        deep_limit = (max_depth * 1.5) if max_depth < 0 else -100.0
        formula = (
            f'A * ((A >= {deep_limit}) * (B > -9000)) '
            f'+ (-9999.0) * (1 - ((A >= {deep_limit}) * (B > -9000)))'
        )
        calc_res = processing.run(
            "gdal:rastercalculator",
            {
                'INPUT_A': depth_raster, 'BAND_A': 1,
                'INPUT_B': feature_stack_raster, 'BAND_B': 1,
                'FORMULA': formula,
                'NO_DATA': -9999.0,
                'RTYPE': 5,  # Float32
                'OUTPUT': out_path
            },
            context=context, feedback=feedback, is_child_algorithm=True
        )
        return calc_res['OUTPUT']

    def slope_filter_depth(self, depth_raster, slope_threshold, out_path, context, feedback):
        """Removes unrealistic sharp depth jumps near shorelines based on slope."""
        feedback.pushInfo(f">>> Applying slope filter (threshold={slope_threshold} degrees)...")
        slope_temp = out_path.replace(".tif", "_slope.tif")

        processing.run(
            "gdal:slope",
            {
                'INPUT': depth_raster,
                'BAND': 1,
                'SCALE': 1.0,
                'AS_PERCENT': False,
                'COMPUTE_EDGES': True,
                'ZEVENBERGEN': False,
                'OUTPUT': slope_temp
            },
            context=context, feedback=feedback, is_child_algorithm=True
        )

        formula = f'A * (B <= {slope_threshold}) + (-9999.0) * (B > {slope_threshold})'
        result = processing.run(
            "gdal:rastercalculator",
            {
                'INPUT_A': depth_raster, 'BAND_A': 1,
                'INPUT_B': slope_temp,   'BAND_B': 1,
                'FORMULA': formula,
                'NO_DATA': -9999.0,
                'RTYPE': 5,
                'OUTPUT': out_path
            },
            context=context, feedback=feedback, is_child_algorithm=True
        )
        return result['OUTPUT']

    def remove_positive_pixels(self, in_path, out_path, feedback):
        """Masks out any positive values (>= 0) from the raster using rasterio."""
        feedback.pushInfo(f">>> Removing positive values (>= 0) from: {in_path}")
        try:
            with rasterio.open(in_path) as src:
                data = src.read(1)
                meta = src.profile
                nodata_val = src.nodata if src.nodata is not None else -9999.0

            # Mask identifying >= 0 values
            mask = (data >= 0) & (data != nodata_val)
            
            # Replace positive values with NoData
            data[mask] = nodata_val

            meta.update(
                dtype='float32',
                nodata=nodata_val,
                count=1
            )

            with rasterio.open(out_path, 'w', **meta) as dst:
                dst.write(data.astype(np.float32), 1)
                
            return out_path
        except Exception as e:
            feedback.pushWarning(f"Failed to remove positive pixels: {str(e)}")
            return in_path


    # =======================================================================
    # 6. MAIN EXECUTION PIPELINE
    # =======================================================================
    def processAlgorithm(self, parameters, context, feedback):

        # --- [0] Base Setup & Extraction ---
        out_dir = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        os.makedirs(out_dir, exist_ok=True)
        
        log_path = os.path.join(out_dir, "SDB_Full_Log.txt")
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"SDB LOG - {datetime.datetime.now()}\n\n")

        self.append_log(">>> Workflow Started...", log_path, feedback)
        
        input_raster          = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        target_crs            = input_raster.crs()
        final_water_mask      = None

        max_depth             = self.parameterAsDouble(parameters, self.MAX_DEPTH_THRESHOLD, context)
        shrink_dist           = self.parameterAsDouble(parameters, self.SHRINK_EDGE_DIST, context)
        
        remove_positives_flag = self.parameterAsBool(parameters, self.REMOVE_POSITIVES, context)
        apply_slope_filter    = self.parameterAsBool(parameters, self.ENABLE_SLOPE_FILTER, context)
        slope_threshold_val   = self.parameterAsDouble(parameters, self.SLOPE_THRESHOLD, context)

        # --- [1] PRE-CLIP: Apply and Shrink Water Mask ---
        water_mask_poly = self.parameterAsVectorLayer(parameters, self.WATER_MASK_POLY, context)
        if water_mask_poly:
            self.append_log("\n>>> Pre-Clipping: Applying Ready-made Water Mask Polygon...", log_path, feedback)

            temp_mask_path   = os.path.join(out_dir, 'temp_water_mask.gpkg')
            final_water_mask = self.reproject_layer_if_needed(water_mask_poly, target_crs, temp_mask_path, context, feedback)

            # Fix geometries
            fixed_mask_path = os.path.join(out_dir, 'temp_water_mask_fixed.gpkg')
            fix_res = processing.run(
                "native:fixgeometries",
                {'INPUT': final_water_mask, 'OUTPUT': fixed_mask_path},
                context=context, feedback=feedback, is_child_algorithm=True
            )
            final_water_mask = fix_res['OUTPUT']

            # Shrink polygon
            if shrink_dist < 0:
                self.append_log(f">>> Shrinking Water Polygon by {shrink_dist} units to remove Edge Effects...", log_path, feedback)
                shrunk_path = os.path.join(out_dir, 'temp_water_mask_shrunk.gpkg')
                buffer_res = processing.run(
                    "native:buffer",
                    {
                        'INPUT': final_water_mask,
                        'DISTANCE': shrink_dist,
                        'SEGMENTS': 5,
                        'END_CAP_STYLE': 0, 'JOIN_STYLE': 0, 'MITER_LIMIT': 2,
                        'DISSOLVE': False,
                        'OUTPUT': shrunk_path
                    },
                    context=context, feedback=feedback, is_child_algorithm=True
                )
                final_water_mask = buffer_res['OUTPUT']

            # Clip raster
            clipped_raster_path = os.path.join(out_dir, 'temp_masked_input_raster.tif')
            clip_res = processing.run(
                "gdal:cliprasterbymasklayer",
                {
                    'INPUT': input_raster,
                    'MASK': final_water_mask,
                    'SOURCE_CRS': target_crs,
                    'TARGET_CRS': target_crs,
                    'NODATA': -9999,
                    'ALPHA_BAND': False,
                    'CROP_TO_CUTLINE': False,
                    'KEEP_RESOLUTION': True,
                    'DATA_TYPE': 6,
                    'OUTPUT': clipped_raster_path
                },
                context=context, feedback=feedback, is_child_algorithm=True
            )
            if not os.path.exists(clipped_raster_path):
                raise QgsProcessingException("ERROR: Failed to clip the raster.")
            
            input_raster = QgsRasterLayer(clip_res['OUTPUT'], "Masked Input Raster")

        # --- [2] Reproject & Filter Training / Validation Points ---
        field_depth = self.parameterAsString(parameters, self.FIELD_DEPTH, context)
        temp_train  = os.path.join(out_dir, 'temp_reprojected_train.gpkg')
        
        final_train = self.reproject_layer_if_needed(
            self.parameterAsVectorLayer(parameters, self.INPUT_TRAIN, context),
            target_crs, temp_train, context, feedback
        )
        final_train = self.filter_by_depth(final_train, field_depth, max_depth, context, feedback)

        enable_val = self.parameterAsBool(parameters, self.ENABLE_VALIDATION, context)
        final_test = None
        
        if enable_val:
            t_layer = self.parameterAsVectorLayer(parameters, self.INPUT_TEST, context)
            if t_layer:
                temp_test  = os.path.join(out_dir, 'temp_reprojected_test.gpkg')
                final_test = self.reproject_layer_if_needed(t_layer, target_crs, temp_test, context, feedback)
                final_test = self.filter_by_depth(
                    final_test,
                    self.parameterAsString(parameters, self.FIELD_TEST_DEPTH, context),
                    max_depth, context, feedback
                )
            else:
                enable_val = False

        # -------------------------------------------------------------------
        # PHASE 1: Pre-processing Execution
        # -------------------------------------------------------------------
        self.append_log("\n>>> Phase 1: Pre-processing...", log_path, feedback)
        p1 = processing.run(
            "sdb_tools:sdb_phase1_preprocessing",
            {
                'INPUT_RASTER':         input_raster,
                'COASTAL_BAND':         parameters[self.COASTAL_BAND],
                'BLUE_BAND':            parameters[self.BLUE_BAND],
                'GREEN_BAND':           parameters[self.GREEN_BAND],
                'RED_BAND':             parameters[self.RED_BAND],
                'NIR_BAND':             parameters[self.NIR_BAND],
                'APPLY_SUNGLINT':       parameters[self.APPLY_SUNGLINT],
                'NIR_BAND_SUNGLINT':    parameters[self.NIR_BAND_SUNGLINT],
                'SUNGLINT_PERCENTILE':  parameters[self.SUNGLINT_PERCENTILE],
                'INPUT_WATER_POLY':     final_water_mask if water_mask_poly else None,
                'ENABLE_MASKING':       parameters[self.ENABLE_MASKING],
                'MASKING_METHOD':       parameters[self.MASKING_METHOD],
                'MANUAL_THRESHOLD':     parameters[self.MANUAL_THRESHOLD],
                'OTSU_ADJUSTMENT':      parameters[self.OTSU_ADJUSTMENT],
                'MASK_KERNEL_SIZE':     parameters[self.MASK_KERNEL_SIZE],
                'FEATURE_SELECTION':    parameters[self.FEATURE_SELECTION],
                'ENABLE_BAND_CALC':     parameters[self.ENABLE_BAND_CALC],
                'BAND_MATH_FORMULA':    parameters[self.BAND_MATH_FORMULA],
                'NUM_THREADS':          parameters[self.NUM_THREADS],
                'OUTPUT_FOLDER':        out_dir
            },
            context=context, feedback=feedback, is_child_algorithm=True
        )

        # POST-CLIP: Strict "knife" cut on Phase 1 outputs
        if water_mask_poly and final_water_mask:
            self.append_log("\n>>> Applying Strict 'Knife' Cut to Phase 1 Outputs...", log_path, feedback)

            feat_clipped = os.path.join(out_dir, 'clipped_features_stack.tif')
            processing.run(
                "gdal:cliprasterbymasklayer",
                {
                    'INPUT': p1['OUTPUT_FEATURES'], 'MASK': final_water_mask,
                    'SOURCE_CRS': target_crs, 'TARGET_CRS': target_crs,
                    'NODATA': -9999.0, 'ALPHA_BAND': False,
                    'CROP_TO_CUTLINE': False, 'KEEP_RESOLUTION': True, 'DATA_TYPE': 0,
                    'OUTPUT': feat_clipped
                },
                context=context, feedback=feedback, is_child_algorithm=True
            )
            if os.path.exists(feat_clipped):
                p1['OUTPUT_FEATURES'] = feat_clipped

            if p1.get('OUTPUT_MASK') and os.path.exists(p1['OUTPUT_MASK']):
                mask_clipped = os.path.join(out_dir, 'clipped_water_mask.tif')
                processing.run(
                    "gdal:cliprasterbymasklayer",
                    {
                        'INPUT': p1['OUTPUT_MASK'], 'MASK': final_water_mask,
                        'SOURCE_CRS': target_crs, 'TARGET_CRS': target_crs,
                        'NODATA': -9999, 'ALPHA_BAND': False,
                        'CROP_TO_CUTLINE': False, 'KEEP_RESOLUTION': True, 'DATA_TYPE': 6,
                        'OUTPUT': mask_clipped
                    },
                    context=context, feedback=feedback, is_child_algorithm=True
                )
                if os.path.exists(mask_clipped):
                    p1['OUTPUT_MASK'] = mask_clipped

        # -------------------------------------------------------------------
        # PHASE 2: Filtering & Uncertainty Execution
        # -------------------------------------------------------------------
        path_clean = final_train
        if self.parameterAsBool(parameters, self.ENABLE_RANSAC, context):
            self.append_log("\n>>> Phase 2: Filtering & Uncertainty...", log_path, feedback)
            p2 = processing.run(
                "sdb_tools:sdb_02_filtering",
                {
                    'INPUT_STACK':        p1['OUTPUT_FEATURES'],
                    'INPUT_POINTS':       final_train,
                    'FIELD_DEPTH':        field_depth,
                    'BLUE_BAND':          parameters[self.BLUE_BAND],
                    'GREEN_BAND':         parameters[self.GREEN_BAND],
                    'FILTER_MODE':        parameters[self.FILTER_MODE],
                    'RESIDUAL_THRESHOLD': parameters[self.RANSAC_THRESHOLD],
                    'RANSAC_MAX_TRIALS':  parameters[self.RANSAC_MAX_TRIALS],
                    'OUTPUT_FOLDER':      out_dir
                },
                context=context, feedback=feedback, is_child_algorithm=True
            )
            path_clean = p2['OUTPUT_CLEAN_VEC']

        # -------------------------------------------------------------------
        # PHASE 3: Global Modeling Execution
        # -------------------------------------------------------------------
        self.append_log("\n>>> Phase 3: Global Modeling...", log_path, feedback)
        p3 = processing.run(
            "sdb_tools:sdb_03_initial_modeling",
            {
                'INPUT_STACK':        p1['OUTPUT_FEATURES'],
                'INPUT_MASK':         p1['OUTPUT_MASK'],
                'INPUT_POINTS':       path_clean,
                'FIELD_DEPTH':        field_depth,
                'FIELD_WEIGHT':       self.parameterAsString(parameters, self.FIELD_WEIGHT, context),
                'SELECTED_ALGOS':     parameters[self.SELECTED_ALGOS],
                'OPTIMIZER_METHOD':   parameters[self.OPTIMIZER_METHOD],
                'COLLISION_HANDLING': parameters[self.COLLISION_HANDLING],
                'N_ITERATIONS':       parameters[self.N_ITERATIONS],
                'MEDIAN_SIZE':        parameters[self.MEDIAN_SIZE],
                'OUTPUT_FOLDER':      out_dir,
                'LOG_FILE':           log_path,
                'PARAM_RF':           parameters[self.PARAM_RF],
                'PARAM_GB':           parameters[self.PARAM_GB],
                'PARAM_ET':           parameters[self.PARAM_ET],
                'PARAM_SVR':          parameters[self.PARAM_SVR],
                'PARAM_MLP':          parameters[self.PARAM_MLP]
            },
            context=context, feedback=feedback, is_child_algorithm=True
        )
        
        if 'BEST_R2' in p3:
            self.append_log(f"[Phase 3] R2: {p3['BEST_R2']:.4f}", log_path, feedback)

        # -------------------------------------------------------------------
        # PHASE 4: Adaptive Refinement Execution
        # -------------------------------------------------------------------
        path_refined = p3['OUTPUT_DEPTH_MAP']
        if self.parameterAsBool(parameters, self.ENABLE_ADAPTIVE, context):
            self.append_log("\n>>> Phase 4: Adaptive Refinement...", log_path, feedback)
            
            ad_layer       = self.parameterAsVectorLayer(parameters, self.INPUT_ADAPTIVE_TRAIN, context)
            temp_adapt     = os.path.join(out_dir, 'temp_reprojected_adaptive.gpkg')
            final_ad       = self.reproject_layer_if_needed(ad_layer, target_crs, temp_adapt, context, feedback)
            field_ad_depth = self.parameterAsString(parameters, self.FIELD_ADAPTIVE_DEPTH, context)
            final_ad       = self.filter_by_depth(final_ad, field_ad_depth, max_depth, context, feedback)

            p4 = processing.run(
                "sdb_tools:sdb_phase4_adaptive",
                {
                    'INPUT_GLOBAL_RASTER':   p3['OUTPUT_DEPTH_MAP'],
                    'INPUT_ORIGINAL_FEAT':   p1['OUTPUT_FEATURES'],
                    'INPUT_UNCERTAINTY':     None,
                    'INPUT_MASK':            p1['OUTPUT_MASK'],
                    'INPUT_TRAIN':           final_ad,
                    'FIELD_TRAIN':           field_ad_depth,
                    'SELECTED_ALGOS':        parameters[self.SELECTED_ALGOS],
                    'OPTIMIZER_METHOD':      parameters[self.OPTIMIZER_METHOD],
                    'COLLISION_HANDLING':    parameters[self.COLLISION_HANDLING],
                    'N_ITERATIONS':          parameters[self.N_ITERATIONS],
                    'MEDIAN_SIZE':           parameters[self.MEDIAN_SIZE],
                    'OUTPUT_FOLDER':         out_dir,
                    'LOG_FILE':              log_path,
                    'PARAM_RF':              parameters[self.PARAM_RF],
                    'PARAM_GB':              parameters[self.PARAM_GB],
                    'PARAM_ET':              parameters[self.PARAM_ET],
                    'PARAM_SVR':             parameters[self.PARAM_SVR],
                    'PARAM_MLP':             parameters[self.PARAM_MLP]
                },
                context=context, feedback=feedback, is_child_algorithm=True
            )
            path_refined = p4['OUTPUT_FINAL']
            
            if 'BEST_R2' in p4:
                self.append_log(f"[Phase 4] R2: {p4['BEST_R2']:.4f}", log_path, feedback)

        # -------------------------------------------------------------------
        # POST-PROCESS: Output Cleanup (Clamping, Slope, Positives)
        # -------------------------------------------------------------------
        feat_stack = p1['OUTPUT_FEATURES']

        # --- Clean Initial Map (Phase 3) ---
        if p3.get('OUTPUT_DEPTH_MAP') and os.path.exists(p3['OUTPUT_DEPTH_MAP']):
            p3_clamped = os.path.join(out_dir, 'Phase3_Depth_Cleaned.tif')
            self.clean_depth_map(p3['OUTPUT_DEPTH_MAP'], feat_stack, max_depth, p3_clamped, context, feedback)
            
            if remove_positives_flag:
                p3_no_pos = os.path.join(out_dir, 'Phase03_Depth_Final_NoPositives.tif')
                self.remove_positive_pixels(p3_clamped, p3_no_pos, feedback)
                p3['OUTPUT_DEPTH_MAP'] = p3_no_pos
            else:
                p3['OUTPUT_DEPTH_MAP'] = p3_clamped

        # --- Clean Refined Map (Phase 4) ---
        if path_refined and os.path.exists(path_refined):
            p4_clamped = os.path.join(out_dir, 'Final_Depth_Cleaned.tif')
            self.clean_depth_map(path_refined, feat_stack, max_depth, p4_clamped, context, feedback)

            # Apply Optional Slope Filter
            if apply_slope_filter:
                slope_filtered = os.path.join(out_dir, 'Final_Depth_SlopeFiltered.tif')
                path_refined = self.slope_filter_depth(
                    p4_clamped,
                    slope_threshold=slope_threshold_val,
                    out_path=slope_filtered,
                    context=context,
                    feedback=feedback
                )
            else:
                path_refined = p4_clamped
            
            # Remove Positives
            if remove_positives_flag:
                p4_no_pos = os.path.join(out_dir, 'Phase04_Final_Depth_NoPositives.tif')
                self.remove_positive_pixels(path_refined, p4_no_pos, feedback)
                path_refined = p4_no_pos

        # -------------------------------------------------------------------
        # PHASE 5: Validation Execution
        # -------------------------------------------------------------------
        if enable_val and final_test:
            self.append_log("\n>>> Phase 5: Validation...", log_path, feedback)
            processing.run(
                "sdb_tools:sdb_05_reporting",
                {
                    'INPUT_MAP_P3':     p3['OUTPUT_DEPTH_MAP'],
                    'INPUT_MAP_P4':     path_refined,
                    'INPUT_TRAIN':      path_clean,
                    'FIELD_TRAIN':      field_depth,
                    'INPUT_VALIDATION': final_test,
                    'FIELD_VAL_DEPTH':  self.parameterAsString(parameters, self.FIELD_TEST_DEPTH, context),
                    'OUTPUT_FOLDER':    out_dir
                },
                context=context, feedback=feedback, is_child_algorithm=True
            )

        # -------------------------------------------------------------------
        # CANVAS LOADING
        # -------------------------------------------------------------------
        if p3.get('OUTPUT_DEPTH_MAP') and os.path.exists(p3['OUTPUT_DEPTH_MAP']):
            details_init = QgsProcessingContext.LayerDetails('Initial SDB Map [Phase 3]', QgsProject.instance(), 'Initial SDB')
            context.addLayerToLoadOnCompletion(p3['OUTPUT_DEPTH_MAP'], details_init)

        if path_refined and os.path.exists(path_refined):
            details_ref = QgsProcessingContext.LayerDetails('Refined SDB Map [Phase 4]', QgsProject.instance(), 'Refined SDB')
            context.addLayerToLoadOnCompletion(path_refined, details_ref)

        self.append_log("\n>>> Workflow Complete.", log_path, feedback)
        
        return {}