import warnings

from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
    QgsProcessingParameterBand,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterString,
    QgsProcessingParameterDefinition,
)

warnings.filterwarnings("ignore")


class SDBMasterOrchestrator(QgsProcessingAlgorithm):

    # =======================================================================
    # 1. PARAMETER CONSTANTS
    # =======================================================================
    SELECTED_ALGOS = "SELECTED_ALGOS"
    INPUT_RASTER = "INPUT_RASTER"
    OUTPUT_FOLDER = "OUTPUT_FOLDER"
    NUM_THREADS = "NUM_THREADS"

    # [1] Pre-processing (Bands, Sunglint, Masking, Features)
    ENABLE_PREPROCESSING = "ENABLE_PREPROCESSING"
    COASTAL_BAND = "COASTAL_BAND"
    BLUE_BAND = "BLUE_BAND"
    GREEN_BAND = "GREEN_BAND"
    RED_BAND = "RED_BAND"
    NIR_BAND = "NIR_BAND"
    SWIR_BAND = "SWIR_BAND"

    APPLY_SUNGLINT = "APPLY_SUNGLINT"
    SUNGLINT_PERCENTILE = "SUNGLINT_PERCENTILE"

    WATER_MASK_POLY = "WATER_MASK_POLY"
    SHRINK_EDGE_DIST = "SHRINK_EDGE_DIST"
    ENABLE_MASKING = "ENABLE_MASKING"
    MASKING_METHOD = "MASKING_METHOD"
    MANUAL_THRESHOLD = "MANUAL_THRESHOLD"
    OTSU_ADJUSTMENT = "OTSU_ADJUSTMENT"
    MASK_KERNEL_SIZE = "MASK_KERNEL_SIZE"

    FEATURE_SELECTION = "FEATURE_SELECTION"
    ENABLE_BAND_CALC = "ENABLE_BAND_CALC"
    BAND_MATH_FORMULA = "BAND_MATH_FORMULA"

    # OSW Filter
    APPLY_DEEPWATER = "APPLY_DEEPWATER"
    DEEPWATER_METHOD = "DEEPWATER_METHOD"
    DEEPWATER_ROI = "DEEPWATER_ROI"
    NIR_PERCENTILE_OSW = "NIR_PERCENTILE_OSW"
    OSW_MEDIAN_SIZE = "OSW_MEDIAN_SIZE"
    FILL_INTERNAL_HOLES = "FILL_INTERNAL_HOLES"
    EXTRACT_POLYGON = "EXTRACT_POLYGON"

    # [2] Filtering & Training Data
    INPUT_TRAIN = "INPUT_TRAIN"
    FIELD_DEPTH = "FIELD_DEPTH"
    FIELD_WEIGHT = "FIELD_WEIGHT"
    MAX_DEPTH_THRESHOLD = "MAX_DEPTH_THRESHOLD"

    ENABLE_RANSAC = "ENABLE_RANSAC"
    FILTER_MODE = "FILTER_MODE"
    FILTER_NUMERATOR_BAND = "FILTER_NUMERATOR_BAND"
    FILTER_DENOMINATOR_BAND = "FILTER_DENOMINATOR_BAND"
    RANSAC_THRESHOLD = "RANSAC_THRESHOLD"
    RANSAC_MAX_TRIALS = "RANSAC_MAX_TRIALS"

    # [3] Global Modeling (Auto-ML)
    SELECTED_ALGOS = "SELECTED_ALGOS"
    OPTIMIZER_METHOD = "OPTIMIZER_METHOD"
    COLLISION_HANDLING = "COLLISION_HANDLING"
    N_ITERATIONS = "N_ITERATIONS"
    MEDIAN_SIZE = "MEDIAN_SIZE"
    FEATURE_CORR_METHOD = "FEATURE_CORR_METHOD"
    FEATURE_CORR_THRESHOLD = "FEATURE_CORR_THRESHOLD"

    PARAM_RF = "PARAM_RF"
    PARAM_GB = "PARAM_GB"
    PARAM_ET = "PARAM_ET"
    PARAM_SVR = "PARAM_SVR"
    PARAM_MLP = "PARAM_MLP"
    PARAM_RIDGE = "PARAM_RIDGE"
    PARAM_LASSO = "PARAM_LASSO"
    PARAM_ELASTICNET = "PARAM_ELASTICNET"
    PARAM_KNN = "PARAM_KNN"
    PARAM_DT = "PARAM_DT"
    PARAM_HUBER = "PARAM_HUBER"
    PARAM_XGB = "PARAM_XGB"
    PARAM_LGBM = "PARAM_LGBM"
    PARAM_CATBOOST = "PARAM_CATBOOST"

    ENABLE_ENSEMBLE = "ENABLE_ENSEMBLE"
    ENSEMBLE_METHOD = "ENSEMBLE_METHOD"
    ENSEMBLE_SIZE = "ENSEMBLE_SIZE"
    RESIDUAL_INTERP_METHOD = "RESIDUAL_INTERP_METHOD"
    KNN_NEIGHBORS = "KNN_NEIGHBORS"
    SPATIAL_CV_P3 = "SPATIAL_CV_P3"
    SPATIAL_CV_P4 = "SPATIAL_CV_P4"

    TRAIN_TEST_SPLIT = "TRAIN_TEST_SPLIT"
    RANDOM_STATE = "RANDOM_STATE"
    CV_FOLDS = "CV_FOLDS"
    UNCERT_TREES = "UNCERT_TREES"
    MAX_GPR_SAMPLES = "MAX_GPR_SAMPLES"
    OUTPUT_FORMAT = "OUTPUT_FORMAT"

    # [4] Adaptive Refinement
    ENABLE_ADAPTIVE = "ENABLE_ADAPTIVE"
    ENABLE_ENSEMBLE_P4 = "ENABLE_ENSEMBLE_P4"
    ENSEMBLE_METHOD_P4 = "ENSEMBLE_METHOD_P4"
    ENSEMBLE_SIZE_P4 = "ENSEMBLE_SIZE_P4"
    INPUT_ADAPTIVE_TRAIN = "INPUT_ADAPTIVE_TRAIN"
    FIELD_ADAPTIVE_DEPTH = "FIELD_ADAPTIVE_DEPTH"
    STACK_COMPONENTS_P4 = "STACK_COMPONENTS_P4"
    FEATURE_CORR_METHOD_P4 = "FEATURE_CORR_METHOD_P4"
    FEATURE_CORR_THRESHOLD_P4 = "FEATURE_CORR_THRESHOLD_P4"

    # [5] Validation & Output Cleanup
    ENABLE_VALIDATION = "ENABLE_VALIDATION"
    INPUT_TEST = "INPUT_TEST"
    FIELD_TEST_DEPTH = "FIELD_TEST_DEPTH"

    ENABLE_SLOPE_FILTER = "ENABLE_SLOPE_FILTER"
    SLOPE_THRESHOLD = "SLOPE_THRESHOLD"
    REMOVE_POSITIVES = "REMOVE_POSITIVES"

    # =======================================================================
    # 2. OPTION LISTS (DROPDOWNS)
    # =======================================================================
    FILTER_MODES_NAMES = ["Linear RANSAC", "LS Variance Fit", "Huber Variance Fit"]
    MODEL_LIST_NAMES = [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting",
        "Extra Trees",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "KNN",
        "Decision Tree",
        "MLP",
        "SVR",
        "Huber Regressor",
        "XGBoost",
        "LightGBM",
        "CatBoost",
    ]
    OPTIMIZER_LIST_NAMES = ["Random Search", "Grid Search", "Bayesian Search"]
    COLLISION_LIST_NAMES = [
        "Keep All Points",
        "Highest Confidence",
        "Closest to Pixel Center",
        "Hybrid",
        "Strict Center",
    ]
    MASK_METHODS_NAMES = ["Otsu (Automatic NDWI)", "Manual NDWI Threshold", "3 Indices Equation (NDWI, MNDWI, NWI)"]
    OSW_METHODS_NAMES = ["Manual Polygon ROI", "Automatic (Lowest NIR Percentile)", "Shallow Water Bound (OSW Polygon)"]
    FEATURE_CORR_THRESHOLDS = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    FEATURE_CORR_THRESHOLDS_P4 = ["Use Phase 03 (-1.0)", "0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    FEATURE_OPTIONS_NAMES = [
        "[All Raw] All Bands from Input Image",
        "[Log] Log(Coastal)",
        "[Log] Log(Blue)",
        "[Log] Log(Green)",
        "[Log] Log(Red)",
        "[Log] Log(NIR)",
        "[Ratio] Log(Blue) / Log(Green)",
        "[Ratio] Log(Blue) / Log(Red)",
        "[Ratio] Log(Coastal) / Log(Green)",
        "[Ratio] Log(Green) / Log(NIR)",
        "[Ratio] Log(Red) / Log(NIR)",
        "[Index] NDWI (Green - NIR) / (Green + NIR)",
        "[Custom] Band Math Calculator",
    ]

    # =======================================================================
    # 3. ALGORITHM METADATA & HELP STRINGS
    # =======================================================================
    def name(self):
        return "sdb_master_orchestrator"

    def displayName(self):
        return "SDB Master Workflow (Full Pipeline)"

    def group(self):
        return ""

    def groupId(self):
        return ""

    def createInstance(self):
        return SDBMasterOrchestrator()

    def shortHelpString(self):
        return """
        <div style="font-family: Arial, sans-serif; line-height: 1.2;">
            <h2 style="margin-bottom: 5px;">🛰️ <span style="color: #2E86C1;">Bathymetrix-AI</span>: Master SDB Workflow</h2>
            <p style="margin-top: 0; margin-bottom: 10px;">An advanced 5-phase pipeline for high-precision Satellite-Derived Bathymetry with Auto-ML.</p>

            <b style="display: block; margin-bottom: 2px;">🌊 Phase 01: Advanced Pre-processing & Masking</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li><b>Glint Correction:</b> Sun-glint correction using Hedley's method.</li>
                <li><b>Automated Masking:</b> NDWI, MNDWI, or 3-Indices equations with water edge shrink.</li>
                <li><b>Deep Water Filter:</b> Automatic (lowest NIR percentile) or manual polygon OSW mask.</li>
                <li><b>Custom Calculator:</b> Build custom features with band math formulas.</li>
            </ul>

            <b style="display: block; margin-bottom: 2px;">🎯 Phase 02: Robust Altimetry Filtering</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li><b>Outlier Rejection:</b> Aggressively clean altimetry noise using Linear RANSAC, LS Variance Fit, or Huber Variance Fit.</li>
            </ul>

            <b style="display: block; margin-bottom: 2px;">🤖 Phase 03: Global Auto-ML & Feature Analysis</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li><b>Benchmarking:</b> Ranks 15+ machine learning algorithms (RF, GB, MLP, SVR, etc.).</li>
                <li><b>Hyperparameter Tuning:</b> Optimization via Random, Grid, or Bayesian Search.</li>
                <li><b>Feature Analysis:</b> Drop weak bands using dropdown threshold selection (default: Automatic-RANSAC, with Disabled option).</li>
                <li><b>Spatial Block CV:</b> Independent control of spatial validation block settings.</li>
            </ul>

            <b style="display: block; margin-bottom: 2px;">📍 Phase 04: Localized Adaptive Refinement</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li><b>Residual Modeling:</b> Map local errors using Standard KNN, Robust KNN (Huber), or Kriging/GPR.</li>
                <li><b>Flexible Refinement Stack:</b> Retrain on depth map, error grid, or full feature stack (unchecked by default).</li>
                <li><b>Ensemble Blending:</b> Blend top models for localized spatial corrections.</li>
            </ul>

            <b style="display: block; margin-bottom: 2px;">📉 Phase 05: Scientific Validation</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li><b>Accuracy Reports:</b> Computes RMSE, R², MAE, and wMAPE against unseen validation points.</li>
                <li><b>Visual Diagnostics:</b> Auto-generates density scatter, residuals, and error histogram plots.</li>
            </ul>

            <b style="display: block; margin-bottom: 2px;">🧽 Output Cleanup & Formats</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li><b>Post-processing:</b> Positive depth removal and physical slope spike filtering.</li>
                <li><b>Format Support:</b> Outputs float32, float64, or uint16 rasters.</li>
            </ul>

            <b style="display: block; margin-bottom: 2px; margin-top: 15px;">📚 Key References</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px; font-size: 12px;">
                <li><b>Stumpf et al. (2003):</b> Log-Ratio Algorithm for SDB inversion.</li>
                <li><b>Hedley et al. (2005):</b> Physics-based sun-glint correction.</li>
                <li><b>Wheaton et al. (2010):</b> Accounting for uncertainty in DEMs from repeat topographic surveys.</li>
                <li><b>Lane & Chandler (2003):</b> The application of topographic surveying to fluvial studies.</li>
            </ul>

            <p style="margin-top: 10px; border-top: 1px solid #ccc; padding-top: 5px;">
                <b>Developer:</b> Mohamed Aly Nasef
            </p>
        </div>
        """

    def helpString(self):
        return """<b>SDB Master Workflow (Full Pipeline)</b><br><br>
        This tool executes a complete Auto-ML pipeline for Satellite-Derived Bathymetry (SDB) using a 5-phase scientific methodology.<br><br>
        <b>Outputs Explained:</b><br>
        • <b>Phase 01:</b> Outputs intermediate feature layers including sun-glint corrected bands, water masks, and physical log-ratio features.<br>
        • <b>Phase 02:</b> Outputs the filtered and robust ground truth altimetry data (e.g., ICESat-2).<br>
        • <b>Phase 03 (Initial SDB Map):</b> The base depth map produced after global machine learning benchmarking and hyperparameter optimization.<br>
        • <b>Phase 04 (Refined SDB Map):</b> The final, highly accurate depth map after applying adaptive localized corrections based on spatial residual analysis.<br>
        • <b>Phase 05 (Validation):</b> Outputs comprehensive accuracy assessment reports and error distribution charts.<br><br>
        <i>* Note: All output files are automatically saved to your specified 'Main Output Folder' and loaded cleanly into the map canvas upon completion.</i>
        """

    # =======================================================================
    # 4. ALGORITHM INIT (QGIS FRONT-END UI SETUP)
    # =======================================================================
    def initAlgorithm(self, config=None):

        # -------------------------------------------------------------------
        # [0] General Settings
        # -------------------------------------------------------------------
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_RASTER, "📁 [0] Input Satellite Image"
            )
        )
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_FOLDER, "📁 [0] Main Output Folder"
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.NUM_THREADS,
                "⚙️ [0] Processing Threads",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=4,
            )
        )

        # -------------------------------------------------------------------
        # [1] Phase 01: Advanced Pre-processing
        # -------------------------------------------------------------------
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_PREPROCESSING,
                "⚙️ [1] Enable Phase 01 Pre-processing",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                self.COASTAL_BAND,
                "📡 [1.1] Coastal Band",
                parentLayerParameterName=self.INPUT_RASTER,
                defaultValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                self.BLUE_BAND,
                "📡 [1.1] Blue Band",
                parentLayerParameterName=self.INPUT_RASTER,
                defaultValue=2,
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                self.GREEN_BAND,
                "📡 [1.1] Green Band",
                parentLayerParameterName=self.INPUT_RASTER,
                defaultValue=3,
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                self.RED_BAND,
                "📡 [1.1] Red Band",
                parentLayerParameterName=self.INPUT_RASTER,
                defaultValue=4,
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                self.NIR_BAND,
                "🌍 [1] NIR Band",
                parentLayerParameterName=self.INPUT_RASTER,
                defaultValue=8,
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                self.SWIR_BAND,
                "🌍 [1] SWIR Band (For 3 Indices Mask)",
                parentLayerParameterName=self.INPUT_RASTER,
                defaultValue=11,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.APPLY_SUNGLINT,
                "☀️ [1.2] Apply Sunglint Correction",
                defaultValue=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.SUNGLINT_PERCENTILE,
                "☀️ [1.2] Sunglint Deep Water %",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1.0,
            )
        )

        # -------------------------------------------------------------------
        # [1] Phase 01: Advanced Pre-processing (Masking & Features)
        # -------------------------------------------------------------------
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.WATER_MASK_POLY,
                "🗺️ [1.3] Ready-made Water Mask Polygon",
                types=[QgsProcessing.TypeVectorPolygon],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SHRINK_EDGE_DIST,
                "🗺️ [1.3] Water Edge Shrink (Map Units, e.g. -10)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_MASKING,
                "🏖️ [1.3] Enable Automated Water Masking",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.MASKING_METHOD,
                "🏖️ [1.3] Water Masking Method",
                options=self.MASK_METHODS_NAMES,
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MANUAL_THRESHOLD,
                "🏖️ [1.3] Manual Threshold",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.OTSU_ADJUSTMENT,
                "🏖️ [1.3] Otsu Threshold Adjustment",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MASK_KERNEL_SIZE,
                "🏖️ [1.3] Mask Cleanup Kernel Size",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=3,
            )
        )

        default_feats = list(range(len(self.FEATURE_OPTIONS_NAMES)))
        self.addParameter(
            QgsProcessingParameterEnum(
                self.FEATURE_SELECTION,
                "📊 [1.4] Output Feature Stack",
                options=self.FEATURE_OPTIONS_NAMES,
                allowMultiple=True,
                defaultValue=default_feats,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_BAND_CALC,
                "🧮 [1.4] Enable Custom Band Math",
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.BAND_MATH_FORMULA,
                "🧮 [1.4] Band Math Formula",
                defaultValue="",
                optional=True,
            )
        )

        # -------------------------------------------------------------------
        # [1] Phase 01: Advanced Pre-processing (OSW Filter)
        # -------------------------------------------------------------------
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.APPLY_DEEPWATER,
                "🌊 [1.5] Apply Deep Water Filter",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.DEEPWATER_METHOD,
                "🌊 [1.5] Deep Water Definition Method",
                options=self.OSW_METHODS_NAMES,
                defaultValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.DEEPWATER_ROI,
                "🌊 [1.5] Deep Water ROI (Polygon) [Optional if Manual Mode]",
                types=[QgsProcessing.TypeVectorPolygon],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.NIR_PERCENTILE_OSW,
                "🌊 [1.5] NIR Percentile for Deep Water (e.g. 10%)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=10.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.OSW_MEDIAN_SIZE,
                "🌊 [1.5] OSW Mask Median Filter Size",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=3,
            )
        )
        p_fill = QgsProcessingParameterBoolean(
            self.FILL_INTERNAL_HOLES,
            "🌊 [1.5] Fill All Internal Holes in OSW Mask",
            defaultValue=True,
        )
        p_fill.setFlags(p_fill.flags() | QgsProcessingParameterDefinition.FlagHidden)
        self.addParameter(p_fill)

        p_extract = QgsProcessingParameterBoolean(
            self.EXTRACT_POLYGON,
            "🌊 [1.5] Extract OSW Mask as Polygon",
            defaultValue=True,
        )
        p_extract.setFlags(p_extract.flags() | QgsProcessingParameterDefinition.FlagHidden)
        self.addParameter(p_extract)

        # -------------------------------------------------------------------
        # [2] Phase 02: Robust Filtering
        # -------------------------------------------------------------------
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_TRAIN, "📍 [2.1] Main Training Points"
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_DEPTH,
                "📏 [2.1] Depth Field",
                parentLayerParameterName=self.INPUT_TRAIN,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_WEIGHT,
                "⚖️ [2.1] Weight Field",
                parentLayerParameterName=self.INPUT_TRAIN,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_DEPTH_THRESHOLD,
                "🛑 [2.1] Maximum Depth Threshold (e.g. -30)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=-30.0,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_RANSAC,
                "🧹 [2.2] Enable Data Filtering (Noise Removal)",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.FILTER_MODE,
                "🧹 [2.2] Filtering Strategy",
                options=self.FILTER_MODES_NAMES,
                defaultValue=2,
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                self.FILTER_NUMERATOR_BAND,
                "🧹 [2.2] Log-Ratio Numerator Band",
                parentLayerParameterName=self.INPUT_RASTER,
                defaultValue=2,
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                self.FILTER_DENOMINATOR_BAND,
                "🧹 [2.2] Log-Ratio Denominator Band",
                parentLayerParameterName=self.INPUT_RASTER,
                defaultValue=3,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.RANSAC_THRESHOLD,
                "🧹 [2.2] Threshold / Sigma Multiplier",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=3.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.RANSAC_MAX_TRIALS,
                "🧹 [2.2] RANSAC Trials",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=100,
            )
        )

        # -------------------------------------------------------------------
        # [3] Phase 03: Global Auto-ML & Feature Analysis
        # -------------------------------------------------------------------
        self.addParameter(
            QgsProcessingParameterEnum(
                self.SELECTED_ALGOS,
                "🤖 [3] Algorithms to Benchmark",
                options=self.MODEL_LIST_NAMES,
                allowMultiple=True,
                defaultValue=[3, 12, 13, 14], # Extra Trees, XGBoost, LightGBM, CatBoost
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.OPTIMIZER_METHOD,
                "🤖 [3] Optimizer Method",
                options=self.OPTIMIZER_LIST_NAMES,
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.COLLISION_HANDLING,
                "🤖 [3] Collision Handling",
                options=self.COLLISION_LIST_NAMES,
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.N_ITERATIONS,
                "🤖 [3] Optimization Iterations",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=20,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_ENSEMBLE,
                "⚙️ [3] Enable Ensemble of Top Models",
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.ENSEMBLE_METHOD,
                "📊 [3] Ensemble Blending Method",
                options=["Average", "Median", "Stacking"],
                defaultValue=0,
            )
        )
        p_ens_size = QgsProcessingParameterNumber(
            self.ENSEMBLE_SIZE,
            "📊 [Phase 03] Ensemble Size (Top N Models to blend)",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=3,
            minValue=2,
            maxValue=5,
        )
        p_ens_size.setFlags(p_ens_size.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_ens_size)
        p_sp_p3 = QgsProcessingParameterBoolean(
            self.SPATIAL_CV_P3,
            "🌍 [Phase 03] Enable Spatial Block Cross-Validation",
            defaultValue=False,
        )
        p_sp_p3.setFlags(p_sp_p3.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_sp_p3)
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MEDIAN_SIZE,
                "🤖 [3] Output Median Filter Size",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=5,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.FEATURE_CORR_METHOD,
                "🤖 [3] Feature Correlation Method",
                options=["Disabled", "Pearson (Linear)", "Spearman (Rank)", "Automatic-RANSAC", "Automatic-Random Forest"],
                defaultValue=3,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.FEATURE_CORR_THRESHOLD,
                "🤖 [3] Feature Correlation Threshold",
                options=self.FEATURE_CORR_THRESHOLDS,
                defaultValue=2,
            )
        )

        # Hyperparameters and ML Settings
        p_split = QgsProcessingParameterNumber(
            self.TRAIN_TEST_SPLIT, "🎛️ [Phase 03 & 04] Training Data Ratio (e.g., 0.8 for 80%)", type=QgsProcessingParameterNumber.Double, defaultValue=0.8
        )
        p_split.setFlags(p_split.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_split)

        p_rs = QgsProcessingParameterNumber(
            self.RANDOM_STATE, "⚙️ [General] Random State for ML Split", type=QgsProcessingParameterNumber.Integer, defaultValue=42
        )
        p_rs.setFlags(p_rs.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_rs)

        p_fmt = QgsProcessingParameterEnum(
            self.OUTPUT_FORMAT, "⚙️ [General] Output Raster Format (Bit Depth)", options=["float32", "float64", "uint16"], defaultValue=0
        )
        p_fmt.setFlags(p_fmt.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_fmt)

        p_cv = QgsProcessingParameterNumber(
            self.CV_FOLDS, "🎛️ [Advanced] ML Cross-Validation Folds", type=QgsProcessingParameterNumber.Integer, defaultValue=5
        )
        p_cv.setFlags(p_cv.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_cv)

        p_uncert = QgsProcessingParameterNumber(
            self.UNCERT_TREES, "🎛️ [Advanced] Uncertainty Model Estimators (Trees)", type=QgsProcessingParameterNumber.Integer, defaultValue=200
        )
        p_uncert.setFlags(p_uncert.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_uncert)

        p_gpr = QgsProcessingParameterNumber(
            self.MAX_GPR_SAMPLES, "📍 [Advanced] Max GPR Training Samples (Phase 04)", type=QgsProcessingParameterNumber.Integer, defaultValue=1500
        )
        p_gpr.setFlags(p_gpr.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_gpr)

        p_rf = QgsProcessingParameterString(self.PARAM_RF, "🎛️ [General ML] Random Forest Hyperparameters", defaultValue="'n_estimators':[100, 500], 'max_depth':[10, 30]")
        p_rf.setFlags(p_rf.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_rf)

        p_gb = QgsProcessingParameterString(self.PARAM_GB, "🎛️ [General ML] Gradient Boosting Hyperparameters", defaultValue="'n_estimators':[100, 300], 'learning_rate':[0.05, 0.1]")
        p_gb.setFlags(p_gb.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_gb)

        p_et = QgsProcessingParameterString(self.PARAM_ET, "🎛️ [General ML] Extra Trees Hyperparameters", defaultValue="'n_estimators':[100, 500], 'max_depth':[10, 30]")
        p_et.setFlags(p_et.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_et)

        p_svr = QgsProcessingParameterString(self.PARAM_SVR, "🎛️ [General ML] SVR Hyperparameters", defaultValue="'C':[1, 10, 100], 'kernel':['rbf'], 'cache_size':[1000], 'max_iter':[20000]")
        p_svr.setFlags(p_svr.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_svr)

        p_mlp = QgsProcessingParameterString(self.PARAM_MLP, "🎛️ [General ML] MLP Hyperparameters", defaultValue="'hidden_layer_sizes':[(100,), (50, 50)], 'max_iter':[500]")
        p_mlp.setFlags(p_mlp.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_mlp)

        p_ridge = QgsProcessingParameterString(self.PARAM_RIDGE, "🎛️ [General ML] Ridge Hyperparameters", defaultValue="'alpha':[0.1, 1.0]", optional=True)
        p_ridge.setFlags(p_ridge.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_ridge)

        p_lasso = QgsProcessingParameterString(self.PARAM_LASSO, "🎛️ [General ML] Lasso Hyperparameters", defaultValue="'alpha':[0.01, 0.1]", optional=True)
        p_lasso.setFlags(p_lasso.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_lasso)

        p_en = QgsProcessingParameterString(self.PARAM_ELASTICNET, "🎛️ [General ML] ElasticNet Hyperparameters", defaultValue="'l1_ratio':[0.5]", optional=True)
        p_en.setFlags(p_en.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_en)

        p_knn = QgsProcessingParameterString(self.PARAM_KNN, "🎛️ [General ML] KNN Hyperparameters", defaultValue="'n_neighbors':[5, 10]", optional=True)
        p_knn.setFlags(p_knn.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_knn)

        p_dt = QgsProcessingParameterString(self.PARAM_DT, "🎛️ [General ML] Decision Tree Hyperparameters", defaultValue="'max_depth':[5, 10]", optional=True)
        p_dt.setFlags(p_dt.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_dt)

        p_huber = QgsProcessingParameterString(self.PARAM_HUBER, "🎛️ [General ML] Huber Hyperparameters", defaultValue="'epsilon':[1.1, 1.35, 1.5]", optional=True)
        p_huber.setFlags(p_huber.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_huber)

        p_xgb = QgsProcessingParameterString(self.PARAM_XGB, "🎛️ [General ML] XGBoost Hyperparameters", defaultValue="'n_estimators':[100, 200], 'max_depth':[4, 6], 'learning_rate':[0.05, 0.1]", optional=True)
        p_xgb.setFlags(p_xgb.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_xgb)

        p_lgbm = QgsProcessingParameterString(self.PARAM_LGBM, "🎛️ [General ML] LightGBM Hyperparameters", defaultValue="'n_estimators':[100, 200], 'max_depth':[4, 6], 'learning_rate':[0.05, 0.1]", optional=True)
        p_lgbm.setFlags(p_lgbm.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_lgbm)

        p_cat = QgsProcessingParameterString(self.PARAM_CATBOOST, "🎛️ [General ML] CatBoost Hyperparameters", defaultValue="'iterations':[100, 200], 'depth':[4, 6], 'learning_rate':[0.05, 0.1]", optional=True)
        p_cat.setFlags(p_cat.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_cat)

        # -------------------------------------------------------------------
        # [4] Phase 04: Adaptive Refinement
        # -------------------------------------------------------------------
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_ADAPTIVE,
                "🎯 [4] Enable Adaptive Refinement",
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.STACK_COMPONENTS_P4,
                "🎯 [4] Features for Retraining",
                options=["Feature Stack (Phase 01)", "Phase 03 Depth Map", "Residual Error Grid"],
                allowMultiple=True,
                defaultValue=[1, 2],
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.FEATURE_CORR_METHOD_P4,
                "🤖 [4] Feature Correlation Method",
                options=["Disabled", "Pearson (Linear)", "Spearman (Rank)", "Automatic-RANSAC", "Automatic-Random Forest"],
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
                options=["Standard KNN", "Robust KNN (Huber Weights)", "Gaussian Process / Kriging"],
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

        p_ens_p4 = QgsProcessingParameterBoolean(
            self.ENABLE_ENSEMBLE_P4,
            "⚙️ [Phase 04] Enable Ensemble of Top Models",
            defaultValue=False,
        )
        p_ens_p4.setFlags(p_ens_p4.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_ens_p4)

        p_ens_meth_p4 = QgsProcessingParameterEnum(
            self.ENSEMBLE_METHOD_P4,
            "📊 [Phase 04] Ensemble Blending Method",
            options=["Average", "Median", "Stacking"],
            defaultValue=0,
        )
        p_ens_meth_p4.setFlags(p_ens_meth_p4.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_ens_meth_p4)

        p_ens_size_p4 = QgsProcessingParameterNumber(
            self.ENSEMBLE_SIZE_P4,
            "📊 [Phase 04] Ensemble Size (Top N Models to blend)",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=3,
            minValue=2,
            maxValue=5,
        )
        p_ens_size_p4.setFlags(p_ens_size_p4.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_ens_size_p4)

        p_sp_p4 = QgsProcessingParameterBoolean(
            self.SPATIAL_CV_P4,
            "🌍 [Phase 04] Enable Spatial Block Cross-Validation",
            defaultValue=False,
        )
        p_sp_p4.setFlags(p_sp_p4.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_sp_p4)
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_ADAPTIVE_TRAIN, "🎯 [4] Adaptive Points", optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_ADAPTIVE_DEPTH,
                "🎯 [4] Adaptive Depth Field",
                parentLayerParameterName=self.INPUT_ADAPTIVE_TRAIN,
                optional=True,
            )
        )

        # -------------------------------------------------------------------
        # [5] Phase 05: Validation & Reporting
        # -------------------------------------------------------------------
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_VALIDATION, "📉 [5] Enable Validation", defaultValue=False
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
                parentLayerParameterName=self.INPUT_TEST,
                optional=True,
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

    def processAlgorithm(self, parameters, context, feedback):
        from ...core.pipeline import run_master_pipeline

        return run_master_pipeline(self, parameters, context, feedback)
