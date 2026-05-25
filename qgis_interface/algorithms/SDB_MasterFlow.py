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
)

warnings.filterwarnings("ignore")


class SDBMasterOrchestrator(QgsProcessingAlgorithm):

    # =======================================================================
    # 1. PARAMETER CONSTANTS
    # =======================================================================

    # [0] General I/O
    INPUT_RASTER = "INPUT_RASTER"
    OUTPUT_FOLDER = "OUTPUT_FOLDER"
    NUM_THREADS = "NUM_THREADS"

    # [1] Pre-processing (Bands, Sunglint, Masking, Features)
    COASTAL_BAND = "COASTAL_BAND"
    BLUE_BAND = "BLUE_BAND"
    GREEN_BAND = "GREEN_BAND"
    RED_BAND = "RED_BAND"
    NIR_BAND = "NIR_BAND"

    APPLY_SUNGLINT = "APPLY_SUNGLINT"
    NIR_BAND_SUNGLINT = "NIR_BAND_SUNGLINT"
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

    # [2] Filtering & Training Data
    INPUT_TRAIN = "INPUT_TRAIN"
    FIELD_DEPTH = "FIELD_DEPTH"
    FIELD_WEIGHT = "FIELD_WEIGHT"
    MAX_DEPTH_THRESHOLD = "MAX_DEPTH_THRESHOLD"

    ENABLE_RANSAC = "ENABLE_RANSAC"
    FILTER_MODE = "FILTER_MODE"
    RANSAC_THRESHOLD = "RANSAC_THRESHOLD"
    RANSAC_MAX_TRIALS = "RANSAC_MAX_TRIALS"

    # [3] Global Modeling (Auto-ML)
    SELECTED_ALGOS = "SELECTED_ALGOS"
    OPTIMIZER_METHOD = "OPTIMIZER_METHOD"
    COLLISION_HANDLING = "COLLISION_HANDLING"
    N_ITERATIONS = "N_ITERATIONS"
    MEDIAN_SIZE = "MEDIAN_SIZE"

    PARAM_RF = "PARAM_RF"
    PARAM_GB = "PARAM_GB"
    PARAM_ET = "PARAM_ET"
    PARAM_SVR = "PARAM_SVR"
    PARAM_MLP = "PARAM_MLP"

    # [4] Adaptive Refinement
    ENABLE_ADAPTIVE = "ENABLE_ADAPTIVE"
    INPUT_ADAPTIVE_TRAIN = "INPUT_ADAPTIVE_TRAIN"
    FIELD_ADAPTIVE_DEPTH = "FIELD_ADAPTIVE_DEPTH"

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
    ]
    OPTIMIZER_LIST_NAMES = ["Random Search", "Grid Search", "Bayesian Search"]
    COLLISION_LIST_NAMES = [
        "Keep All Points",
        "Highest Confidence",
        "Closest to Pixel Center",
        "Hybrid",
        "Strict Center",
    ]
    MASK_METHODS_NAMES = ["Otsu (Automatic NDWI)", "Manual NDWI Threshold"]
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
        # [1] Phase 1: Pre-processing (Bands & Sunglint)
        # -------------------------------------------------------------------
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
                "📡 [1.1] NIR Band",
                parentLayerParameterName=self.INPUT_RASTER,
                defaultValue=8,
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
            QgsProcessingParameterBand(
                self.NIR_BAND_SUNGLINT,
                "☀️ [1.2] Sunglint NIR Band",
                parentLayerParameterName=self.INPUT_RASTER,
                defaultValue=8,
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
        # [1] Phase 1: Masking & Features
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
                defaultValue=False,
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
        # [2] Phase 2: Training Data & Filtering
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
        # [3] Phase 3: Global Modeling (Auto-ML)
        # -------------------------------------------------------------------
        self.addParameter(
            QgsProcessingParameterEnum(
                self.SELECTED_ALGOS,
                "🤖 [3] Algorithms to Benchmark",
                options=self.MODEL_LIST_NAMES,
                allowMultiple=True,
                defaultValue=[0, 1, 2, 3],
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
                defaultValue=10,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MEDIAN_SIZE,
                "🤖 [3] Output Median Filter Size",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=5,
            )
        )

        # Hyperparameters
        self.addParameter(
            QgsProcessingParameterString(
                self.PARAM_RF,
                "🎛️ [3] Random Forest Params",
                defaultValue="'n_estimators':[100, 300], 'max_depth':[10, 30]",
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.PARAM_GB,
                "🎛️ [3] Gradient Boosting Params",
                defaultValue="'n_estimators':[100, 300], 'learning_rate':[0.05, 0.1]",
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.PARAM_ET,
                "🎛️ [3] Extra Trees Params",
                defaultValue="'n_estimators':[100, 300], 'max_depth':[10, 30]",
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.PARAM_SVR,
                "🎛️ [3] SVR Params",
                defaultValue="'C':[1, 10, 100], 'kernel':['rbf']",
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.PARAM_MLP,
                "🎛️ [3] MLP Params",
                defaultValue="'hidden_layer_sizes':[(100,), (50, 50)]",
            )
        )

        # -------------------------------------------------------------------
        # [4] Phase 4: Adaptive Refinement
        # -------------------------------------------------------------------
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_ADAPTIVE,
                "🎯 [4] Enable Adaptive Refinement",
                defaultValue=True,
            )
        )
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
        # [5] Phase 5: Validation & Output Cleanup
        # -------------------------------------------------------------------
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_VALIDATION, "📉 [5] Enable Validation", defaultValue=True
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
