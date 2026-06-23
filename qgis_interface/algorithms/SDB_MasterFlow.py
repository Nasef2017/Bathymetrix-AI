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
    SWIR_BAND = "SWIR_BAND"

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

    TRAIN_TEST_SPLIT = "TRAIN_TEST_SPLIT"
    RANDOM_STATE = "RANDOM_STATE"
    OUTPUT_FORMAT = "OUTPUT_FORMAT"

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
    MASK_METHODS_NAMES = ["Otsu (Automatic NDWI)", "Manual NDWI Threshold", "3 Indices Equation (NDWI, MNDWI, NWI)"]
    OSW_METHODS_NAMES = ["Manual Polygon ROI", "Automatic (Lowest NIR Percentile)"]
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
                <li><b>Advanced Water Masking</b> using 3-Indices (NDWI, MNDWI, NWI).</li>
                <li>Physics-based Log-Ratio features computation.</li>
                <li><b>Deep Water Filter</b> (OSW Mask), customized for ML algorithms.</li>
            </ul>

            <b style="display: block; margin-bottom: 2px;">🎯 Phase 02: Robust Filtering</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li>Noise removal using <b>Linear RANSAC</b>, <b>LS Variance Fit</b>, or <b>Huber Variance Fit</b> <i>(Zhang et al., 2021)</i>.</li>
            </ul>

            <b style="display: block; margin-bottom: 2px;">🤖 Phase 03: Global Auto-ML & Feature Analysis</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li><b>Feature Analysis:</b> Optionally drop weak bands based on their Pearson or Spearman correlation with the target depth.</li>
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
                defaultValue=20,
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
        self.addParameter(
            QgsProcessingParameterEnum(
                self.FEATURE_CORR_METHOD,
                "🤖 [3] Feature Correlation Method",
                options=["Pearson (Linear)", "Spearman (Rank)"],
                defaultValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.FEATURE_CORR_THRESHOLD,
                "🤖 [3] Feature Correlation Threshold",
                options=['0.0 (Disable)', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],
                defaultValue=2,
            )
        )

        # Hyperparameters and ML Settings
        p_split = QgsProcessingParameterNumber(
            self.TRAIN_TEST_SPLIT, "🎛️ [3] Training Data Ratio (e.g., 0.8 for 80%)", type=QgsProcessingParameterNumber.Double, defaultValue=0.8
        )
        p_split.setFlags(p_split.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_split)

        p_rs = QgsProcessingParameterNumber(
            self.RANDOM_STATE, "🎛️ [3] Random State", type=QgsProcessingParameterNumber.Integer, defaultValue=42
        )
        p_rs.setFlags(p_rs.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_rs)

        p_fmt = QgsProcessingParameterEnum(
            self.OUTPUT_FORMAT, "🎛️ [3] Output Format", options=["float32", "float64", "uint16"], defaultValue=0
        )
        p_fmt.setFlags(p_fmt.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_fmt)

        p_rf = QgsProcessingParameterString(self.PARAM_RF, "🎛️ [3] Random Forest Params", defaultValue="'n_estimators':[100, 300], 'max_depth':[10, 30]")
        p_rf.setFlags(p_rf.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_rf)

        p_gb = QgsProcessingParameterString(self.PARAM_GB, "🎛️ [3] Gradient Boosting Params", defaultValue="'n_estimators':[100, 300], 'learning_rate':[0.05, 0.1]")
        p_gb.setFlags(p_gb.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_gb)

        p_et = QgsProcessingParameterString(self.PARAM_ET, "🎛️ [3] Extra Trees Params", defaultValue="'n_estimators':[100, 300], 'max_depth':[10, 30]")
        p_et.setFlags(p_et.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_et)

        p_svr = QgsProcessingParameterString(self.PARAM_SVR, "🎛️ [3] SVR Params", defaultValue="'C':[1, 10, 100], 'kernel':['rbf'], 'cache_size':[1000], 'max_iter':[20000]")
        p_svr.setFlags(p_svr.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_svr)

        p_mlp = QgsProcessingParameterString(self.PARAM_MLP, "🎛️ [3] MLP Params", defaultValue="'hidden_layer_sizes':[(100,), (50, 50)], 'max_iter':[500]")
        p_mlp.setFlags(p_mlp.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_mlp)

        p_ridge = QgsProcessingParameterString(self.PARAM_RIDGE, "🎛️ [3] Ridge Params", defaultValue="'alpha':[0.1, 1.0]", optional=True)
        p_ridge.setFlags(p_ridge.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_ridge)

        p_lasso = QgsProcessingParameterString(self.PARAM_LASSO, "🎛️ [3] Lasso Params", defaultValue="'alpha':[0.01, 0.1]", optional=True)
        p_lasso.setFlags(p_lasso.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_lasso)

        p_en = QgsProcessingParameterString(self.PARAM_ELASTICNET, "🎛️ [3] ElasticNet Params", defaultValue="'l1_ratio':[0.5]", optional=True)
        p_en.setFlags(p_en.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_en)

        p_knn = QgsProcessingParameterString(self.PARAM_KNN, "🎛️ [3] KNN Params", defaultValue="'n_neighbors':[5, 10]", optional=True)
        p_knn.setFlags(p_knn.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_knn)

        p_dt = QgsProcessingParameterString(self.PARAM_DT, "🎛️ [3] Decision Tree Params", defaultValue="'max_depth':[5, 10]", optional=True)
        p_dt.setFlags(p_dt.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_dt)

        # -------------------------------------------------------------------
        # [4] Phase 04: Adaptive Refinement
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
        # [5] Phase 05: Validation & Reporting
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
