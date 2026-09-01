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
    ENABLE_MAX_DEPTH_FILTER = "ENABLE_MAX_DEPTH_FILTER"
    MAX_DEPTH_THRESHOLD = "MAX_DEPTH_THRESHOLD"

    ENABLE_RANSAC = "ENABLE_RANSAC"
    FILTER_MODE = "FILTER_MODE"
    FILTER_NUMERATOR_BAND = "FILTER_NUMERATOR_BAND"
    FILTER_DENOMINATOR_BAND = "FILTER_DENOMINATOR_BAND"
    RANSAC_THRESHOLD = "RANSAC_THRESHOLD"
    RANSAC_MAX_TRIALS = "RANSAC_MAX_TRIALS"

    # [3] Global Modeling (Auto-ML)
    ENABLE_MODELING = "ENABLE_MODELING"
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
    ENABLE_DEPTH_VARIANCE_CORR = "ENABLE_DEPTH_VARIANCE_CORR"
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

    SCORE_SELECTION_STRATEGY = "SCORE_SELECTION_STRATEGY"
    SCORE_METRICS = "SCORE_METRICS"
    SCORE_CUSTOM_CONFIG = "SCORE_CUSTOM_CONFIG"

    SCORE_STRATEGY_OPTIONS = [
        "Winner Stability (Monte Carlo Sensitivity Analysis) [Default]",
        "Highest SDB Composite Score (Max Baseline Score 0-100)",
        "Highest R² Accuracy",
        "Lowest RMSE (Minimum Vertical Error)",
        "Lowest wMAPE (%)",
        "Lowest |Bias| (Zero-Mean Residual Offset)",
        "Lowest MAE (Mean Absolute Error)",
    ]

    SCORE_METRIC_OPTIONS = [
        "R² Accuracy (Correlation & Explained Variance)",
        "RMSE (Root Mean Squared Vertical Error)",
        "wMAPE (Weighted Mean Absolute Percentage Error)",
        "|Bias| (Zero-Mean Residual Shift Offset)",
        "MAE (Mean Absolute Error)",
    ]

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
    ENABLE_DEPTH_VARIANCE_CORR_P4 = "ENABLE_DEPTH_VARIANCE_CORR_P4"
    ENABLE_SPATIAL_RESIDUAL_CORR_P4 = "ENABLE_SPATIAL_RESIDUAL_CORR_P4"

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
        "MLP (Neural Net)",
        "SVR",
        "Huber Regressor",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "Ensemble (Average)",
        "Ensemble (Median)",
        "Ensemble (Stacking)",
        "Ensemble (Uncertainty-Weighted Fusion)",
    ]
    OPTIMIZER_LIST_NAMES = ["Random Search", "Grid Search", "Bayesian Search"]
    COLLISION_LIST_NAMES = [
        "Keep All Points",
        "Highest Confidence",
        "Closest to Pixel Center",
        "Hybrid",
        "Strict Center",
    ]
    MASK_METHODS_NAMES = [
        "Otsu (Automatic NDWI)",
        "Manual NDWI Threshold",
        "3 Indices Equation (NDWI, MNDWI, NWI)",
        "Smart Hybrid (Dynamic Auto)",
    ]
    OSW_METHODS_NAMES = [
        "Automated Knee-Point Extinction",
        "Turbidity-Invariant Log-Ratio Extinction",
        "Multi-Otsu / GMM Spectral Clustering [Recommended]",
        "Automatic (NIR Percentile Fallback)",
        "Manual Polygon ROI",
        "Shallow Water Bound (OSW Polygon)",
    ]
    FEATURE_CORR_THRESHOLDS = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    FEATURE_CORR_THRESHOLDS_P4 = ["Use Phase 03 (-1.0)", "0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    FEATURE_OPTIONS_NAMES = [
        "[All Raw Bands] All Bands from Input Image (Raw Reflectance / DN)",
        "[All Log Bands] All Bands from Input Image (Log-Transformed)",
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
        return "1.1 SDB Single-Scene Masterflow"

    def group(self):
        return "1. End-to-End Masterflows"

    def groupId(self):
        return "masterflows"

    def createInstance(self):
        return SDBMasterOrchestrator()

    def shortHelpString(self):
        return """
        <div style="font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.5; color: #2C3E50;">
            <h2 style="margin-bottom: 5px; color: #2E86C1;">🛰️ Bathymetrix-AI: Single Masterflow</h2>
            <p style="margin-top: 0; margin-bottom: 15px; font-size: 13px;">
                The standard end-to-end <b>Auto-ML Pipeline</b> for Satellite-Derived Bathymetry (SDB) from a single satellite scene. 
                Seamlessly connects 5 scientific phases: atmospheric pre-processing, altimetry outlier rejection, global machine learning benchmarking, 
                localized spatial error compensation, and independent scientific validation.
            </p>

            <h3 style="color: #D35400; margin-bottom: 5px; border-bottom: 2px solid #D35400; padding-bottom: 3px;">⚙️ 5-Phase Scientific Methodology</h3>
            <ul style="font-size: 12px; margin-top: 5px; padding-left: 20px;">
                <li><b>Phase 01 — Advanced Pre-processing:</b> Sun-glint removal <i>(Hedley et al., 2005)</i> with robust NaN/Inf handling, water masking (NDWI, MNDWI, NWI with edge shrink), Optically Shallow Water (OSW) deep-water filtering (automatic NIR percentile / dynamic Elbow Point detection / polygon mask & vector export), and flexible Feature Extraction (<code>[All Raw Bands]</code> and/or <code>[All Log Bands]</code> and Stumpf Log-Ratios). <i>(⚠️ Note: If input imagery is already log-transformed, select [All Raw Bands] and avoid [All Log Bands] to prevent double log-transformation).</i></li>
                <li><b>Phase 02 — Robust Altimetry Filtering:</b> Outlier rejection on ICESat-2 (ATL24) LiDAR / sonar training data using <b>Linear RANSAC</b>, <b>LS Variance Fit</b>, or <b>Huber Variance Fit</b> with dynamic percentile diagnostic plots.</li>
                <li><b>Phase 03 — Global Auto-ML & Feature Analysis:</b> Multicollinearity analysis (Pearson / Spearman, Auto-RANSAC, Auto-Random Forest), benchmarks <b>15+ ML models</b> (RF, XGBoost, LightGBM, CatBoost, SVR, MLP, Extra Trees), hyperparameter tuning (Bayesian, Random, Grid), Spatial Block CV, and Ensemble Blending (Average, Median, Stacking, Uncertainty-Weighted Pixel Fusion).</li>
                <li><b>Phase 04 — Spatial Residual Correction:</b> Zero-Mean Centered Spatial Residual modeling using <b>Leave-One-Out (LOO) Robust Huber Weighting</b> and Smoothed IDW (1 / (d + 1.0)) to eliminate local bias drift, secondary stacked retraining, and generates a <b>95% Confidence Spatial Uncertainty Raster Map</b>.</li>
                <li><b>Phase 05 — Validation & IHO S-44 Compliance:</b> Evaluates accuracy against unseen validation data (RMSE, R², MAE, Bias, wMAPE), depth-stratified zoning (0–5m, 5–10m, etc.), <b>IHO Order 1a/2 Total Vertical Uncertainty (TVU)</b> compliance check, and automated interactive HTML dashboard generation.</li>
            </ul>

            <h3 style="color: #117A65; margin-top: 15px; margin-bottom: 5px; border-bottom: 2px solid #117A65; padding-bottom: 3px;">🧽 Output Cleanup & Export</h3>
            <p style="font-size: 12px; margin-top: 5px;">
                Includes automated removal of positive (land) pixels, physical slope spike filtering, and flexible precision formats (Float32, Float64, Int16).
            </p>

            <h3 style="color: #8E44AD; margin-top: 15px; margin-bottom: 5px; border-bottom: 2px solid #8E44AD; padding-bottom: 3px;">📚 Key References</h3>
            <ul style="font-size: 11px; margin-top: 5px; padding-left: 20px; color: #555;">
                <li><b>Stumpf et al. (2003):</b> Determination of shallow water bathymetry with high-resolution satellite imagery.</li>
                <li><b>Hedley et al. (2005):</b> Simple and robust removal of sun glint for high-resolution imagery.</li>
                <li><b>Fischler & Bolles (1981):</b> Random Sample Consensus (RANSAC).</li>
                <li><b>Alevizos (2020):</b> Spatial residual refinement and error compensation in shallow coastal waters.</li>
            </ul>
            <br><b>Developer:</b> Mohamed Aly Nasef
        </div>
        """

    def helpString(self):
        return self.shortHelpString()

    # =======================================================================
    # 4. ALGORITHM INIT (QGIS FRONT-END UI SETUP)
    # =======================================================================
    def initAlgorithm(self, config=None):

        # ===================================================================
        # [0] General Settings
        # ===================================================================
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

        # ===================================================================
        # [1] Phase 01: Advanced Pre-processing
        # ===================================================================
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_PREPROCESSING,
                "━━━━━━━━━ ⚙️ [1] Phase 01: Pre-processing & Feature Extraction ━━━━━━━━━",
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
            QgsProcessingParameterVectorLayer(
                self.WATER_MASK_POLY,
                "🗺️ [1.3] Ready-made Water Mask Polygon",
                types=[QgsProcessing.TypeVectorPolygon],
                optional=True,
            )
        )

        default_feats = list(range(1, len(self.FEATURE_OPTIONS_NAMES)))
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
            QgsProcessingParameterString(
                self.BAND_MATH_FORMULA,
                "🧮 [1.4] Custom Band Math Formula (e.g. (B2-B3)/(B2+B3) or Log_B2/Log_B3)",
                defaultValue="",
                optional=True,
            )
        )

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
                defaultValue=2,
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

        # ===================================================================
        # [2] Phase 02: Robust Filtering
        # ===================================================================
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_RANSAC,
                "━━━━━━━━━ 🛡️ [2] Enable Phase 02 Data Filtering ━━━━━━━━━",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_TRAIN, "📍 [2.1] Main Training Points (ICESat-2 / In-situ)"
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_DEPTH,
                "📏 [2.1] Depth Field",
                defaultValue="ortho_h",
                parentLayerParameterName=self.INPUT_TRAIN,
                type=QgsProcessingParameterField.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_WEIGHT,
                "⚖️ [2.1] Weight Field [optional]",
                defaultValue="confidence",
                parentLayerParameterName=self.INPUT_TRAIN,
                type=QgsProcessingParameterField.Numeric,
                optional=True,
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

        # ===================================================================
        # [3] Phase 03: Global Auto-ML & Feature Analysis
        # ===================================================================
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_MODELING,
                "━━━━━━━━━ 🤖 [3] Phase 03: Global Auto-ML SDB Modeling ━━━━━━━━━",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.SELECTED_ALGOS,
                "🤖 [3.1] Algorithms to Benchmark",
                options=self.MODEL_LIST_NAMES,
                allowMultiple=True,
                defaultValue=[3, 12, 13, 14, 15, 17],
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.OPTIMIZER_METHOD,
                "🤖 [3.2] Optimizer Method",
                options=self.OPTIMIZER_LIST_NAMES,
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.FEATURE_CORR_METHOD,
                "🤖 [3.3] Feature Correlation Method",
                options=["Disabled", "Pearson (Linear)", "Spearman (Rank)", "Automatic-RANSAC", "Automatic-Random Forest"],
                defaultValue=3,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.FEATURE_CORR_THRESHOLD,
                "🤖 [3.3] Feature Correlation Threshold",
                options=self.FEATURE_CORR_THRESHOLDS,
                defaultValue=2,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.SPATIAL_CV_P3,
                "🌍 [3.4] Enable Spatial Block Cross-Validation",
                defaultValue=True,
            )
        )

        # ===================================================================
        # [4] Phase 04: Adaptive Refinement
        # ===================================================================
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_ADAPTIVE,
                "━━━━━━━━━ 🎯 [4] Phase 04: Adaptive Refinement ━━━━━━━━━",
                defaultValue=False,
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
                defaultValue="ortho_h",
                parentLayerParameterName=self.INPUT_ADAPTIVE_TRAIN,
                type=QgsProcessingParameterField.Numeric,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_DEPTH_VARIANCE_CORR_P4,
                "🎛️ [4] Enable Depth Variance Correction (Datum Mean Shift)",
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_SPATIAL_RESIDUAL_CORR_P4,
                "📍 [4] Enable Spatial Residual Correction (KNN / Kriging Grid)",
                defaultValue=True,
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
        self.addParameter(
            QgsProcessingParameterEnum(
                self.STACK_COMPONENTS_P4,
                "🎯 [4] Features for Retraining",
                options=["Feature Stack (Phase 01)", "Phase 03 Depth Map", "Residual Error Grid"],
                allowMultiple=True,
                defaultValue=[1, 2],
            )
        )

        # ===================================================================
        # [5] Phase 05: Validation & Reporting
        # ===================================================================
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_VALIDATION, "━━━━━━━━━ 📉 [5] Phase 05: Validation & Reporting ━━━━━━━━━", defaultValue=False
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

        # ===================================================================
        # ➕ ADVANCED PARAMETERS (Organized cleanly by phase)
        # ===================================================================

        # --- [System] ---
        p_threads = QgsProcessingParameterNumber(
            self.NUM_THREADS,
            "⚙️ [System] Processing Threads (Multi-processing)",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=4,
        )
        p_threads.setFlags(p_threads.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_threads)

        # --- [Phase 01] ---
        p_sg_pct = QgsProcessingParameterNumber(
            self.SUNGLINT_PERCENTILE,
            "☀️ [Phase 01] Sunglint Deep Water %",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=1.0,
        )
        p_sg_pct.setFlags(p_sg_pct.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_sg_pct)

        p_shrink = QgsProcessingParameterNumber(
            self.SHRINK_EDGE_DIST,
            "🗺️ [Phase 01] Water Edge Shrink (Map Units, e.g. -10)",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.0,
        )
        p_shrink.setFlags(p_shrink.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_shrink)

        p_man_th = QgsProcessingParameterNumber(
            self.MANUAL_THRESHOLD,
            "🏖️ [Phase 01] Manual Water Mask Threshold",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.0,
            optional=True,
        )
        p_man_th.setFlags(p_man_th.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_man_th)

        p_otsu = QgsProcessingParameterNumber(
            self.OTSU_ADJUSTMENT,
            "🏖️ [Phase 01] Otsu Threshold Adjustment",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.0,
        )
        p_otsu.setFlags(p_otsu.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_otsu)

        p_mask_k = QgsProcessingParameterNumber(
            self.MASK_KERNEL_SIZE,
            "🏖️ [Phase 01] Mask Cleanup Kernel Size",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=3,
        )
        p_mask_k.setFlags(p_mask_k.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_mask_k)

        p_nir_pct = QgsProcessingParameterNumber(
            self.NIR_PERCENTILE_OSW,
            "🌊 [Phase 01] NIR Percentile for Deep Water (e.g. 10%)",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=10.0,
        )
        p_nir_pct.setFlags(p_nir_pct.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_nir_pct)

        p_osw_med = QgsProcessingParameterNumber(
            self.OSW_MEDIAN_SIZE,
            "🌊 [Phase 01] OSW Mask Median Filter Size",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=3,
        )
        p_osw_med.setFlags(p_osw_med.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_osw_med)

        p_fill = QgsProcessingParameterBoolean(
            self.FILL_INTERNAL_HOLES,
            "🌊 [Phase 01] Fill All Internal Holes in OSW Mask",
            defaultValue=True,
        )
        p_fill.setFlags(p_fill.flags() | QgsProcessingParameterDefinition.FlagHidden)
        self.addParameter(p_fill)

        p_extract = QgsProcessingParameterBoolean(
            self.EXTRACT_POLYGON,
            "🌊 [Phase 01] Extract OSW Mask as Polygon",
            defaultValue=True,
        )
        p_extract.setFlags(p_extract.flags() | QgsProcessingParameterDefinition.FlagHidden)
        self.addParameter(p_extract)

        # --- [Phase 02] ---
        p_r_th = QgsProcessingParameterNumber(
            self.RANSAC_THRESHOLD,
            "🧹 [Phase 02] RANSAC Threshold / Sigma Multiplier",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=3.0,
        )
        p_r_th.setFlags(p_r_th.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_r_th)

        p_r_tr = QgsProcessingParameterNumber(
            self.RANSAC_MAX_TRIALS,
            "🧹 [Phase 02] RANSAC Maximum Trials",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=100,
        )
        p_r_tr.setFlags(p_r_tr.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_r_tr)

        p_en_max_d = QgsProcessingParameterBoolean(
            self.ENABLE_MAX_DEPTH_FILTER,
            "🛑 [Phase 02] Enable Maximum Depth Cutoff Filter",
            defaultValue=False,
        )
        p_en_max_d.setFlags(p_en_max_d.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_en_max_d)

        p_max_d = QgsProcessingParameterNumber(
            self.MAX_DEPTH_THRESHOLD,
            "🛑 [Phase 02] Maximum Depth Cutoff Threshold (e.g. -30.0)",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=-30.0,
        )
        p_max_d.setFlags(p_max_d.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_max_d)

        # --- [Phase 03] ---
        p_col = QgsProcessingParameterEnum(
            self.COLLISION_HANDLING,
            "🤖 [Phase 03] In-Situ Points Collision Handling",
            options=self.COLLISION_LIST_NAMES,
            defaultValue=0,
        )
        p_col.setFlags(p_col.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_col)

        p_n_iter = QgsProcessingParameterNumber(
            self.N_ITERATIONS,
            "🤖 [Phase 03] Hyperparameter Optimization Iterations",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=20,
        )
        p_n_iter.setFlags(p_n_iter.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_n_iter)

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

        p_cv = QgsProcessingParameterNumber(
            self.CV_FOLDS, "🎛️ [Phase 03] ML Cross-Validation Folds", type=QgsProcessingParameterNumber.Integer, defaultValue=5
        )
        p_cv.setFlags(p_cv.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_cv)

        p_uncert = QgsProcessingParameterNumber(
            self.UNCERT_TREES, "🎛️ [Phase 03] Uncertainty Model Estimators (Trees)", type=QgsProcessingParameterNumber.Integer, defaultValue=200
        )
        p_uncert.setFlags(p_uncert.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_uncert)

        p_split = QgsProcessingParameterNumber(
            self.TRAIN_TEST_SPLIT, "🎛️ [Phase 03 & 04] Training Data Ratio (e.g., 0.8 for 80%)", type=QgsProcessingParameterNumber.Double, defaultValue=0.8
        )
        p_split.setFlags(p_split.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_split)

        p_var_corr = QgsProcessingParameterBoolean(
            self.ENABLE_DEPTH_VARIANCE_CORR,
            "🤖 [Phase 03] Enable Depth Variance Correction",
            defaultValue=False,
        )
        p_var_corr.setFlags(p_var_corr.flags() | QgsProcessingParameterDefinition.FlagAdvanced | QgsProcessingParameterDefinition.FlagHidden)
        self.addParameter(p_var_corr)

        # --- [Phase 04] ---
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
            self.MAX_GPR_SAMPLES, "📍 [Phase 04] Max GPR Training Samples", type=QgsProcessingParameterNumber.Integer, defaultValue=1500
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

        p_corr_m_p4 = QgsProcessingParameterEnum(
            self.FEATURE_CORR_METHOD_P4,
            "🤖 [Phase 04] Feature Correlation Method",
            options=["Disabled", "Pearson (Linear)", "Spearman (Rank)", "Automatic-RANSAC", "Automatic-Random Forest"],
            defaultValue=3,
        )
        p_corr_m_p4.setFlags(p_corr_m_p4.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_corr_m_p4)

        p_corr_th_p4 = QgsProcessingParameterEnum(
            self.FEATURE_CORR_THRESHOLD_P4,
            "🤖 [Phase 04] Feature Correlation Threshold",
            options=self.FEATURE_CORR_THRESHOLDS_P4,
            defaultValue=0,
        )
        p_corr_th_p4.setFlags(p_corr_th_p4.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_corr_th_p4)

        # --- [Post-Processing Cleanup & Filtering] ---
        p_med = QgsProcessingParameterNumber(
            self.MEDIAN_SIZE,
            "🧽 [Phase 03 & 04] Output Depth Median Filter Size",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=5,
        )
        p_med.setFlags(p_med.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_med)

        p_rem_pos = QgsProcessingParameterBoolean(
            self.REMOVE_POSITIVES,
            "🧽 [Post-Processing] Remove Positive Depths (>= 0)",
            defaultValue=True,
        )
        p_rem_pos.setFlags(p_rem_pos.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_rem_pos)

        p_slope_f = QgsProcessingParameterBoolean(
            self.ENABLE_SLOPE_FILTER,
            "🧽 [Post-Processing] Apply Physical Slope Filter",
            defaultValue=True,
        )
        p_slope_f.setFlags(p_slope_f.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_slope_f)

        p_slope_th = QgsProcessingParameterNumber(
            self.SLOPE_THRESHOLD,
            "🧽 [Post-Processing] Slope Filter Threshold (Degrees)",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=35.0,
        )
        p_slope_th.setFlags(p_slope_th.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_slope_th)

        # --- [General System & Ranking] ---
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
            defaultValue=[0, 1, 2, 3, 4],
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

        # --- Hyperparameters Search Grids ---
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


    def processAlgorithm(self, parameters, context, feedback):
        from ...core.pipeline import run_master_pipeline

        return run_master_pipeline(self, parameters, context, feedback)
