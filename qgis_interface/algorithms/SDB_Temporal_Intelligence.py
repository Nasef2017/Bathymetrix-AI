import os
import warnings
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
    QgsProcessingParameterBand,
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterString,
    QgsProcessingParameterDefinition,
    QgsProcessingException,
)

try:
    from ...core.temporal.data_scanner import TemporalDataScanner
    from ...core.temporal.temporal_sdb_runner import TemporalSDBRunner
    from ...core.temporal.shoreline_tracker import ShorelineDynamicsTracker
    from ...core.temporal.temporal_analytics import TemporalAnalyticsEngine
    from ...core.temporal.temporal_reporting import TemporalReportGenerator
except ImportError:
    from Bathymetrix_AI.core.temporal.data_scanner import TemporalDataScanner
    from Bathymetrix_AI.core.temporal.temporal_sdb_runner import TemporalSDBRunner
    from Bathymetrix_AI.core.temporal.shoreline_tracker import ShorelineDynamicsTracker
    from Bathymetrix_AI.core.temporal.temporal_analytics import TemporalAnalyticsEngine
    from Bathymetrix_AI.core.temporal.temporal_reporting import TemporalReportGenerator

warnings.filterwarnings("ignore")


class SDBTemporalIntelligence(QgsProcessingAlgorithm):
    """
    Bathymetrix-AI Temporal Intelligence Master Processing Algorithm
    Full MasterFlow capabilities with pure English UI, Advanced Parameters section, and clear help text.
    """

    # [0] General Settings & I/O
    INPUT_IMAGE_ROOT = "INPUT_IMAGE_ROOT"
    OUTPUT_MASTER_FOLDER = "OUTPUT_MASTER_FOLDER"
    NUM_THREADS = "NUM_THREADS"

    # [1] Phase 01: Pre-processing (Bands, Sunglint, Masking, Features, OSW)
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
    SHORELINE_ROI = "SHORELINE_ROI"
    SHRINK_EDGE_DIST = "SHRINK_EDGE_DIST"
    ENABLE_MASKING = "ENABLE_MASKING"
    MASKING_METHOD = "MASKING_METHOD"
    MANUAL_THRESHOLD = "MANUAL_THRESHOLD"
    OTSU_ADJUSTMENT = "OTSU_ADJUSTMENT"
    MASK_KERNEL_SIZE = "MASK_KERNEL_SIZE"

    FEATURE_SELECTION = "FEATURE_SELECTION"
    ENABLE_BAND_CALC = "ENABLE_BAND_CALC"
    BAND_MATH_FORMULA = "BAND_MATH_FORMULA"

    APPLY_DEEPWATER = "APPLY_DEEPWATER"
    DEEPWATER_METHOD = "DEEPWATER_METHOD"
    DEEPWATER_ROI = "DEEPWATER_ROI"
    NIR_PERCENTILE_OSW = "NIR_PERCENTILE_OSW"
    OSW_MEDIAN_SIZE = "OSW_MEDIAN_SIZE"

    # [2] Phase 02: Filtering & Training Data
    INPUT_TRAIN = "INPUT_TRAIN"
    FIELD_DEPTH = "FIELD_DEPTH"
    FIELD_WEIGHT = "FIELD_WEIGHT"
    FIELD_YEAR_ICESAT = "FIELD_YEAR_ICESAT"
    MAX_DEPTH_THRESHOLD = "MAX_DEPTH_THRESHOLD"

    ENABLE_RANSAC = "ENABLE_RANSAC"
    FILTER_MODE = "FILTER_MODE"
    FILTER_NUMERATOR_BAND = "FILTER_NUMERATOR_BAND"
    FILTER_DENOMINATOR_BAND = "FILTER_DENOMINATOR_BAND"
    RANSAC_THRESHOLD = "RANSAC_THRESHOLD"
    RANSAC_MAX_TRIALS = "RANSAC_MAX_TRIALS"

    # [3] & [4] Phase 03 & 04: Global Auto-ML & Adaptive Refinement

    SELECTED_ALGOS = "SELECTED_ALGOS"
    OPTIMIZER_METHOD = "OPTIMIZER_METHOD"
    COLLISION_HANDLING = "COLLISION_HANDLING"
    N_ITERATIONS = "N_ITERATIONS"
    MEDIAN_SIZE = "MEDIAN_SIZE"
    FEATURE_CORR_METHOD = "FEATURE_CORR_METHOD"
    FEATURE_CORR_THRESHOLD = "FEATURE_CORR_THRESHOLD"

    ENABLE_ENSEMBLE = "ENABLE_ENSEMBLE"
    ENSEMBLE_METHOD = "ENSEMBLE_METHOD"
    ENSEMBLE_SIZE = "ENSEMBLE_SIZE"

    ENABLE_ADAPTIVE = "ENABLE_ADAPTIVE"
    RESIDUAL_INTERP_METHOD = "RESIDUAL_INTERP_METHOD"
    KNN_NEIGHBORS = "KNN_NEIGHBORS"

    # Advanced Hyperparameters & Settings
    TRAIN_TEST_SPLIT = "TRAIN_TEST_SPLIT"
    RANDOM_STATE = "RANDOM_STATE"
    CV_FOLDS = "CV_FOLDS"
    UNCERT_TREES = "UNCERT_TREES"
    MAX_GPR_SAMPLES = "MAX_GPR_SAMPLES"
    OUTPUT_FORMAT = "OUTPUT_FORMAT"
    SPATIAL_CV_P3 = "SPATIAL_CV_P3"

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

    ENABLE_ENSEMBLE_P4 = "ENABLE_ENSEMBLE_P4"
    ENSEMBLE_METHOD_P4 = "ENSEMBLE_METHOD_P4"
    ENSEMBLE_SIZE_P4 = "ENSEMBLE_SIZE_P4"
    STACK_COMPONENTS_P4 = "STACK_COMPONENTS_P4"
    FEATURE_CORR_METHOD_P4 = "FEATURE_CORR_METHOD_P4"
    FEATURE_CORR_THRESHOLD_P4 = "FEATURE_CORR_THRESHOLD_P4"
    SPATIAL_CV_P4 = "SPATIAL_CV_P4"

    # [5] Phase 05: Validation, Post-processing & Temporal Products
    ENABLE_SLOPE_FILTER = "ENABLE_SLOPE_FILTER"
    SLOPE_THRESHOLD = "SLOPE_THRESHOLD"
    REMOVE_POSITIVES = "REMOVE_POSITIVES"
    OVERALL_TREND_METHOD = "OVERALL_TREND_METHOD"
    COMPARISON_MODE = "COMPARISON_MODE"

    ENABLE_DSAS = "ENABLE_DSAS"
    TRANSECT_SPACING = "TRANSECT_SPACING"
    FIELD_ADAPTIVE_DEPTH = "FIELD_ADAPTIVE_DEPTH"
    FIELD_TEST_DEPTH = "FIELD_TEST_DEPTH"

    # Dropdown Lists (Identical to MasterFlow)
    TREND_METHODS_NAMES = ["Long-term Trend", "Net Difference"]
    COMPARISON_MODE_NAMES = ["Sequential (Year-to-Year)", "Baseline Reference (First Year Fixed)"]
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

    def name(self):
        return "sdb_multiyear_dynamics"

    def displayName(self):
        return "Coastal Dynamics Analysis"

    def group(self):
        return ""

    def groupId(self):
        return ""

    def createInstance(self):
        return SDBTemporalIntelligence()

    def shortHelpString(self):
        return """
        <div style="font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.5; color: #2C3E50;">
            <h2 style="margin-bottom: 5px; color: #2E86C1;">🌊 Bathymetrix-AI: Coastal Dynamics Analysis</h2>
            <p style="margin-top: 0; margin-bottom: 15px; font-size: 13px;">
                An advanced engineering tool for autonomous multi-year coastal analysis. It extracts Satellite Derived Bathymetry (SDB), 
                calculates shoreline migration, and computes robust sediment mass balance over time.
            </p>
            
            <h3 style="color: #117A65; margin-bottom: 5px; border-bottom: 2px solid #117A65; padding-bottom: 3px;">📁 Required Workspace Structure</h3>
            <pre style="background: #F8F9F9; padding: 10px; border-left: 4px solid #2E86C1; font-family: Consolas, monospace; font-size: 12px; margin-top: 5px; border-radius: 4px;">
📁 MultiYear_Workspace/
   ├── 📁 2019/
   │      ├── Satellite_Image_2019.tif    (Satellite Image - REQUIRED)
   │      ├── Training_Points_2019.shp    (Depth Points - OPTIONAL if global layer provided)
   │      ├── Control_Points_2019.shp     (Control Points for Phase 04 - OPTIONAL)
   │      └── Validation_2019.shp         (Validation Points for Phase 05 - OPTIONAL)
   │
   └── 📁 2024/
          ├── Satellite_Image_2024.tif          
          ├── Training_Points_2024.shp    
          ├── Control_Points_2024.shp     
          └── Validation_2024.shp         
            </pre>

            <h3 style="color: #D35400; margin-top: 20px; margin-bottom: 5px; border-bottom: 2px solid #D35400; padding-bottom: 3px;">📐 Mathematical Formulations & Scientific Methodology</h3>
            
            <div style="background-color: #E8F8F5; padding: 12px; border-left: 4px solid #117A65; margin-bottom: 15px; border-radius: 4px;">
                <b style="color: #117A65; font-size: 14px;">1. Uncertainty-Aware Morphological Stability Index (MSI)</b><br>
                <p style="margin: 5px 0 8px 0; font-size: 12px;">
                    Quantifies true physical seabed elevation variance across time by subtracting SDB measurement uncertainty variance (σ<sub>U</sub>²):
                </p>
                <div style="background: #FFF; padding: 8px; border: 1px solid #A2D9CE; text-align: center; font-family: 'Courier New', monospace; font-size: 13px; font-weight: bold; color: #16A085;">
                    σ<sub>true</sub>(x, y) = sqrt( max(0, σ<sub>obs</sub>²(x, y) - σ<sub>U</sub>²(x, y)) )<br>
                    MSI(x, y) = max(0, min(1, 1 - [ σ<sub>true</sub>(x, y) / ( |μ<sub>Z</sub>(x, y)| + ε ) ]))
                </div>
                <ul style="font-size: 12px; margin-top: 8px; padding-left: 20px; margin-bottom: 0;">
                    <li><b>σ<sub>obs</sub>(x, y):</b> Total observed temporal standard deviation across N years.</li>
                    <li><b>σ<sub>U</sub>(x, y):</b> Root Mean Square of SDB measurement uncertainties.</li>
                    <li><b>ε = 0.5 m:</b> Shallow-water stabilization constant preventing numerical singularity.</li>
                    <li><b>Interpretation:</b> If observed variance is purely measurement noise (σ<sub>obs</sub> ≤ σ<sub>U</sub>), then σ<sub>true</sub> = 0 → <b>MSI = 1.0</b> (Physically Stable Seabed).</li>
                </ul>
            </div>

            <div style="background-color: #FEF9E7; padding: 12px; border-left: 4px solid #F1C40F; margin-bottom: 15px; border-radius: 4px;">
                <b style="color: #B7950B; font-size: 14px;">2. StatCD & Probabilistic Level of Detection (LoD / DoD)</b><br>
                <p style="margin: 5px 0 8px 0; font-size: 12px;">
                    Calculates statistically significant depth differences between two epoch maps (t₁ and t₂) with spatial coherence opening filter (Wheaton et al. 2010; Lane & Chandler):
                </p>
                <div style="background: #FFF; padding: 8px; border: 1px solid #F9E79F; text-align: center; font-family: 'Courier New', monospace; font-size: 13px; font-weight: bold; color: #D4AC0D;">
                    ΔZ(x, y) = Z<sub>t2</sub>(x, y) - Z<sub>t1</sub>(x, y)<br>
                    σ<sub>ΔZ</sub>(x, y) = sqrt( σ<sub>t1</sub>²(x, y) + σ<sub>t2</sub>²(x, y) - 2·Cov(Z₁, Z₂) ) &nbsp; [Cov = 0 assuming independent epochs]<br>
                    Z<sub>score</sub>(x, y) = ΔZ(x, y) / σ<sub>ΔZ</sub>(x, y)
                </div>
                <p style="margin: 8px 0 5px 0; font-size: 12px;">
                    <b>Statistical Significance & Spatial Coherence (95% Confidence, |Z<sub>score</sub>| > 1.96 + 3x3 MMU Opening):</b>
                </p>
                <ul style="font-size: 12px; margin-top: 2px; padding-left: 20px; margin-bottom: 0;">
                    <li><b>Accretion Mask (M<sub>Acc</sub>):</b> Z<sub>score</sub> < -1.96 (for positive depth maps where seabed elevates/shallows).</li>
                    <li><b>Erosion Mask (M<sub>Ero</sub>):</b> Z<sub>score</sub> > +1.96 (for positive depth maps where seabed deepens).</li>
                    <li><b>MMU Opening:</b> 3×3 spatial coherence filter removes isolated single-pixel noise artifacts.</li>
                </ul>
            </div>

            <div style="background-color: #F5EEF8; padding: 12px; border-left: 4px solid #9B59B6; margin-bottom: 15px; border-radius: 4px;">
                <b style="color: #8E44AD; font-size: 14px;">3. Sediment Volumetric Mass Balance (m³)</b><br>
                <p style="margin: 5px 0 8px 0; font-size: 12px;">
                    Integrates depth change over the statistically significant, spatially coherent pixel area A<sub>pixel</sub> = Δx × Δy (m²):
                </p>
                <div style="background: #FFF; padding: 8px; border: 1px solid #D7BDE2; text-align: center; font-family: 'Courier New', monospace; font-size: 13px; font-weight: bold; color: #7D3C98;">
                    V<sub>Accretion</sub> = ∑<sub>(x,y) ∈ M<sub>Acc</sub></sub> (-ΔZ(x, y)) × A<sub>pixel</sub> &nbsp;&nbsp; [m³]<br>
                    V<sub>Erosion</sub> = ∑<sub>(x,y) ∈ M<sub>Ero</sub></sub> ΔZ(x, y) × A<sub>pixel</sub> &nbsp;&nbsp; [m³]<br>
                    V<sub>Net</sub> = V<sub>Accretion</sub> - V<sub>Erosion</sub> &nbsp;&nbsp; [m³]
                </div>
            </div>

            <h3 style="color: #8E44AD; margin-top: 20px; margin-bottom: 5px; border-bottom: 2px solid #8E44AD; padding-bottom: 3px;">💡 Why Linear Regression is Used for Volumetric Change & MSI?</h3>
            <p style="margin-top: 5px; font-size: 13px;">
                While heavy Machine Learning models (Extra Trees, XGBoost, LightGBM, CatBoost) are recommended for single-date high-precision SDB mapping, <b>Linear Regression</b> is specifically applied for Volumetric Change and Morphological Stability Index (MSI) analytics.<br>
                <b>Scientific Rationale:</b> Linear Regression provides smooth, continuous spatial derivatives across time (&part;Z / &part;t). Unlike non-linear decision trees which construct step-wise decision boundaries that introduce artificial high-frequency noise when subtracting multi-temporal DEMs, Linear Regression cancels out static background biases upon subtraction (&Delta;Z = Z<sub>t2</sub> - Z<sub>t1</sub>), yielding smooth, highly reliable, and physically realistic sediment volumetric change calculations (m³).
            </p>
            
            <h3 style="color: #34495E; margin-top: 20px; margin-bottom: 5px; border-bottom: 2px solid #34495E; padding-bottom: 3px;">📚 Key References</h3>
            <ul style="margin-top: 5px; font-size: 13px; padding-left: 20px;">
                <li><b>Wheaton et al. (2010):</b> Accounting for uncertainty in DEMs from repeat topographic surveys (StatCD / Volumetric significance).</li>
                <li><b>Lane & Chandler (2003):</b> The application of topographic surveying to fluvial studies (MMU spatial coherence filters).</li>
            </ul>
        </div>
        """

    def helpString(self):
        return self.shortHelpString()

    def initAlgorithm(self, config=None):
        # -------------------------------------------------------------------
        # [0] General Input Directory FIRST
        # -------------------------------------------------------------------
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_IMAGE_ROOT,
                "📁 [0] Multi-Year Dataset Root Directory (contains /2020/, /2021/, ...)",
                behavior=QgsProcessingParameterFile.Folder,
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
                self.COASTAL_BAND, "📡 [1.1] Coastal Blue Band", defaultValue=1
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                self.BLUE_BAND, "📡 [1.1] Blue Band", defaultValue=2
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                self.GREEN_BAND, "📡 [1.1] Green Band", defaultValue=3
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                self.RED_BAND, "📡 [1.1] Red Band", defaultValue=4
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                self.NIR_BAND, "🌍 [1.1] NIR Band", defaultValue=5
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                self.SWIR_BAND, "🌍 [1.1] SWIR Band (For 3-Indices Mask)", defaultValue=11
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

        # -------------------------------------------------------------------
        # [2] Phase 02: Filtering & Training Data Setup
        # -------------------------------------------------------------------
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_TRAIN,
                "📍 [2.1] [Optional] Global Training Points / ICESat-2 Vector Layer (Leave blank if training points are inside year subfolders)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_DEPTH,
                "📏 [2.1] Depth Field Name",
                parentLayerParameterName=self.INPUT_TRAIN,
                defaultValue="h_mean",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_WEIGHT,
                "⚖️ [2.1] Weight Field Name",
                parentLayerParameterName=self.INPUT_TRAIN,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_YEAR_ICESAT,
                "📅 [2.1] [Optional] Year Field (leave blank if 1 file per year)",
                parentLayerParameterName=self.INPUT_TRAIN,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.FIELD_ADAPTIVE_DEPTH,
                "🎯 [Optional] Control Points Depth Field Name (e.g. field_3, ortho_h)",
                defaultValue="field_3",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.FIELD_TEST_DEPTH,
                "✅ [Optional] Validation Points Depth Field Name (e.g. field_3, ortho_h)",
                defaultValue="field_3",
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
                defaultValue=2,
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                self.FILTER_DENOMINATOR_BAND,
                "🧹 [2.2] Log-Ratio Denominator Band",
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
        # [3] & [4] Phase 03 & 04: Global Auto-ML & Adaptive Refinement
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

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_ADAPTIVE,
                "📍 [4] Enable Adaptive Spatial Refinement (Phase 04)",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.RESIDUAL_INTERP_METHOD,
                "📍 [4] Residual Interpolation Method",
                options=["Standard KNN", "Robust KNN (Huber)", "Kriging / Gaussian Process"],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.KNN_NEIGHBORS,
                "📍 [4] KNN Neighbors / Kriging Range",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=5,
            )
        )

        # -------------------------------------------------------------------
        # Advanced Hyperparameters & Settings (FlagAdvanced)
        # -------------------------------------------------------------------
        p_split = QgsProcessingParameterNumber(
            self.TRAIN_TEST_SPLIT, "🎛️ [Advanced] Training Data Ratio (e.g., 0.8 for 80%)", type=QgsProcessingParameterNumber.Double, defaultValue=0.8
        )
        p_split.setFlags(p_split.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_split)

        p_rs = QgsProcessingParameterNumber(
            self.RANDOM_STATE, "⚙️ [Advanced] Random State for ML Split", type=QgsProcessingParameterNumber.Integer, defaultValue=42
        )
        p_rs.setFlags(p_rs.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_rs)

        p_fmt = QgsProcessingParameterEnum(
            self.OUTPUT_FORMAT, "⚙️ [Advanced] Output Raster Format (Bit Depth)", options=["float32", "float64", "uint16"], defaultValue=0
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

        p_rf = QgsProcessingParameterString(self.PARAM_RF, "🎛️ [Advanced ML] Random Forest Hyperparameters", defaultValue="'n_estimators':[100, 500], 'max_depth':[10, 30]")
        p_rf.setFlags(p_rf.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_rf)

        p_gb = QgsProcessingParameterString(self.PARAM_GB, "🎛️ [Advanced ML] Gradient Boosting Hyperparameters", defaultValue="'n_estimators':[100, 300], 'learning_rate':[0.05, 0.1]")
        p_gb.setFlags(p_gb.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_gb)

        p_et = QgsProcessingParameterString(self.PARAM_ET, "🎛️ [Advanced ML] Extra Trees Hyperparameters", defaultValue="'n_estimators':[100, 500], 'max_depth':[10, 30]")
        p_et.setFlags(p_et.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_et)

        p_svr = QgsProcessingParameterString(self.PARAM_SVR, "🎛️ [Advanced ML] SVR Hyperparameters", defaultValue="'C':[1, 10, 100], 'kernel':['rbf'], 'cache_size':[1000], 'max_iter':[20000]")
        p_svr.setFlags(p_svr.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_svr)

        p_mlp = QgsProcessingParameterString(self.PARAM_MLP, "🎛️ [Advanced ML] MLP Hyperparameters", defaultValue="'hidden_layer_sizes':[(100,), (50, 50)], 'max_iter':[500]")
        p_mlp.setFlags(p_mlp.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_mlp)

        p_ridge = QgsProcessingParameterString(self.PARAM_RIDGE, "🎛️ [Advanced ML] Ridge Hyperparameters", defaultValue="'alpha':[0.1, 1.0, 10.0]")
        p_ridge.setFlags(p_ridge.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_ridge)

        p_lasso = QgsProcessingParameterString(self.PARAM_LASSO, "🎛️ [Advanced ML] Lasso Hyperparameters", defaultValue="'alpha':[0.01, 0.1, 1.0]")
        p_lasso.setFlags(p_lasso.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_lasso)

        p_elastic = QgsProcessingParameterString(self.PARAM_ELASTICNET, "🎛️ [Advanced ML] ElasticNet Hyperparameters", defaultValue="'alpha':[0.01, 0.1, 1.0], 'l1_ratio':[0.2, 0.5, 0.8]")
        p_elastic.setFlags(p_elastic.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_elastic)

        p_knn = QgsProcessingParameterString(self.PARAM_KNN, "🎛️ [Advanced ML] KNN Hyperparameters", defaultValue="'n_neighbors':[3, 5, 10], 'weights':['uniform', 'distance']")
        p_knn.setFlags(p_knn.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_knn)

        p_dt = QgsProcessingParameterString(self.PARAM_DT, "🎛️ [Advanced ML] Decision Tree Hyperparameters", defaultValue="'max_depth':[10, 20, None]")
        p_dt.setFlags(p_dt.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_dt)

        p_huber = QgsProcessingParameterString(self.PARAM_HUBER, "🎛️ [Advanced ML] Huber Regressor Hyperparameters", defaultValue="'epsilon':[1.35, 1.5, 1.75]")
        p_huber.setFlags(p_huber.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_huber)

        p_xgb = QgsProcessingParameterString(self.PARAM_XGB, "🎛️ [Advanced ML] XGBoost Hyperparameters", defaultValue="'n_estimators':[100, 300], 'learning_rate':[0.05, 0.1], 'max_depth':[3, 6]")
        p_xgb.setFlags(p_xgb.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_xgb)

        p_lgbm = QgsProcessingParameterString(self.PARAM_LGBM, "🎛️ [Advanced ML] LightGBM Hyperparameters", defaultValue="'n_estimators':[100, 300], 'learning_rate':[0.05, 0.1], 'num_leaves':[31, 50]")
        p_lgbm.setFlags(p_lgbm.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_lgbm)

        p_catboost = QgsProcessingParameterString(self.PARAM_CATBOOST, "🎛️ [Advanced ML] CatBoost Hyperparameters", defaultValue="'iterations':[100, 300], 'learning_rate':[0.05, 0.1], 'depth':[4, 6]")
        p_catboost.setFlags(p_catboost.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_catboost)

        p_ens_p4 = QgsProcessingParameterBoolean(self.ENABLE_ENSEMBLE_P4, "⚙️ [Advanced] Phase 04 Enable Ensemble", defaultValue=False)
        p_ens_p4.setFlags(p_ens_p4.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_ens_p4)

        p_ens_meth_p4 = QgsProcessingParameterEnum(self.ENSEMBLE_METHOD_P4, "📊 [Advanced] Phase 04 Ensemble Method", options=["Average", "Median", "Stacking"], defaultValue=0)
        p_ens_meth_p4.setFlags(p_ens_meth_p4.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_ens_meth_p4)

        p_ens_size_p4 = QgsProcessingParameterNumber(self.ENSEMBLE_SIZE_P4, "📊 [Advanced] Phase 04 Ensemble Size", type=QgsProcessingParameterNumber.Integer, defaultValue=3)
        p_ens_size_p4.setFlags(p_ens_size_p4.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_ens_size_p4)

        p_stack_p4 = QgsProcessingParameterEnum(
            self.STACK_COMPONENTS_P4,
            "🎯 [Advanced] Phase 04 Features for Retraining",
            options=["Feature Stack (Phase 01)", "Phase 03 Depth Map", "Residual Error Grid"],
            allowMultiple=True,
            defaultValue=[1, 2],
        )
        p_stack_p4.setFlags(p_stack_p4.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_stack_p4)

        p_corr_m_p4 = QgsProcessingParameterEnum(self.FEATURE_CORR_METHOD_P4, "🤖 [Advanced] Phase 04 Correlation Method", options=["Disabled", "Pearson (Linear)", "Spearman (Rank)", "Automatic-RANSAC", "Automatic-Random Forest"], defaultValue=3)
        p_corr_m_p4.setFlags(p_corr_m_p4.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_corr_m_p4)

        p_corr_t_p4 = QgsProcessingParameterEnum(self.FEATURE_CORR_THRESHOLD_P4, "🤖 [Advanced] Phase 04 Correlation Threshold", options=self.FEATURE_CORR_THRESHOLDS_P4, defaultValue=0)
        p_corr_t_p4.setFlags(p_corr_t_p4.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_corr_t_p4)

        p_sp_p4 = QgsProcessingParameterBoolean(self.SPATIAL_CV_P4, "🌍 [Advanced] Phase 04 Spatial CV", defaultValue=False)
        p_sp_p4.setFlags(p_sp_p4.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_sp_p4)

        # -------------------------------------------------------------------
        # [5] Phase 05: Post-processing & Temporal Products
        # -------------------------------------------------------------------
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.SHORELINE_ROI,
                "🗺️ [5] Shoreline Temporal ROI (Polygon)",
                types=[QgsProcessing.TypeVectorPolygon],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_SLOPE_FILTER,
                "🧽 [5] Enable Physical Slope Filtering",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SLOPE_THRESHOLD,
                "🧽 [5] Slope Angle Threshold (Degrees)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=35.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.REMOVE_POSITIVES,
                "🧽 [5] Remove Positive Depths (Keep Water Only <= 0)",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.OVERALL_TREND_METHOD,
                "📊 [5] Volumetric Trend Method (Overall Time Span)",
                options=self.TREND_METHODS_NAMES,
                defaultValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.COMPARISON_MODE,
                "📊 [5] Volumetric Change Analysis Mode (Sequential vs First Year Fixed)",
                options=self.COMPARISON_MODE_NAMES,
                defaultValue=0,
            )
        )



        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_MASTER_FOLDER, "📁 Main Output Folder"
            )
        )

    def _generate_linear_sdb_map(self, image_path, points_path, depth_field, output_path, osw_poly_path, feedback):
        """
        Fits a fast, smooth Linear Regression model on training points and predicts a 
        clean Linear SDB raster specifically for multi-temporal volumetric & morphology analysis.
        """
        if os.path.exists(output_path):
            return output_path
            
        try:
            import numpy as np
            import rasterio
            from sklearn.linear_model import LinearRegression
            from qgis.core import QgsVectorLayer
            
            try:
                from ...core.ml.trainers import extract_samples, predict_map
            except Exception:
                from Bathymetrix_AI.core.ml.trainers import extract_samples, predict_map

            vec_layer = points_path if isinstance(points_path, QgsVectorLayer) else QgsVectorLayer(points_path, "pts", "ogr")
            if not vec_layer or not vec_layer.isValid():
                if feedback:
                    feedback.pushWarning(f"⚠️ Invalid vector points layer for {image_path}")
                return None

            feedback.pushInfo(f"📐 [TEMPORAL ANALYTICS] Generating Linear Regression SDB map for {os.path.basename(image_path)}...")
            X, y, weights, coords = extract_samples(
                image_path, vec_layer, depth_field, None, 0
            )
            if X is not None and len(y) >= 5:
                model = LinearRegression()
                model.fit(X, y)
                predict_map(model, image_path, None, output_path, med_size=0, output_format="float32")
                
                if osw_poly_path and os.path.exists(osw_poly_path):
                    import processing
                    temp_clipped = output_path.replace(".tif", "_clipped.tif")
                    processing.run("gdal:cliprasterbymasklayer", {
                        "INPUT": output_path,
                        "MASK": osw_poly_path,
                        "NODATA": -9999.0,
                        "OUTPUT": temp_clipped
                    })
                    if os.path.exists(temp_clipped):
                        import shutil
                        if os.path.exists(output_path):
                            os.remove(output_path)
                        shutil.move(temp_clipped, output_path)
                return output_path
        except Exception as e:
            if feedback:
                feedback.pushWarning(f"⚠️ Linear Regression SDB fallback warning: {e}")
        return None

    def processAlgorithm(self, parameters, context, feedback):
        img_root = self.parameterAsString(parameters, self.INPUT_IMAGE_ROOT, context)
        out_folder = self.parameterAsString(parameters, self.OUTPUT_MASTER_FOLDER, context)
        num_threads = self.parameterAsInt(parameters, self.NUM_THREADS, context)

        coastal_idx = self.parameterAsInt(parameters, self.COASTAL_BAND, context)
        blue_idx = self.parameterAsInt(parameters, self.BLUE_BAND, context)
        green_idx = self.parameterAsInt(parameters, self.GREEN_BAND, context)
        red_idx = self.parameterAsInt(parameters, self.RED_BAND, context)
        nir_idx = self.parameterAsInt(parameters, self.NIR_BAND, context)
        
        field_depth = self.parameterAsString(parameters, self.FIELD_DEPTH, context)
        icesat_layer = self.parameterAsVectorLayer(parameters, self.INPUT_TRAIN, context)
        icesat_year_field = self.parameterAsString(parameters, self.FIELD_YEAR_ICESAT, context)
        
        osw_layer = self.parameterAsVectorLayer(parameters, self.DEEPWATER_ROI, context)
        osw_shp_path = osw_layer.source() if osw_layer else None

        shoreline_roi_layer = self.parameterAsVectorLayer(parameters, self.SHORELINE_ROI, context)
        shoreline_roi_shp = shoreline_roi_layer.source() if shoreline_roi_layer else None

        overall_trend_method_idx = self.parameterAsEnum(parameters, self.OVERALL_TREND_METHOD, context)
        overall_trend_method = self.TREND_METHODS_NAMES[overall_trend_method_idx]

        comparison_mode_idx = self.parameterAsEnum(parameters, self.COMPARISON_MODE, context)
        comparison_mode = self.COMPARISON_MODE_NAMES[comparison_mode_idx]


        os.makedirs(out_folder, exist_ok=True)

        feedback.pushInfo("==========================================================")
        feedback.pushInfo("🌊 Bathymetrix-AI Temporal Intelligence Initialized")
        feedback.pushInfo("==========================================================")

        # Step 0: Auto-discover years and pair multi-year dataset
        scanner = TemporalDataScanner(img_root)
        yearly_datasets = scanner.scan_yearly_datasets(
            icesat_layer=icesat_layer,
            icesat_year_field=icesat_year_field,
        )

        detected_years = sorted(list(yearly_datasets.keys()))
        feedback.pushInfo(f"📅 Auto-discovered target years ({len(detected_years)}): {detected_years}")

        # Directly pass native QGIS parameters dictionary to MasterFlow runner
        masterflow_params = parameters.copy()
        masterflow_params["ENABLE_VALIDATION"] = False

        # Step 1: Automated SDB MasterFlow per year
        sdb_runner = TemporalSDBRunner(out_folder)
        yearly_sdb_results = {}
        first_year_osw_path = None
        
        for yr in detected_years:
            current_masterflow_params = masterflow_params.copy()
            
            # Apply OSW polygon from the first year to all subsequent years
            if first_year_osw_path and os.path.exists(first_year_osw_path):
                current_masterflow_params["DEEPWATER_METHOD"] = 2  # Shallow Water Bound (OSW Polygon)
                current_masterflow_params["DEEPWATER_ROI"] = first_year_osw_path
                feedback.pushInfo(f"♻️ Reusing OSW Deep Water Polygon from the first year for {yr}")

            res = sdb_runner.run_year(
                year_info=yearly_datasets[yr],
                masterflow_params=current_masterflow_params,
                feedback=feedback,
                context=context,
            )
            yearly_sdb_results[yr] = res
            
            # Capture the OSW polygon from the first year
            if not first_year_osw_path:
                candidate_osw = os.path.join(res["year_out_dir"], "Phase_01_Preprocessing", "07_OSW_Boundary_Polygon.gpkg")
                if os.path.exists(candidate_osw):
                    first_year_osw_path = candidate_osw

        # Step 2: Benthic Vegetation Classifier (Disabled for now per user request)
        benthic_results = {}
        # from ...core.temporal.benthic_classifier import BenthicVegetationClassifier
        # benthic_classifier = BenthicVegetationClassifier(coastal_idx, blue_idx, green_idx, red_idx)
        # for yr, sdb_res in sorted(yearly_sdb_results.items()):
        #     b_res = benthic_classifier.process_year_benthic(
        #         year=yr,
        #         image_path=sdb_res["image_path"],
        #         sdb_depth_path=sdb_res["sdb_depth_map"],
        #         output_dir=sdb_res["year_out_dir"],
        #         feedback=feedback
        #     )
        #     benthic_results[yr] = b_res

        # Step 3: Shoreline Change Polygons
        shoreline_out_folder = os.path.join(out_folder, "Shoreline_Analysis")
        os.makedirs(shoreline_out_folder, exist_ok=True)
        change_polygons = []
        
        shoreline_tracker = ShorelineDynamicsTracker()
        yearly_masks = {}
        for yr, sdb_res in sorted(yearly_sdb_results.items()):
            m_path = shoreline_tracker.extract_year_shoreline(
                year=yr,
                image_path=sdb_res["image_path"],
                output_dir=sdb_res["year_out_dir"],
                feedback=feedback,
                nir_idx=nir_idx,
                green_idx=green_idx
            )
            yearly_masks[yr] = m_path

        # Resolve effective OSW polygon (use manual polygon if supplied, else auto-generated Year 1 polygon)
        effective_osw_shp = osw_shp_path if (osw_shp_path and os.path.exists(osw_shp_path)) else first_year_osw_path

        years = sorted(list(yearly_masks.keys()))
        if len(years) >= 2:
            # 1. Overall Time Span ONLY
            yFirst, yLast = years[0], years[-1]
            poly_shp = shoreline_tracker.compute_shoreline_change_polygons(
                yFirst, yLast, yearly_masks[yFirst], yearly_masks[yLast], effective_osw_shp, shoreline_out_folder, feedback, shoreline_roi_shp
            )
            if poly_shp: change_polygons.append(poly_shp)

        # Step 4: StatCD Bathymetric Change Detection & Sediment Mass Balance (m³)
        bathymetric_out_folder = os.path.join(out_folder, "Bathymetric_Change_Analysis")
        os.makedirs(bathymetric_out_folder, exist_ok=True)
        analytics_engine = TemporalAnalyticsEngine()
        
        # Build / Retrieve Linear Regression SDB maps specifically for Volumetric & MSI analytics
        linear_sdb_maps = {}
        linear_uncert_maps = {}
        for yr, res in yearly_sdb_results.items():
            if "sdb_linear_map" in res and res["sdb_linear_map"] and os.path.exists(res["sdb_linear_map"]):
                linear_sdb_maps[yr] = res["sdb_linear_map"]
                linear_uncert_maps[yr] = res.get("linear_uncertainty_map", res.get("uncertainty_map", ""))
                feedback.pushInfo(f"✅ Utilizing Phase 03 Isolated Linear Regression Map & Uncertainty for {yr} Analytics.")
            else:
                linear_sdb_maps[yr] = res["sdb_depth_map"]
                linear_uncert_maps[yr] = res.get("uncertainty_map", "")
                feedback.pushWarning(f"⚠️ Phase 03 Linear Regression Map not found for {yr}. Falling back to Best Model map.")

        analytics_results = analytics_engine.process_temporal_change(
            sdb_maps=linear_sdb_maps,
            uncertainty_maps=linear_uncert_maps,
            output_dir=bathymetric_out_folder,
            feedback=feedback,
            osw_shp=effective_osw_shp,
            overall_trend_method=overall_trend_method,
            comparison_mode=comparison_mode
        )

        # Step 5: QGIS Layer Group Organization & Summary Reports
        report_gen = TemporalReportGenerator()
        report_gen.generate_layer_group_and_reports(
            yearly_sdb_results=yearly_sdb_results,
            change_polygons=change_polygons,
            analytics_results=analytics_results,
            benthic_results=benthic_results,
            output_dir=out_folder,
            feedback=feedback,
        )

        # Stop QGIS from auto-loading intermediate child outputs (e.g., Phase03_Depth_OSW_Clipped)
        try:
            if hasattr(context, "layersToLoad"):
                context.layersToLoad().clear()
        except Exception:
            pass

        feedback.pushInfo("🎉 Bathymetrix-AI Multi-Year Coastal Dynamics Analysis execution complete!")
        return {self.OUTPUT_MASTER_FOLDER: out_folder}
