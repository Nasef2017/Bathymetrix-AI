# SDB_01_Preprocessing.py
# ---------------------------------------------------------------------------
# MODULE 01: COMPREHENSIVE PRE-PROCESSING
# Features: Vector Polygon Masking, Auto/Manual NDWI Masking, Hedley Sunglint, Feature Gen
# Updates: Removed Dummy Mask, Added Vector Polygon Support with CRS alignment,
#          Float64 Hedley Covariance math maintained.
# ---------------------------------------------------------------------------

import warnings

from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterBand,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
    QgsProcessingParameterDefinition,
    QgsProcessingOutputRasterLayer,
    QgsProcessingOutputVectorLayer,
)

warnings.filterwarnings("ignore")


class SDBPhase1Preprocessing(QgsProcessingAlgorithm):
    INPUT_RASTER = "INPUT_RASTER"
    OUTPUT_FOLDER = "OUTPUT_FOLDER"
    OUTPUT_FEATURES = "OUTPUT_FEATURES"
    OUTPUT_MASK = "OUTPUT_MASK"
    OUTPUT_OSW_POLY = "OUTPUT_OSW_POLY"

    COASTAL_BAND = "COASTAL_BAND"
    BLUE_BAND = "BLUE_BAND"
    GREEN_BAND = "GREEN_BAND"
    RED_BAND = "RED_BAND"
    NIR_BAND = "NIR_BAND"
    SWIR_BAND = "SWIR_BAND"

    INPUT_WATER_POLY = "INPUT_WATER_POLY"
    ENABLE_MASKING = "ENABLE_MASKING"
    MASKING_METHOD = "MASKING_METHOD"
    MANUAL_THRESHOLD = "MANUAL_THRESHOLD"
    OTSU_ADJUSTMENT = "OTSU_ADJUSTMENT"
    MASK_KERNEL_SIZE = "MASK_KERNEL_SIZE"

    APPLY_SUNGLINT = "APPLY_SUNGLINT"
    SUNGLINT_PERCENTILE = "SUNGLINT_PERCENTILE"

    FEATURE_SELECTION = "FEATURE_SELECTION"
    ENABLE_BAND_CALC = "ENABLE_BAND_CALC"
    BAND_MATH_FORMULA = "BAND_MATH_FORMULA"
    
    # OSW Filter Constants
    APPLY_DEEPWATER = "APPLY_DEEPWATER"
    DEEPWATER_METHOD = "DEEPWATER_METHOD"
    DEEPWATER_ROI = "DEEPWATER_ROI"
    NIR_PERCENTILE_OSW = "NIR_PERCENTILE_OSW"
    APPLY_TURBIDITY = "APPLY_TURBIDITY"
    TURBIDITY_THRESHOLD = "TURBIDITY_THRESHOLD"
    OSW_MEDIAN_SIZE = "OSW_MEDIAN_SIZE"
    FILL_INTERNAL_HOLES = "FILL_INTERNAL_HOLES"
    EXTRACT_POLYGON = "EXTRACT_POLYGON"

    NUM_THREADS = "NUM_THREADS"

    MASK_METHODS = ["Otsu (Automatic NDWI)", "Manual NDWI Threshold", "3 Indices Equation (NDWI, MNDWI, NWI)", "Smart Hybrid (Dynamic Auto)"]
    OSW_METHODS = [
        "Automated Knee-Point Extinction [Recommended]",
        "Turbidity-Invariant Log-Ratio Extinction",
        "Multi-Otsu / GMM Spectral Clustering",
        "Automatic (NIR Percentile Fallback)",
        "Manual Polygon ROI",
        "Shallow Water Bound (OSW Polygon)",
    ]

    FEATURE_OPTIONS = [
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

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_RASTER, "Input Satellite Image (Raw)"
            )
        )
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_FOLDER, "Output Folder (For Module 1 Results)"
            )
        )

        self.addParameter(
            QgsProcessingParameterBand(
                self.COASTAL_BAND,
                "Coastal/Aerosol Band",
                parentLayerParameterName=self.INPUT_RASTER,
                defaultValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                self.BLUE_BAND,
                "Blue Band",
                parentLayerParameterName=self.INPUT_RASTER,
                defaultValue=2,
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                self.GREEN_BAND,
                "Green Band",
                parentLayerParameterName=self.INPUT_RASTER,
                defaultValue=3,
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                self.RED_BAND,
                "Red Band",
                parentLayerParameterName=self.INPUT_RASTER,
                defaultValue=4,
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                self.NIR_BAND,
                "NIR Band",
                parentLayerParameterName=self.INPUT_RASTER,
                defaultValue=8,
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                self.SWIR_BAND,
                "SWIR Band (For 3 Indices Mask)",
                parentLayerParameterName=self.INPUT_RASTER,
                defaultValue=11,
            )
        )

        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_WATER_POLY,
                "1A. Input Water Polygon (Optional - Overrides Auto Mask)",
                types=[QgsProcessing.TypeVectorPolygon],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_MASKING,
                "1B. Enable Auto/Manual Masking (If no Polygon provided)",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.MASKING_METHOD,
                "Water Masking Method",
                options=self.MASK_METHODS,
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MANUAL_THRESHOLD,
                "Manual Threshold",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.OTSU_ADJUSTMENT,
                "Otsu Threshold Adjustment",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MASK_KERNEL_SIZE,
                "Mask Cleanup Kernel Size (Smoothness)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=3,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.APPLY_SUNGLINT,
                "2. Apply Sunglint Correction (Hedley)",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SUNGLINT_PERCENTILE,
                "Sunglint NIR Minimum Percentile",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1.0,
            )
        )

        num_options = len(self.FEATURE_OPTIONS)
        default_selection = list(range(num_options))
        self.addParameter(
            QgsProcessingParameterEnum(
                self.FEATURE_SELECTION,
                "3. Output Feature Stack Selection",
                options=self.FEATURE_OPTIONS,
                allowMultiple=True,
                defaultValue=default_selection,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ENABLE_BAND_CALC, "Enable Custom Band Math", defaultValue=True
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.BAND_MATH_FORMULA,
                "Band Math Formula (e.g. (B2-B3)/(B2+B3))",
                defaultValue="",
                optional=True,
            )
        )

        # 4. OSW Filter
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.APPLY_DEEPWATER,
                "4. Apply Deep Water Filter",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.DEEPWATER_METHOD,
                "Deep Water Definition Method",
                options=self.OSW_METHODS,
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.DEEPWATER_ROI,
                "Deep Water ROI (Polygon) [Optional if Manual Mode]",
                types=[QgsProcessing.TypeVectorPolygon],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.NIR_PERCENTILE_OSW,
                "NIR Percentile for Deep Water (e.g. 10%)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=10.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.APPLY_TURBIDITY,
                "Apply Turbidity Filter",
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.TURBIDITY_THRESHOLD,
                "Turbidity Threshold (e.g. 0.02 or 200)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=200,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.OSW_MEDIAN_SIZE,
                "OSW Mask Median Filter Size (0 to disable)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=5,
            )
        )
        p_fill = QgsProcessingParameterBoolean(
            self.FILL_INTERNAL_HOLES,
            "Fill All Internal Holes in OSW Mask",
            defaultValue=True,
        )
        p_fill.setFlags(p_fill.flags() | QgsProcessingParameterDefinition.FlagHidden)
        self.addParameter(p_fill)

        p_extract = QgsProcessingParameterBoolean(
            self.EXTRACT_POLYGON,
            "Extract OSW Mask as Polygon",
            defaultValue=True,
        )
        p_extract.setFlags(p_extract.flags() | QgsProcessingParameterDefinition.FlagHidden)
        self.addParameter(p_extract)

        self.addParameter(
            QgsProcessingParameterNumber(
                self.NUM_THREADS,
                "Processing Threads",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=4,
            )
        )

        self.addOutput(
            QgsProcessingOutputRasterLayer(
                self.OUTPUT_FEATURES,
                "Output Feature Stack"
            )
        )
        self.addOutput(
            QgsProcessingOutputRasterLayer(
                self.OUTPUT_MASK,
                "Output Water Mask"
            )
        )
        self.addOutput(
            QgsProcessingOutputVectorLayer(
                self.OUTPUT_OSW_POLY,
                "Output OSW Polygon"
            )
        )

    def name(self):
        return "sdb_phase1_preprocessing"

    def displayName(self):
        return "1. SDB Module 01: Pre-processing"

    def group(self):
        return "SDB Research Tools"

    def groupId(self):
        return "sdb_tools"

    def createInstance(self):
        return SDBPhase1Preprocessing()

    def shortHelpString(self):
        return """
        <div style="font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.5; color: #2C3E50;">
            <h2 style="margin-bottom: 5px; color: #2E86C1;">🌊 SDB Module 01: Pre-processing</h2>
            <p style="margin-top: 0; margin-bottom: 15px; font-size: 13px;">
                Prepares multispectral satellite imagery for optical bathymetry by isolating the aquatic domain, eliminating sun-glint, and extracting depth-sensitive Log-Ratio features.
            </p>
            
            <h3 style="color: #117A65; margin-bottom: 5px; border-bottom: 2px solid #117A65; padding-bottom: 3px;">⚙️ Key Capabilities</h3>
            <ul style="font-size: 12px; margin-top: 5px; padding-left: 20px;">
                <li><b>Sun-Glint Removal:</b> Physics-based de-glinting <i>(Hedley et al., 2005)</i> with robust NaN and Infinity protection to improve shallow seabed contrast.</li>
                <li><b>Automated Water Masking:</b> Aquatic domain segmentation via NDWI, MNDWI, or 3-Indices formulas with optional shoreline edge shrinking to avoid land-water mixing.</li>
                <li><b>Deep Water OSW Filtering:</b> Removes light-extinct deep waters via automatic NIR percentile thresholding, dynamic Elbow Point Detection, or manual polygon boundary masking.</li>
                <li><b>OSW Vector Export:</b> Automatically saves and exports the Optically Shallow Water boundary as a GeoPackage vector layer matching source CRS.</li>
                <li><b>Log-Ratio Spectral Features:</b> Computes physics-based log-transformed spectral ratios (e.g. ln(Blue)/ln(Green), ln(Coastal)/ln(Yellow)) based on differential light attenuation.</li>
            </ul>
            <br><b>Developer:</b> Mohamed Aly Nasef
        </div>
        """

    def helpString(self):
        return self.shortHelpString()

    def processAlgorithm(self, parameters, context, feedback):
        from ...core.processing.preprocess import run_phase01_preprocessing

        return run_phase01_preprocessing(self, parameters, context, feedback)
