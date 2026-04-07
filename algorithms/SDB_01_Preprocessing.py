# SDB_01_Preprocessing.py
# ---------------------------------------------------------------------------
# MODULE 01: COMPREHENSIVE PRE-PROCESSING
# Features: Vector Polygon Masking, Auto/Manual NDWI Masking, Hedley Sunglint, Feature Gen
# Updates: Removed Dummy Mask, Added Vector Polygon Support with CRS alignment,
#          Float64 Hedley Covariance math maintained.
# ---------------------------------------------------------------------------

import warnings

from qgis.core import (
    QgsProcessing, QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer, QgsProcessingParameterVectorLayer,
    QgsProcessingParameterBand, QgsProcessingParameterFolderDestination,
    QgsProcessingParameterBoolean, QgsProcessingParameterEnum, QgsProcessingParameterNumber,
    QgsProcessingParameterString,
)

warnings.filterwarnings("ignore")


class SDBPhase1Preprocessing(QgsProcessingAlgorithm):
    INPUT_RASTER = 'INPUT_RASTER'
    OUTPUT_FOLDER = 'OUTPUT_FOLDER'

    COASTAL_BAND = 'COASTAL_BAND'
    BLUE_BAND = 'BLUE_BAND'
    GREEN_BAND = 'GREEN_BAND'
    RED_BAND = 'RED_BAND'
    NIR_BAND = 'NIR_BAND'

    INPUT_WATER_POLY = 'INPUT_WATER_POLY'
    ENABLE_MASKING = 'ENABLE_MASKING'
    MASKING_METHOD = 'MASKING_METHOD'
    MANUAL_THRESHOLD = 'MANUAL_THRESHOLD'
    OTSU_ADJUSTMENT = 'OTSU_ADJUSTMENT'
    MASK_KERNEL_SIZE = 'MASK_KERNEL_SIZE'

    APPLY_SUNGLINT = 'APPLY_SUNGLINT'
    NIR_BAND_SUNGLINT = 'NIR_BAND_SUNGLINT'
    SUNGLINT_PERCENTILE = 'SUNGLINT_PERCENTILE'

    FEATURE_SELECTION = 'FEATURE_SELECTION'
    ENABLE_BAND_CALC = 'ENABLE_BAND_CALC'
    BAND_MATH_FORMULA = 'BAND_MATH_FORMULA'
    NUM_THREADS = 'NUM_THREADS'

    MASK_METHODS = ['Otsu (Automatic NDWI)', 'Manual NDWI Threshold']

    FEATURE_OPTIONS = [
        '[All Raw] All Bands from Input Image',
        '[Log] Log(Coastal)',
        '[Log] Log(Blue)',
        '[Log] Log(Green)',
        '[Log] Log(Red)',
        '[Log] Log(NIR)',
        '[Ratio] Log(Blue) / Log(Green)',
        '[Ratio] Log(Blue) / Log(Red)',
        '[Ratio] Log(Coastal) / Log(Green)',
        '[Custom] Band Math Calculator'
    ]

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_RASTER, 'Input Satellite Image (Raw)'))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, 'Output Folder (For Module 1 Results)'))

        self.addParameter(QgsProcessingParameterBand(self.COASTAL_BAND, 'Coastal/Aerosol Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=1))
        self.addParameter(QgsProcessingParameterBand(self.BLUE_BAND, 'Blue Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=2))
        self.addParameter(QgsProcessingParameterBand(self.GREEN_BAND, 'Green Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=3))
        self.addParameter(QgsProcessingParameterBand(self.RED_BAND, 'Red Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=4))
        self.addParameter(QgsProcessingParameterBand(self.NIR_BAND, 'NIR Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=8))

        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_WATER_POLY, '1A. Input Water Polygon (Optional - Overrides Auto Mask)', types=[QgsProcessing.TypeVectorPolygon], optional=True))
        self.addParameter(QgsProcessingParameterBoolean(self.ENABLE_MASKING, '1B. Enable Auto/Manual Masking (If no Polygon provided)', defaultValue=True))
        self.addParameter(QgsProcessingParameterEnum(self.MASKING_METHOD, 'Water Masking Method', options=self.MASK_METHODS, defaultValue=0))
        self.addParameter(QgsProcessingParameterNumber(self.MANUAL_THRESHOLD, 'Manual Threshold', type=QgsProcessingParameterNumber.Double, defaultValue=0.0, optional=True))
        self.addParameter(QgsProcessingParameterNumber(self.OTSU_ADJUSTMENT, 'Otsu Threshold Adjustment', type=QgsProcessingParameterNumber.Double, defaultValue=0.0))
        self.addParameter(QgsProcessingParameterNumber(self.MASK_KERNEL_SIZE, 'Mask Cleanup Kernel Size (Smoothness)', type=QgsProcessingParameterNumber.Integer, defaultValue=3))

        self.addParameter(QgsProcessingParameterBoolean(self.APPLY_SUNGLINT, '2. Apply Sunglint Correction (Hedley)', defaultValue=True))
        self.addParameter(QgsProcessingParameterBand(self.NIR_BAND_SUNGLINT, 'Sunglint NIR Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=8))
        self.addParameter(QgsProcessingParameterNumber(self.SUNGLINT_PERCENTILE, 'Sunglint NIR Minimum Percentile', type=QgsProcessingParameterNumber.Double, defaultValue=1.0))

        num_options = len(self.FEATURE_OPTIONS)
        default_selection = list(range(num_options))
        self.addParameter(QgsProcessingParameterEnum(self.FEATURE_SELECTION, '3. Output Feature Stack Selection', options=self.FEATURE_OPTIONS, allowMultiple=True, defaultValue=default_selection, optional=True))
        self.addParameter(QgsProcessingParameterBoolean(self.ENABLE_BAND_CALC, 'Enable Custom Band Math', defaultValue=True))
        self.addParameter(QgsProcessingParameterString(self.BAND_MATH_FORMULA, 'Band Math Formula (e.g. (B2-B3)/(B2+B3))', defaultValue='', optional=True))
        self.addParameter(QgsProcessingParameterNumber(self.NUM_THREADS, 'Processing Threads', type=QgsProcessingParameterNumber.Integer, defaultValue=4))

    def name(self): return 'sdb_phase1_preprocessing'
    def displayName(self): return '1. SDB Module 01: Pre-processing'
    def group(self): return 'SDB Research Tools'
    def groupId(self): return 'sdb_tools'
    def createInstance(self): return SDBPhase1Preprocessing()
    def shortHelpString(self): return "<p><b>Masking Logic:</b> If a Vector Polygon is provided, it perfectly masks the water. If not, NDWI is used. If masking is disabled entirely, Sunglint and Features process the whole valid image.</p>"
    def helpString(self): return self.shortHelpString()

    def processAlgorithm(self, parameters, context, feedback):
        from ..sdb_pipeline.phases.phase01_preprocess import run_phase01_preprocessing
        return run_phase01_preprocessing(self, parameters, context, feedback)
