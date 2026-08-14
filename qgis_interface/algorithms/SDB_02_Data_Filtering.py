import warnings

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterEnum,
    QgsProcessingParameterBand,
    QgsProcessingOutputVectorLayer,
)

warnings.filterwarnings("ignore")


class SDBModule02(QgsProcessingAlgorithm):
    INPUT_STACK = "INPUT_STACK"
    INPUT_POINTS = "INPUT_POINTS"
    FIELD_DEPTH = "FIELD_DEPTH"
    BLUE_BAND = "BLUE_BAND"
    GREEN_BAND = "GREEN_BAND"
    RESIDUAL_THRESHOLD = "RESIDUAL_THRESHOLD"
    RANSAC_MAX_TRIALS = "RANSAC_MAX_TRIALS"
    FILTER_MODE = "FILTER_MODE"
    OUTPUT_FOLDER = "OUTPUT_FOLDER"
    OUTPUT_CLEAN_VEC = "OUTPUT_CLEAN_VEC"

    FILTER_MODES = ["Linear RANSAC", "LS Variance Fit", "Huber Variance Fit"]

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_STACK, "Input Feature Stack (Phase 01)"
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(self.INPUT_POINTS, "Raw ICESat-2 Points")
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_DEPTH,
                "Depth Field",
                defaultValue="depth",
                parentLayerParameterName=self.INPUT_POINTS,
                type=QgsProcessingParameterField.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.FILTER_MODE,
                "Filtering Strategy",
                options=self.FILTER_MODES,
                defaultValue=2,
            )
        )
        
        self.addParameter(
            QgsProcessingParameterBand(
                self.BLUE_BAND,
                "Numerator Band (e.g., Blue)",
                parentLayerParameterName=self.INPUT_STACK,
                defaultValue=2,
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                self.GREEN_BAND,
                "Denominator Band (e.g., Green)",
                parentLayerParameterName=self.INPUT_STACK,
                defaultValue=3,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.RESIDUAL_THRESHOLD,
                "Threshold/Multiplier (0=Auto)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.RANSAC_MAX_TRIALS,
                "RANSAC Max Trials",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=100,
            )
        )
        self.addParameter(
            QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, "Output Folder")
        )
        
        self.addOutput(
            QgsProcessingOutputVectorLayer(
                self.OUTPUT_CLEAN_VEC,
                "Output Clean Vector Points"
            )
        )

    def name(self):
        return "sdb_02_filtering"

    def displayName(self):
        return "2. SDB Module 02: Filtering (Multi-Mode)"

    def group(self):
        return "SDB Research Tools"

    def groupId(self):
        return "sdb_tools"

    def createInstance(self):
        return SDBModule02()

    def shortHelpString(self):
        return """
        <div style="font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.5; color: #2C3E50;">
            <h2 style="margin-bottom: 5px; color: #2E86C1;">🧹 SDB Module 02: Robust Data Filtering</h2>
            <p style="margin-top: 0; margin-bottom: 15px; font-size: 13px;">
                Filters noisy ICESat-2 LiDAR tracks, sonar soundings, or other altimetry using advanced statistical outlier rejection strategies to produce a clean, reliable training dataset for AI modeling.
            </p>
            
            <h3 style="color: #117A65; margin-bottom: 5px; border-bottom: 2px solid #117A65; padding-bottom: 3px;">🧹 Filtering Strategies</h3>
            <ul style="font-size: 12px; margin-top: 5px; padding-left: 20px;">
                <li><b>Linear RANSAC (Random Sample Consensus):</b> Highly effective for optical log-ratio relationships contaminated with extreme random outliers and photon scatter.</li>
                <li><b>Least-Squares (LS) Variance Fit:</b> Adaptive filtering suitable for non-linear data distributions where noise dispersion remains relatively uniform.</li>
                <li><b>Huber Variance Fit:</b> Robust estimator that penalizes gross outliers linearly while maintaining quadratic sensitivity for inliers; ideal where depth uncertainty scales with depth.</li>
            </ul>

            <h3 style="color: #D35400; margin-top: 15px; margin-bottom: 5px; border-bottom: 2px solid #D35400; padding-bottom: 3px;">📊 Diagnostic Visualizations</h3>
            <p style="font-size: 12px; margin-top: 5px;">
                Generates dynamic percentile-based scatter and regression diagnostic plots to visualize rejected vs. accepted training observations without distortion from extreme outliers.
            </p>
            <br><b>Developer:</b> Mohamed Aly Nasef
        </div>
        """

    def helpString(self):
        return self.shortHelpString()

    def processAlgorithm(self, parameters, context, feedback):
        from ...core.processing.filtering import run_phase02_filtering

        return run_phase02_filtering(self, parameters, context, feedback)
