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

    FILTER_MODES = ["Linear RANSAC", "LS Variance Fit", "Huber Variance Fit"]

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_STACK, "Input Feature Stack (Phase 1)"
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(self.INPUT_POINTS, "Raw ICESat-2 Points")
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_DEPTH,
                "Depth Field",
                parentLayerParameterName=self.INPUT_POINTS,
                type=QgsProcessingParameterField.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                self.BLUE_BAND,
                "Blue Band Number",
                parentLayerParameterName=self.INPUT_STACK,
                defaultValue=2,
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                self.GREEN_BAND,
                "Green Band Number",
                parentLayerParameterName=self.INPUT_STACK,
                defaultValue=3,
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
        <div style="font-family: Arial, sans-serif; line-height: 1.2;">
            <h2 style="margin-bottom: 5px;">🔬 Filtering Strategy Guide</h2>
            <p>This module filters noisy ICESat-2 data using one of three statistical methods.</p>
            <b style="display: block; margin-bottom: 2px;">1. Linear RANSAC</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li><b>Best for:</b> Data with a clear linear relationship but contaminated with significant, random outliers.</li>
            </ul>
            <b style="display: block; margin-bottom: 2px;">2. LS Variance Fit</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li><b>Best for:</b> Data with a non-linear trend where the noise level is expected to be constant across all depths.</li>
            </ul>
            <b style="display: block; margin-bottom: 2px;">3. Huber Variance Fit</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li><b>Best for:</b> Complex scenarios where data uncertainty increases with depth (heteroscedasticity).</li>
            </ul>
        </div>
        """

    def helpString(self):
        return self.shortHelpString()

    def processAlgorithm(self, parameters, context, feedback):
        from ...core.processing.filtering import run_phase02_filtering

        return run_phase02_filtering(self, parameters, context, feedback)
