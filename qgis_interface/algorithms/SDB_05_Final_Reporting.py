import warnings

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField,
    QgsProcessingParameterFolderDestination,
)

warnings.filterwarnings("ignore")


class SDBModule05(QgsProcessingAlgorithm):

    INPUT_MAP_P3 = "INPUT_MAP_P3"
    INPUT_MAP_P4 = "INPUT_MAP_P4"
    INPUT_TRAIN = "INPUT_TRAIN"
    FIELD_TRAIN = "FIELD_TRAIN"
    INPUT_VALIDATION = "INPUT_VALIDATION"
    FIELD_VAL_DEPTH = "FIELD_VAL_DEPTH"
    OUTPUT_FOLDER = "OUTPUT_FOLDER"

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_MAP_P3, "Phase 03 Depth Map (Initial Global)"
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_MAP_P4, "Phase 04 Depth Map (Final Refined / Best Map)"
            )
        )

        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_TRAIN, "Training Points (Reference)"
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_TRAIN,
                "Depth Field (Training)",
                parentLayerParameterName=self.INPUT_TRAIN,
                type=QgsProcessingParameterField.Numeric,
            )
        )

        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_VALIDATION, "Unseen Validation Points"
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_VAL_DEPTH,
                "Depth Field (Validation)",
                parentLayerParameterName=self.INPUT_VALIDATION,
                type=QgsProcessingParameterField.Numeric,
            )
        )

        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_FOLDER, "Output Folder (Reports)"
            )
        )

    def name(self):
        return "sdb_05_reporting"

    def displayName(self):
        return "5. SDB Module 05: Scientific Validation & Reporting"

    def group(self):
        return "SDB Research Tools"

    def groupId(self):
        return "sdb_tools"

    def createInstance(self):
        return SDBModule05()

    def shortHelpString(self):
        return """
        <div style="font-family: Arial, sans-serif; line-height: 1.2;">
            <h2 style="margin-bottom: 5px;">📉 <span style="color: #2E86C1;">SDB Module 05</span>: Scientific Validation</h2>
            <p style="margin-top: 0; margin-bottom: 10px;">Compares Phase 03 (global model) vs Phase 04 (best final map) against unseen validation points.</p>

            <b style="display: block; margin-bottom: 2px;">📉 Phase 05: Validation Metrics</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li><b>RMSE:</b> Root Mean Square Error (main accuracy metric).</li>
                <li><b>R²:</b> Coefficient of Determination (goodness of fit).</li>
                <li><b>MAE:</b> Mean Absolute Error.</li>
                <li><b>Bias:</b> Systematic over/under-estimation.</li>
                <li><b>wMAPE:</b> Weighted Mean Absolute Percentage Error.</li>
            </ul>

            <b style="display: block; margin-bottom: 2px;">📊 Detailed Analysis</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li><b>Stratified Analysis:</b> Accuracy broken down by depth zones (0-5m, 5-10m, etc.).</li>
                <li><b>Output Plots:</b> Generates Scatter Comparisons, Residuals, and Error Histograms.</li>
            </ul>
        </div>
        """

    def helpString(self):
        return self.shortHelpString()

    def processAlgorithm(self, parameters, context, feedback):
        from ...core.processing.reporting import run_phase05_reporting

        return run_phase05_reporting(self, parameters, context, feedback)
