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
                self.INPUT_MAP_P3, "Phase 3 Depth Map (Initial Global)"
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_MAP_P4, "Phase 4 Depth Map (Final Refined / Best Map)"
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
        <div style="font-family: Arial, sans-serif; line-height: 1.4;">
            <h2 style="color: #2E86C1;">SDB Phase 05 — Scientific Validation</h2>
            <p>Compares Phase 3 (global model) vs Phase 4 (best final map) against unseen validation points.</p>

            <b>Metrics Calculated:</b>
            <ul>
                <li><b>RMSE</b>: Root Mean Square Error (main accuracy metric)</li>
                <li><b>R²</b>: Coefficient of Determination (goodness of fit)</li>
                <li><b>MAE</b>: Mean Absolute Error</li>
                <li><b>Bias</b>: Systematic over/under-estimation</li>
                <li><b>wMAPE</b>: Weighted Mean Absolute Percentage Error</li>
            </ul>

            <b>Stratified Analysis:</b>
            <ul><li>Accuracy broken down by depth zones (0–5m, 5–10m, etc.)</li></ul>

            <b>Output Files:</b>
            <ul>
                <li>5_FINAL_SUMMARY.txt — Comparison report and winner verdict</li>
                <li>5_Validation_Raw_Data.csv — Point-by-point predictions and errors</li>
                <li>5_Stratified_Error_Analysis.csv — Metrics per depth zone</li>
                <li>5_Plot_Scatter_Comparison.png</li>
                <li>5_Plot_Residuals.png</li>
                <li>5_Plot_Error_Histogram.png</li>
            </ul>
        </div>
        """

    def helpString(self):
        return self.shortHelpString()

    def processAlgorithm(self, parameters, context, feedback):
        from ...core.processing.reporting import run_phase05_reporting

        return run_phase05_reporting(self, parameters, context, feedback)
