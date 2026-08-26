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
                defaultValue="ortho_h",
                parentLayerParameterName=self.INPUT_TRAIN,
                type=QgsProcessingParameterField.Numeric,
            )
        )

        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_VALIDATION, "Unseen Validation Points", optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_VAL_DEPTH,
                "Depth Field (Validation)",
                defaultValue="ortho_h",
                parentLayerParameterName=self.INPUT_VALIDATION,
                type=QgsProcessingParameterField.Numeric,
                optional=True,
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
        <div style="font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.5; color: #2C3E50;">
            <h2 style="margin-bottom: 5px; color: #2E86C1;">📉 SDB Module 05: Scientific Validation & IHO Compliance</h2>
            <p style="margin-top: 0; margin-bottom: 15px; font-size: 13px;">
                Performs independent scientific accuracy assessment and hydrographic standard compliance analysis by benchmarking predicted SDB depths against unseen validation observations.
            </p>

            <h3 style="color: #D35400; margin-bottom: 5px; border-bottom: 2px solid #D35400; padding-bottom: 3px;">📊 Statistical Metrics</h3>
            <ul style="font-size: 12px; margin-top: 5px; padding-left: 20px;">
                <li><b>R² (Coefficient of Determination):</b> Quantifies model goodness-of-fit and variance explanation.</li>
                <li><b>RMSE (Root Mean Square Error):</b> Global vertical accuracy standard in meters.</li>
                <li><b>MAE & Mean Bias:</b> Measures average magnitude and systematic positive/negative vertical drift.</li>
                <li><b>wMAPE:</b> Weighted Mean Absolute Percentage Error tailored for bathymetry across depth ranges.</li>
            </ul>

            <h3 style="color: #117A65; margin-top: 15px; margin-bottom: 5px; border-bottom: 2px solid #117A65; padding-bottom: 3px;">🌊 Depth-Stratified & IHO S-44 Assessment</h3>
            <ul style="font-size: 12px; margin-top: 5px; padding-left: 20px;">
                <li><b>Stratified Depth Bins:</b> Computes zoned error metrics (0–2m, 2–5m, 5–10m, 10–15m, 15m+) to isolate nearshore vs deep-water performance.</li>
                <li><b>IHO S-44 Compliance:</b> Evaluates Total Vertical Uncertainty (TVU) against <b>Order 1a</b> and <b>Order 2</b> hydrographic surveying standards.</li>
                <li><b>Interactive HTML Dashboard:</b> Generates a standalone scientific dashboard complete with interactive density scatter plots, residual distributions, and depth histogram diagnostics.</li>
            </ul>
            <br><b>Developer:</b> Mohamed Aly Nasef
        </div>
        """

    def helpString(self):
        return self.shortHelpString()

    def processAlgorithm(self, parameters, context, feedback):
        from ...core.processing.reporting import run_phase05_reporting

        return run_phase05_reporting(self, parameters, context, feedback)
