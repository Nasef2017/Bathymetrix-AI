import warnings

from qgis.core import (
    QgsProcessing, QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer, QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField, QgsProcessingParameterFolderDestination,
    QgsProcessingParameterEnum, QgsProcessingParameterNumber,
    QgsProcessingParameterFile, QgsProcessingParameterString,
)

warnings.filterwarnings("ignore")


class SDBPhase4Adaptive(QgsProcessingAlgorithm):
    INPUT_GLOBAL_RASTER = 'INPUT_GLOBAL_RASTER'
    INPUT_ORIGINAL_FEAT = 'INPUT_ORIGINAL_FEAT'
    INPUT_MASK = 'INPUT_MASK'
    INPUT_TRAIN = 'INPUT_TRAIN'
    FIELD_TRAIN = 'FIELD_TRAIN'

    SELECTED_ALGOS = 'SELECTED_ALGOS'
    OPTIMIZER_METHOD = 'OPTIMIZER_METHOD'
    COLLISION_HANDLING = 'COLLISION_HANDLING'
    N_ITERATIONS = 'N_ITERATIONS'
    MEDIAN_SIZE = 'MEDIAN_SIZE'

    OUTPUT_FOLDER = 'OUTPUT_FOLDER'
    LOG_FILE = 'LOG_FILE'

    PARAM_RF = 'PARAM_RF'
    PARAM_GB = 'PARAM_GB'
    PARAM_ET = 'PARAM_ET'
    PARAM_SVR = 'PARAM_SVR'
    PARAM_MLP = 'PARAM_MLP'

    MODEL_LIST = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'Extra Trees', 'Ridge', 'Lasso', 'ElasticNet', 'KNN', 'Decision Tree', 'MLP (Neural Net)', 'SVR']
    OPTIMIZER_LIST = ['Random Search', 'Grid Search', 'Bayesian Search']
    COLLISION_LIST = ['Keep All', 'Highest Conf', 'Closest', 'Hybrid', 'Strict Center']

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_GLOBAL_RASTER, 'Input Phase 3 Depth Map'))
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_ORIGINAL_FEAT, 'Input Original Feature Stack'))
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_MASK, 'Input Water Mask'))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_TRAIN, 'Adaptive Training Points'))
        self.addParameter(QgsProcessingParameterField(self.FIELD_TRAIN, 'Depth Field', parentLayerParameterName=self.INPUT_TRAIN, type=QgsProcessingParameterField.Numeric))

        self.addParameter(QgsProcessingParameterEnum(self.SELECTED_ALGOS, 'Refinement Algorithms', options=self.MODEL_LIST, allowMultiple=True, defaultValue=[0, 1]))
        self.addParameter(QgsProcessingParameterEnum(self.OPTIMIZER_METHOD, 'Optimizer', options=self.OPTIMIZER_LIST, defaultValue=0))
        self.addParameter(QgsProcessingParameterEnum(self.COLLISION_HANDLING, 'Collision Handling', options=self.COLLISION_LIST, defaultValue=0))

        self.addParameter(QgsProcessingParameterNumber(self.N_ITERATIONS, 'Optimization Iterations', type=QgsProcessingParameterNumber.Integer, defaultValue=10))
        self.addParameter(QgsProcessingParameterNumber(self.MEDIAN_SIZE, 'Output Median Filter Size', type=QgsProcessingParameterNumber.Integer, defaultValue=3))

        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, 'Output Folder'))
        self.addParameter(QgsProcessingParameterFile(self.LOG_FILE, 'Log File (Optional)', optional=True))

        self.addParameter(QgsProcessingParameterString(self.PARAM_RF, 'RF Params', defaultValue="", optional=True))
        self.addParameter(QgsProcessingParameterString(self.PARAM_GB, 'GB Params', defaultValue="", optional=True))
        self.addParameter(QgsProcessingParameterString(self.PARAM_ET, 'ET Params', defaultValue="", optional=True))
        self.addParameter(QgsProcessingParameterString(self.PARAM_SVR, 'SVR Params', defaultValue="", optional=True))
        self.addParameter(QgsProcessingParameterString(self.PARAM_MLP, 'MLP Params', defaultValue="", optional=True))

    def name(self): return 'sdb_phase4_adaptive'
    def displayName(self): return '4. SDB Module 04: Spatial Refinement'
    def group(self): return 'SDB Research Tools'
    def groupId(self): return 'sdb_tools'
    def createInstance(self): return SDBPhase4Adaptive()

    def shortHelpString(self):
        return """
        <div style="font-family: Arial, sans-serif; line-height: 1.2;">
            <h2 style="margin-bottom: 5px;">📍 <span style="color: #2E86C1;">SDB Module 04</span>: Spatial Refinement</h2>
            <p style="margin-top: 0; margin-bottom: 10px;">Corrects local biases and spatially varying errors using Residual Analysis (Stacking).</p>

            <b style="display: block; margin-bottom: 2px;">📉 Residual Analysis</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li>Calculates the error <i>(Residual = True Depth - Phase 3 Depth)</i> at training points.</li>
                <li>Uses <b>KNN Spatial Interpolation</b> to create a continuous "Error Grid" across the entire image.</li>
            </ul>

            <b style="display: block; margin-bottom: 2px;">📚 Stacked Learning</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li>Combines: <b>[Original Bands] + [Global Depth] + [Error Grid]</b>.</li>
                <li>Trains a secondary "Refinement Model" (using selected algorithms & custom hyperparameters) to predict the final, corrected bathymetry.</li>
            </ul>
        </div>
        """

    def helpString(self): return self.shortHelpString()

    def processAlgorithm(self, parameters, context, feedback):
        from ..sdb_pipeline.phases.phase04_spatial_retraining import run_phase04_spatial_retraining
        return run_phase04_spatial_retraining(self, parameters, context, feedback)
