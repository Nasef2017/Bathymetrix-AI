import warnings

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFile,
    QgsProcessingParameterString,
    QgsProcessingOutputRasterLayer,
    QgsProcessingOutputNumber,
    QgsProcessingParameterDefinition,
)

warnings.filterwarnings("ignore")


class SDBPhase4Adaptive(QgsProcessingAlgorithm):
    INPUT_GLOBAL_RASTER = "INPUT_GLOBAL_RASTER"
    INPUT_ORIGINAL_FEAT = "INPUT_ORIGINAL_FEAT"
    INPUT_MASK = "INPUT_MASK"
    INPUT_TRAIN = "INPUT_TRAIN"
    FIELD_TRAIN = "FIELD_TRAIN"

    SELECTED_ALGOS = "SELECTED_ALGOS"
    OPTIMIZER_METHOD = "OPTIMIZER_METHOD"
    COLLISION_HANDLING = "COLLISION_HANDLING"
    N_ITERATIONS = "N_ITERATIONS"
    MEDIAN_SIZE = "MEDIAN_SIZE"

    OUTPUT_FOLDER = "OUTPUT_FOLDER"
    LOG_FILE = "LOG_FILE"
    OUTPUT_FINAL = "OUTPUT_FINAL"
    BEST_R2 = "BEST_R2"

    TRAIN_TEST_SPLIT = "TRAIN_TEST_SPLIT"
    RANDOM_STATE = "RANDOM_STATE"
    NUM_THREADS = "NUM_THREADS"
    OUTPUT_FORMAT = "OUTPUT_FORMAT"

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

    MODEL_LIST = [
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
    ]
    OPTIMIZER_LIST = ["Random Search", "Grid Search", "Bayesian Search"]
    COLLISION_LIST = ["Keep All", "Highest Conf", "Closest", "Hybrid", "Strict Center"]

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_GLOBAL_RASTER, "Input Phase 3 Depth Map"
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_ORIGINAL_FEAT, "Input Original Feature Stack"
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_MASK, "Input Water Mask", optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_TRAIN, "Adaptive Training Points"
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_TRAIN,
                "Depth Field",
                parentLayerParameterName=self.INPUT_TRAIN,
                type=QgsProcessingParameterField.Numeric,
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.SELECTED_ALGOS,
                "Refinement Algorithms",
                options=self.MODEL_LIST,
                allowMultiple=True,
                defaultValue=[0, 1],
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.OPTIMIZER_METHOD,
                "Optimizer",
                options=self.OPTIMIZER_LIST,
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.COLLISION_HANDLING,
                "Collision Handling",
                options=self.COLLISION_LIST,
                defaultValue=0,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.N_ITERATIONS,
                "Optimization Iterations",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=10,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MEDIAN_SIZE,
                "Output Median Filter Size",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=3,
            )
        )

        self.addParameter(
            QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, "Output Folder")
        )
        self.addParameter(
            QgsProcessingParameterFile(
                self.LOG_FILE, "Log File (Optional)", optional=True
            )
        )

        self.addOutput(
            QgsProcessingOutputRasterLayer(
                self.OUTPUT_FINAL,
                "Output Final Adaptive Depth Map"
            )
        )
        self.addOutput(
            QgsProcessingOutputNumber(
                self.BEST_R2,
                "Best Adaptive Model R2 Score"
            )
        )

        p_split = QgsProcessingParameterNumber(
            self.TRAIN_TEST_SPLIT, "Training Data Ratio (e.g., 0.8 for 80%)", type=QgsProcessingParameterNumber.Double, defaultValue=0.8
        )
        p_split.setFlags(p_split.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_split)

        p_rs = QgsProcessingParameterNumber(
            self.RANDOM_STATE, "Random State", type=QgsProcessingParameterNumber.Integer, defaultValue=42
        )
        p_rs.setFlags(p_rs.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_rs)

        p_thr = QgsProcessingParameterNumber(
            self.NUM_THREADS, "Num Threads (n_jobs)", type=QgsProcessingParameterNumber.Integer, defaultValue=-1
        )
        p_thr.setFlags(p_thr.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_thr)

        p_fmt = QgsProcessingParameterEnum(
            self.OUTPUT_FORMAT, "Output Format", options=["float32", "float64", "uint16"], defaultValue=0
        )
        p_fmt.setFlags(p_fmt.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_fmt)

        p_rf = QgsProcessingParameterString(self.PARAM_RF, "RF Params", defaultValue="'n_estimators':[100, 300], 'max_depth':[10, 30]", optional=True)
        p_rf.setFlags(p_rf.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_rf)

        p_gb = QgsProcessingParameterString(self.PARAM_GB, "GB Params", defaultValue="'n_estimators':[100, 300], 'learning_rate':[0.05, 0.1]", optional=True)
        p_gb.setFlags(p_gb.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_gb)

        p_et = QgsProcessingParameterString(self.PARAM_ET, "ET Params", defaultValue="'n_estimators':[100, 300], 'max_depth':[10, 30]", optional=True)
        p_et.setFlags(p_et.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_et)

        p_svr = QgsProcessingParameterString(self.PARAM_SVR, "SVR Params", defaultValue="'C':[1, 10, 100], 'kernel':['rbf'], 'cache_size':[1000], 'max_iter':[20000]", optional=True)
        p_svr.setFlags(p_svr.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_svr)

        p_mlp = QgsProcessingParameterString(self.PARAM_MLP, "MLP Params", defaultValue="'hidden_layer_sizes':[(100,), (50, 50)], 'max_iter':[500]", optional=True)
        p_mlp.setFlags(p_mlp.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_mlp)

        p_ridge = QgsProcessingParameterString(self.PARAM_RIDGE, "Ridge Params", defaultValue="'alpha':[0.1, 1.0]", optional=True)
        p_ridge.setFlags(p_ridge.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_ridge)

        p_lasso = QgsProcessingParameterString(self.PARAM_LASSO, "Lasso Params", defaultValue="'alpha':[0.01, 0.1]", optional=True)
        p_lasso.setFlags(p_lasso.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_lasso)

        p_en = QgsProcessingParameterString(self.PARAM_ELASTICNET, "ElasticNet Params", defaultValue="'l1_ratio':[0.5]", optional=True)
        p_en.setFlags(p_en.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_en)

        p_knn = QgsProcessingParameterString(self.PARAM_KNN, "KNN Params", defaultValue="'n_neighbors':[5, 10]", optional=True)
        p_knn.setFlags(p_knn.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_knn)

        p_dt = QgsProcessingParameterString(self.PARAM_DT, "Decision Tree Params", defaultValue="'max_depth':[5, 10]", optional=True)
        p_dt.setFlags(p_dt.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_dt)

    def name(self):
        return "sdb_phase4_adaptive"

    def displayName(self):
        return "4. SDB Module 04: Spatial Refinement"

    def group(self):
        return "SDB Research Tools"

    def groupId(self):
        return "sdb_tools"

    def createInstance(self):
        return SDBPhase4Adaptive()

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

    def helpString(self):
        return self.shortHelpString()

    def processAlgorithm(self, parameters, context, feedback):
        from ...core.ml.evaluators import run_phase04_spatial_retraining

        return run_phase04_spatial_retraining(self, parameters, context, feedback)
