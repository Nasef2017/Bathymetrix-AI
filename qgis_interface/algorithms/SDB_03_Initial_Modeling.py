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


class SDBModule03(QgsProcessingAlgorithm):
    INPUT_STACK = "INPUT_STACK"
    INPUT_MASK = "INPUT_MASK"
    INPUT_POINTS = "INPUT_POINTS"
    FIELD_DEPTH = "FIELD_DEPTH"
    FIELD_WEIGHT = "FIELD_WEIGHT"
    SELECTED_ALGOS = "SELECTED_ALGOS"
    OPTIMIZER_METHOD = "OPTIMIZER_METHOD"
    COLLISION_HANDLING = "COLLISION_HANDLING"
    N_ITERATIONS = "N_ITERATIONS"
    MEDIAN_SIZE = "MEDIAN_SIZE"
    OUTPUT_FOLDER = "OUTPUT_FOLDER"
    LOG_FILE = "LOG_FILE"
    OUTPUT_DEPTH_MAP = "OUTPUT_DEPTH_MAP"
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
    COLLISION_LIST = [
        "Keep All Points",
        "Highest Confidence",
        "Closest to Pixel Center",
        "Hybrid",
        "Strict Center",
    ]

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(self.INPUT_STACK, "Input Feature Stack")
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_MASK, "Input Water Mask", optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_POINTS, "Cleaned Training Points"
            )
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
            QgsProcessingParameterField(
                self.FIELD_WEIGHT,
                "Weight Field",
                parentLayerParameterName=self.INPUT_POINTS,
                type=QgsProcessingParameterField.Numeric,
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.SELECTED_ALGOS,
                "Select Algorithms",
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
                self.MEDIAN_SIZE, "Output Median Filter Size", defaultValue=3
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
        return "sdb_03_initial_modeling"

    def displayName(self):
        return "3. SDB Module 03: Global Auto-ML"

    def group(self):
        return "SDB Research Tools"

    def groupId(self):
        return "sdb_tools"

    def createInstance(self):
        return SDBModule03()

    def shortHelpString(self):
        return """<h2>3. SDB Module 03: Global Auto-ML</h2>
        <p>This module performs automated machine learning (Auto-ML) to build a global bathymetry prediction model. It evaluates multiple algorithms to find the best fit for your depth points.</p>

        <h3>Inputs:</h3>
        <ul>
            <li><b>Input Feature Stack:</b> The multiband raster containing your predictors (bands, ratios, spatial features).</li>
            <li><b>Input Water Mask:</b> A binary raster (1=Water, 0=Land) to restrict predictions to water areas.</li>
            <li><b>Cleaned Training Points:</b> Your reference depth points.</li>
            <li><b>Depth & Weight Fields:</b> Fields from your points defining the true depth and optional weights for modeling.</li>
        </ul>

        <h3>Settings:</h3>
        <ul>
            <li><b>Select Algorithms:</b> Choose which ML algorithms to benchmark. The tool will pick the winner based on R2 and RMSE.</li>
            <li><b>Optimizer:</b> Method to tune hyperparameters (Random, Grid, or Bayesian Search).</li>
            <li><b>Collision Handling:</b> Determines how to handle multiple points falling within the same raster pixel (e.g., keep all, take the closest to center, or average them).</li>
        </ul>

        <h3>Outputs:</h3>
        <p>Saves all outputs to the selected folder, including:</p>
        <ul>
            <li><b>Initial Global Depth Map:</b> The predicted bathymetry raster using the winning model.</li>
            <li><b>Best Global Model (.pkl):</b> The trained model file to be used in subsequent modules.</li>
            <li><b>Benchmark CSV:</b> Detailed results for all tested algorithms.</li>
            <li><b>Actual Model Input Points (.shp):</b> The exact pixels/values used after resolving point collisions.</li>
        </ul>
        """

    def processAlgorithm(self, parameters, context, feedback):
        from ...core.ml.trainers import run_phase03_initial_modeling

        return run_phase03_initial_modeling(self, parameters, context, feedback)
