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
    QgsProcessingParameterBoolean,
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
    STACK_COMPONENTS = "STACK_COMPONENTS"
    SELECTED_ALGOS = "SELECTED_ALGOS"
    OPTIMIZER_METHOD = "OPTIMIZER_METHOD"
    COLLISION_HANDLING = "COLLISION_HANDLING"
    N_ITERATIONS = "N_ITERATIONS"
    MEDIAN_SIZE = "MEDIAN_SIZE"
    FEATURE_CORR_THRESHOLD = "FEATURE_CORR_THRESHOLD"
    FEATURE_CORR_METHOD = "FEATURE_CORR_METHOD"

    OUTPUT_FOLDER = "OUTPUT_FOLDER"
    LOG_FILE = "LOG_FILE"
    OUTPUT_FINAL = "OUTPUT_FINAL"
    BEST_R2 = "BEST_R2"

    TRAIN_TEST_SPLIT = "TRAIN_TEST_SPLIT"
    RANDOM_STATE = "RANDOM_STATE"
    NUM_THREADS = "NUM_THREADS"
    CV_FOLDS = "CV_FOLDS"
    MAX_GPR_SAMPLES = "MAX_GPR_SAMPLES"
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
    PARAM_HUBER = "PARAM_HUBER"
    PARAM_XGB = "PARAM_XGB"
    PARAM_LGBM = "PARAM_LGBM"
    PARAM_CATBOOST = "PARAM_CATBOOST"

    ENABLE_ENSEMBLE = "ENABLE_ENSEMBLE"
    ENSEMBLE_METHOD = "ENSEMBLE_METHOD"
    ENSEMBLE_SIZE = "ENSEMBLE_SIZE"
    RESIDUAL_INTERP_METHOD = "RESIDUAL_INTERP_METHOD"
    KNN_NEIGHBORS = "KNN_NEIGHBORS"
    SPATIAL_CV = "SPATIAL_CV"

    FEATURE_CORR_THRESHOLDS = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]

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
        "Huber Regressor",
        "XGBoost",
        "LightGBM",
        "CatBoost",
    ]
    OPTIMIZER_LIST = ["Random Search", "Grid Search", "Bayesian Search"]
    COLLISION_LIST = ["Keep All", "Highest Conf", "Closest", "Hybrid", "Strict Center"]

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_GLOBAL_RASTER, "Input Phase 03 Depth Map"
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_ORIGINAL_FEAT, "Input Feature Stack (Phase 01)"
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
                self.STACK_COMPONENTS,
                "Features for Retraining",
                options=["Feature Stack (Phase 01)", "Phase 03 Depth Map", "Residual Error Grid"],
                allowMultiple=True,
                defaultValue=[1, 2],
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
            QgsProcessingParameterEnum(
                self.RESIDUAL_INTERP_METHOD,
                "📍 Spatial Residual Interpolation Method",
                options=["Standard KNN", "Robust KNN (Huber Weights)", "Gaussian Process / Kriging"],
                defaultValue=0,
            )
        )
        p_knn = QgsProcessingParameterNumber(
            self.KNN_NEIGHBORS,
            "📍 KNN Nearest Neighbors (K)",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=15,
            minValue=1,
            maxValue=100,
        )
        p_knn.setFlags(p_knn.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_knn)
        p_sp = QgsProcessingParameterBoolean(
            self.SPATIAL_CV,
            "🌍 Enable Spatial Block Cross-Validation",
            defaultValue=False,
        )
        p_sp.setFlags(p_sp.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_sp)

        p_ens = QgsProcessingParameterBoolean(
            self.ENABLE_ENSEMBLE,
            "⚙️ Enable Ensemble of Top Models",
            defaultValue=False,
        )
        p_ens.setFlags(p_ens.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_ens)

        p_ens_meth = QgsProcessingParameterEnum(
            self.ENSEMBLE_METHOD,
            "📊 Ensemble Blending Method",
            options=["Average", "Median", "Stacking", "Uncertainty-Weighted Fusion"],
            defaultValue=0,
        )
        p_ens_meth.setFlags(p_ens_meth.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_ens_meth)

        p_ens_size = QgsProcessingParameterNumber(
            self.ENSEMBLE_SIZE,
            "📊 Ensemble Size (Top N Models to blend)",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=3,
            minValue=2,
            maxValue=5,
        )
        p_ens_size.setFlags(p_ens_size.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_ens_size)
        self.addParameter(
            QgsProcessingParameterEnum(
                self.FEATURE_CORR_METHOD,
                "Feature Correlation Method",
                options=["Disabled", "Pearson (Linear)", "Spearman (Rank)", "Automatic-RANSAC", "Automatic-Random Forest"],
                defaultValue=3,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.FEATURE_CORR_THRESHOLD,
                "Feature Correlation Threshold",
                options=self.FEATURE_CORR_THRESHOLDS,
                defaultValue=2,
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

        p_cv = QgsProcessingParameterNumber(
            self.CV_FOLDS, "ML Cross-Validation Folds", type=QgsProcessingParameterNumber.Integer, defaultValue=5
        )
        p_cv.setFlags(p_cv.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_cv)

        p_gpr = QgsProcessingParameterNumber(
            self.MAX_GPR_SAMPLES, "Max GPR Training Samples", type=QgsProcessingParameterNumber.Integer, defaultValue=1500
        )
        p_gpr.setFlags(p_gpr.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_gpr)

        p_rf = QgsProcessingParameterString(self.PARAM_RF, "RF Params", defaultValue="'n_estimators':[100, 500], 'max_depth':[10, 30]", optional=True)
        p_rf.setFlags(p_rf.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_rf)

        p_gb = QgsProcessingParameterString(self.PARAM_GB, "GB Params", defaultValue="'n_estimators':[100, 300], 'learning_rate':[0.05, 0.1]", optional=True)
        p_gb.setFlags(p_gb.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_gb)

        p_et = QgsProcessingParameterString(self.PARAM_ET, "ET Params", defaultValue="'n_estimators':[100, 500], 'max_depth':[10, 30]", optional=True)
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

        p_huber = QgsProcessingParameterString(self.PARAM_HUBER, "Huber Params", defaultValue="'epsilon':[1.1, 1.35, 1.5]", optional=True)
        p_huber.setFlags(p_huber.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_huber)

        p_xgb = QgsProcessingParameterString(self.PARAM_XGB, "XGBoost Params", defaultValue="'n_estimators':[100, 200], 'max_depth':[4, 6], 'learning_rate':[0.05, 0.1]", optional=True)
        p_xgb.setFlags(p_xgb.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_xgb)

        p_lgbm = QgsProcessingParameterString(self.PARAM_LGBM, "LightGBM Params", defaultValue="'n_estimators':[100, 200], 'max_depth':[4, 6], 'learning_rate':[0.05, 0.1]", optional=True)
        p_lgbm.setFlags(p_lgbm.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_lgbm)

        p_cat = QgsProcessingParameterString(self.PARAM_CATBOOST, "CatBoost Params", defaultValue="'iterations':[100, 200], 'depth':[4, 6], 'learning_rate':[0.05, 0.1]", optional=True)
        p_cat.setFlags(p_cat.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_cat)

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
            <p style="margin-top: 0; margin-bottom: 10px;">Corrects local biases and spatially varying errors using Zero-Mean Centered Residual Analysis and Empirical Uncertainty Modeling.</p>

            <b style="display: block; margin-bottom: 2px;">📉 Zero-Mean Residual Analysis & LOO Huber Interpolation</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li>Subtracts mean residual offset (Zero-Mean Centering) to eliminate global drift.</li>
                <li>Uses <b>Leave-One-Out (LOO) Robust Huber Weighting</b> and <b>Smoothed IDW Distance Decay (1 / (d + 1.0))</b> to create a spike-free continuous "Error Surface".</li>
            </ul>

            <b style="display: block; margin-bottom: 2px;">📚 Stacked Learning & Empirical Uncertainty</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li>Combines: <b>[Original Bands] + [Global Depth] + [Error Grid]</b>.</li>
                <li>Trains a secondary "Refinement Model" to predict the final, corrected bathymetry.</li>
                <li>Generates an <b>Empirical Residual Uncertainty Map (95% Confidence)</b> to evaluate spatial prediction quality across every pixel.</li>
            </ul>
        </div>
        """

    def helpString(self):
        return self.shortHelpString()

    def processAlgorithm(self, parameters, context, feedback):
        from ...core.ml.evaluators import run_phase04_spatial_retraining

        return run_phase04_spatial_retraining(self, parameters, context, feedback)
