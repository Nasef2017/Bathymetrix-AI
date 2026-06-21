# Default Scikit-learn Hyperparameter Grids

DEFAULT_HYPERPARAMS_STR = {
    "Random Forest": "'n_estimators':[100, 300], 'max_depth':[10, 30]",
    "Gradient Boosting": "'n_estimators':[100, 300], 'learning_rate':[0.05, 0.1]",
    "Extra Trees": "'n_estimators':[100, 300], 'max_depth':[10, 30]",
    "SVR": "'C':[1, 10, 100], 'kernel':['rbf']",
    "MLP": "'hidden_layer_sizes':[(100,), (50, 50)]",
}
