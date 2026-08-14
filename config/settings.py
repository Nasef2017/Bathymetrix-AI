# Bathymetrix-AI Global Settings

NODATA_VALUE = -9999.0

# Masking Methods
MASK_METHODS_NAMES = ["Otsu (Automatic NDWI)", "Manual NDWI Threshold", "3 Indices Equation (NDWI, MNDWI, NWI)", "Smart Hybrid (Dynamic Auto)"]

# Filtering Modes
FILTER_MODES_NAMES = ["Linear RANSAC", "LS Variance Fit", "Huber Variance Fit"]


# Model Lists
MODEL_LIST_NAMES = [
    "Linear Regression",
    "Random Forest",
    "Gradient Boosting",
    "Extra Trees",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "KNN",
    "Decision Tree",
    "MLP",
    "SVR",
]

# Optimizers
OPTIMIZER_LIST_NAMES = ["Random Search", "Grid Search", "Bayesian Search"]

# Collision Handling
COLLISION_LIST_NAMES = [
    "Keep All Points",
    "Highest Confidence",
    "Closest to Pixel Center",
    "Hybrid",
    "Strict Center",
]

# Feature Options
FEATURE_OPTIONS_NAMES = [
    "[All Raw] All Bands from Input Image",
    "[Log] Log(Coastal)",
    "[Log] Log(Blue)",
    "[Log] Log(Green)",
    "[Log] Log(Red)",
    "[Log] Log(NIR)",
    "[Ratio] Log(Blue) / Log(Green)",
    "[Ratio] Log(Blue) / Log(Red)",
    "[Ratio] Log(Coastal) / Log(Green)",
    "[Ratio] Log(Green) / Log(NIR)",
    "[Ratio] Log(Red) / Log(NIR)",
    "[Index] NDWI (Green - NIR) / (Green + NIR)",
    "[Custom] Band Math Calculator",
]
