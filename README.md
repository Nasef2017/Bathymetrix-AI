🛰️ Bathymetrix-AI: Advanced SDB Toolkit

Bathymetrix-AI is a professional QGIS research toolkit designed for high-precision Satellite-Derived Bathymetry (SDB). It integrates Sentinel-2 multispectral imagery with ICESat-2 (ATL24) LiDAR data using a modular and adaptive Machine Learning pipeline.
The tool overcomes traditional bathymetry challenges like sun-glint, water turbidity, and local depth biases through a systematic 4-phase workflow.

🔬 Scientific Methodology
The toolkit follows a modular workflow where each phase is designed to improve the accuracy of depth retrieval.

Phase 1: Automated Pre-processing & Feature Engineering

This phase prepares the satellite imagery by isolating the aquatic domain and correcting radiometric noise.
Sun-Glint Removal: Removes surface reflections to reveal the seabed signal.
Water Segmentation: Uses an adaptive threshold to mask land and clouds.
Log-Ratio Features: Transforms spectral bands into depth-sensitive features based on light attenuation laws.

Key References:

Hedley et al. (2005) – Sun-glint correction.
McFeeters (1996) & Otsu (1979) – Water masking and thresholding.
Stumpf et al. (2003) – Log-ratio bathymetry model.

Phase 2: Robust Altimetry Filtering

To ensure high-quality training data, the tool filters ICESat-2 photon-counting data to remove outliers (e.g., noise from waves or turbid water).
RANSAC Algorithm: Iteratively fits a linear model to identify high-confidence "inlier" depth points.

Key Reference:
Fischler & Bolles (1981) – Random Sample Consensus (RANSAC).

Phase 3: Automated Global Modeling (Auto-ML)

Instead of using one algorithm, the tool benchmarks 11 different Machine Learning models (e.g., Random Forest, Gradient Boosting, SVR, MLP) to find the best fit for your specific coastal area.
Hyperparameter Optimization: Automatically tunes model settings for maximum performance.
Composite Scoring: Ranks models based on R2, RMSE, and wMAPE.

Key Reference:
Bergstra & Bengio (2012) – Random search for optimization.

Phase 4: Residual-Based Spatial Stacking
This final phase corrects local errors that global models might miss by analyzing the "residuals" (differences) between predicted and observed depths.
Spatial Error Mapping: Interpolates local errors using k-Nearest Neighbors (k-NN).
Adaptive Re-training: Combines spectral data with error maps to produce a refined, high-accuracy final depth map.

Key Reference:
Alevizos (2020) – Residual analysis for shallow bathymetry.

📊 Performance Metrics

The tool evaluates results using three main standards:
R2 (Coefficient of Determination): Measures how well the model fits the data.
RMSE (Root Mean Square Error): Measures the average vertical error in meters.
wMAPE (Weighted Mean Absolute Percentage Error): Measures the relative error across different depth ranges.

🛠️ Installation & Dependencies

Open OSGeo4W Shell (as Administrator) and run:
code
Bash
pip install numpy pandas rasterio matplotlib seaborn scikit-learn scipy joblib scikit-optimize sliderule icepyx geopandas

📧 Contact & Citation

Author: Nasef M. Aly
Email: Eng.m.nasef2017@gmail.com, Nasefm.aly@alexu.edu.eg
Citation: Nasef M. Aly. (2025). Bathymetrix-AI V3.0. Zenodo.

🤖 AI Acknowledgment

The development of the Bathymetrix-AI code, its logical structure, and the technical documentation were significantly enhanced and optimized using Google Gemini. The AI assisted in debugging complex workflows and ensuring the implementation follows best practices in data science.
