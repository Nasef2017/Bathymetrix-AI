<table>
<tr>
<td>

<img src="icon.png" width="100">

</td>
<td>

**Bathymetrix-AI: Advanced SDB Toolkit**

</td>
</tr>
</table>


**Bathymetrix-AI** is a professional QGIS research toolkit designed for high-precision Satellite-Derived Bathymetry (SDB). It integrates Sentinel-2 multispectral imagery with ICESat-2 (ATL24) LiDAR data using a modular and adaptive Machine Learning pipeline.
The tool overcomes traditional bathymetry challenges like sun-glint and deep-water noise through a systematic 5-phase workflow.


<p align="center">
  <img src="Bathymetrix_AI _Workflow.png" width="60%">
</p>

🔬 **Scientific Methodology**
The toolkit follows a modular workflow where each phase is designed to improve the accuracy of depth retrieval.

**Phase 01: Advanced Pre-processing**
This phase prepares the satellite imagery by isolating the aquatic domain and correcting radiometric noise.  
**Sun-Glint Removal:** Removes surface reflections to reveal the seabed signal (Hedley et al., 2005).  
**Water Segmentation:** Advanced Water Masking using 3-Indices (NDWI, MNDWI, NWI) alongside adaptive thresholding to accurately isolate the aquatic domain.
**Deep Water OSW Filtering:** Deep Water Filter customized for ML algorithms, automatically calculating deep water statistical thresholds to isolate the Optically Shallow Water (OSW) zone where bathymetry is valid.
**Log-Ratio Features:** Transforms spectral bands into depth-sensitive features based on light attenuation laws, computing physics-based Log-Ratio features.

**Phase 02: Robust Filtering**
To ensure high-quality training data, the tool filters altimetry data to remove outliers and environmental noise.
**Noise Removal:** Iteratively identifies high-confidence "inlier" depth points using Linear RANSAC, LS Variance Fit, or Huber Variance Fit (Zhang et al., 2021).

**Phase 03: Global Auto-ML & Feature Analysis**
Instead of using one algorithm, the tool implements an automated machine learning workflow.
**Feature Analysis:** Optionally drops weak bands based on their Pearson or Spearman correlation with the target depth.
**Algorithm Benchmarking:** Evaluates 15+ different Machine Learning models (e.g., Random Forest, Gradient Boosting, XGBoost, CatBoost, SVR, MLP) to find the best fit for specific coastal areas.
**Hyperparameter Optimization:** Automatically tunes model settings via Random Search, Grid Search, or Bayesian Optimization (Bergstra & Bengio, 2012).
**Spatial Cross-Validation:** Independent spatial block cross-validation control to evaluate base models.
**Customization:** Fully customizable hyperparameters for precise fine-tuning.

**Phase 04: Adaptive Refinement**
This phase corrects local errors that global models might miss by analyzing the "residuals" (differences) between predicted and observed depths.  
**Spatial Error Mapping:** Spatially localized corrections and residual analysis (Alevizos, 2020) to fix local geographic biases.
**Spatial Cross-Validation:** Independent spatial block cross-validation control to evaluate residual modeling.
**Adaptive Re-training:** Combines spectral data with error maps to produce a refined, high-accuracy final depth map.

**Phase 05: Validation & Reporting**
**Independent Accuracy Assessment:** Validates the finalized models on unseen test data to ensure robust accuracy reporting and scientific validity.

📊 **Performance Metrics**

The tool evaluates results using three main standards:  
**R2** (Coefficient of Determination): Measures how well the model fits the data.  
**RMSE** (Root Mean Square Error): Measures the average vertical error in meters.  
**wMAPE** (Weighted Mean Absolute Percentage Error): Measures the relative error across different depth ranges.

🛠️ **Installation & Dependencies**  
Open **OSGeo4W Shell** (as Administrator) and run the following command to install all required libraries. This version is optimized for **QGIS 4.0 (Qt6)** and **NumPy 2.0** support:

```bash
pip install numpy pandas rasterio matplotlib seaborn scikit-learn>=1.5.0 scipy joblib scikit-optimize sliderule icepyx geopandas parquet netCDF4 xgboost lightgbm catboost optuna
```

📧 ***Contact & Citations***

**Author:** Mohamed Aly Nasef  
**Email:** Eng.m.nasef2017@gmail.com, Nasefm.aly@alexu.edu.eg  

Nasef M.Aly. (2026). Nasef2017/Bathymetrix-AI: Bathymetrix_AI V6.0 (v6.0). Zenodo. https://doi.org/10.5281/zenodo.21245364

🤖 **AI Acknowledgment**  
The development of the Bathymetrix-AI code, its logical structure, and the technical documentation were significantly enhanced and optimized using Google Gemini. The AI assisted in debugging complex workflows and ensuring the implementation follows best practices in data science.
