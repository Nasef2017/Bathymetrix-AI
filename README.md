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
The tool overcomes traditional bathymetry challenges like sun-glint and deep-water noise through a systematic 4-phase workflow.


<p align="center">
  <img src="Bathymetrix_AI _Workflow.png" width="60%">
</p>

🔬 **Scientific Methodology**
The toolkit follows a modular workflow where each phase is designed to improve the accuracy of depth retrieval.

**Phase 1: Automated Pre-processing, Feature Engineering & Deep Water OSW Filtering**

This phase prepares the satellite imagery by isolating the aquatic domain and correcting radiometric noise.  
**Sun-Glint Removal:** Removes surface reflections to reveal the seabed signal.  
**Water Segmentation:** Features a **new Advanced Water Mask** using 3-Indices (NDWI, MNDWI, NWI) alongside adaptive thresholding to accurately isolate the aquatic domain.
**Deep Water OSW Filtering:** Features a **Deep Water Filter** fully customized for ML algorithms, automatically calculating deep water statistical thresholds to isolate the Optically Shallow Water (OSW) zone where bathymetry is valid.
**Log-Ratio Features:** Transforms spectral bands into depth-sensitive features based on light attenuation laws.

***Key References:***  
Hedley et al. (2005) – Sun-glint correction.  
McFeeters (1996) & Otsu (1979) – Water masking and thresholding.  
Stumpf et al. (2003) – Log-ratio bathymetry model.

**Phase 2: Robust Altimetry Filtering**

To ensure high-quality training data, the tool filters ICESat-2 photon-counting data to remove outliers.
RANSAC Algorithm: Iteratively fits a linear model to identify high-confidence "inlier" depth points.
**Newly added Statistical filtering Algorithms**  

***Key Reference:***
Fischler & Bolles (1981) – Random Sample Consensus (RANSAC).  
Zhang et al., (2021) – LS Variance Fit, or Huber Variance Fit.  

**Phase 3: Automated Global Modeling (Auto-ML)**

Instead of using one algorithm, the tool benchmarks **11 different Machine Learning models** (e.g., Random Forest, Gradient Boosting, SVR, MLP) to find the best fit for your specific coastal area.  
**Hyperparameter Optimization:** Automatically tunes model settings for maximum performance.  
**Composite Scoring:** Ranks models based on R2, RMSE, and wMAPE.

***Key Reference:***
Bergstra & Bengio (2012) – Random search for optimization.

**Phase 4: Residual-Based Spatial Stacking**

This final phase corrects local errors that global models might miss by analyzing the "residuals" (differences) between predicted and observed depths.  
**Spatial Error Mapping:** Interpolates local errors using k-Nearest Neighbors (k-NN).  
**Adaptive Re-training:** Combines spectral data with error maps to produce a refined, high-accuracy final depth map.

***Key Reference:***
Alevizos (2020) – Residual analysis for shallow bathymetry.

📊 **Performance Metrics**

The tool evaluates results using three main standards:  
**R2** (Coefficient of Determination): Measures how well the model fits the data.  
**RMSE** (Root Mean Square Error): Measures the average vertical error in meters.  
**wMAPE** (Weighted Mean Absolute Percentage Error): Measures the relative error across different depth ranges.

🛠️ **Installation & Dependencies**  
Open **OSGeo4W Shell** (as Administrator) and run the following command to install all required libraries. This version is optimized for **QGIS 4.0 (Qt6)** and **NumPy 2.0** support:

```bash
pip install numpy pandas rasterio matplotlib seaborn scikit-learn>=1.5.0 scipy joblib scikit-optimize sliderule icepyx geopandas parquet netCDF4
```

📧 ***Contact & Citations***

**Author:** Mohamed Aly Nasef  
**Email:** Eng.m.nasef2017@gmail.com, Nasefm.aly@alexu.edu.eg  

Nasef M.Aly. (2026). Nasef2017/Bathymetrix-AI: Bathymetrix_AI V5.0 (v5.0). Zenodo. https://doi.org/10.5281/zenodo.20782533

🤖 **AI Acknowledgment**  
The development of the Bathymetrix-AI code, its logical structure, and the technical documentation were significantly enhanced and optimized using Google Gemini. The AI assisted in debugging complex workflows and ensuring the implementation follows best practices in data science.
