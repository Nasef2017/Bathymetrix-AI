\[🛰️ SDB 🌊] Bathymetrix-AI

Bathymetrix-AI is a specialized QGIS research toolkit designed to derive high-precision Satellite-Derived Bathymetry (SDB). It integrates corrected multispectral imagery (Sentinel-2\_SR) with photon-counting LiDAR data (ICESat-2-ALTS24) using an advanced Machine Learning pipeline.

Unlike traditional regression methods, this tool employs a 5-stage adaptive workflow that combines physics-based preprocessing with "Super-Learner" modeling to overcome common challenges like sun-glint, and local depth bias.

🔬 Scientific Methodology \& Workflow

The toolkit operates as a sequential pipeline (The 5-Module System), designed to maximize depth retrieval accuracy:

1\. Advanced Pre-processing (Physics-Based)

Before modeling, the tool prepares the optical environment:

Sun-Glint Correction: Implements the Hedley et al. (2005) algorithm, enhanced to support the Coastal/Aerosol band alongside visible bands using the NIR channel.

Water Masking: Uses a robust, histogram-based Otsu Thresholding to automatically separate water from land/clouds without manual intervention.

Feature Engineering: Generates the Log-Ratio features (Stumpf et al., 2003) required for the inversion models.

2\. Data Filtering (Linear RANSAC)

ICESat-2 data often contains noise and surface photons.

The tool applies Linear RANSAC (Random Sample Consensus) to fit a robust trend line through the photon data.

Outliers are automatically rejected based on a dynamic residual threshold, ensuring only high-confidence bathymetric points are used for training.

3\. Global Auto-ML Modeling

Instead of relying on a single algorithm, the tool performs Competitive Benchmarking:

It trains 11 Machine Learning Algorithms simultaneously (including Random Forest, ExtraTrees, XGBoost, SVR, and MLPs).

It utilizes Hyperparameter Optimization to find the best-performing global model for the specific water conditions.

4\. Spatial Refinement (The Adaptive Phase)

Global models often miss local underwater geography.

This module applies Universal Spatial Residual Correction (Regression-Kriging).

It calculates the error residuals of the global model, maps them spatially, and "stacks" them back into the prediction. This effectively forces the model to learn local biases, significantly reducing the RMSE.

5\. Rigorous Validation

The tool calculates Stratified wMAPE (Weighted Mean Absolute Percentage Error).

It produces automated validation reports and scatter plots comparing the "Global" vs. "Refined" output against unseen test data.

🛠️ Dependencies

This plugin requires specific scientific libraries.

Open OSGeo4W Shell (as Administrator) and run:

code

Bash

pip install scikit-learn,rasterio,scikit-optimize,scipy,pandas,numpy,matplotlib,sliderule,icepyx,geopandas

💻 Usage

Use the "SDB Master Workflow" script to run the entire pipeline end-to-end:

Input: Sentinel-2\_SR Raster \& ICESat-2 Shapefile.

Configuration: Select bands and target algorithms.

Execution: The tool handles CRS reprojection, training, and export automatically.

Output: Returns the final Refined\_Depth\_Map.tif and a statistical report.

📧 Contact

Author: Nasef M. Aly

Email: Eng.m.nasef2017@gmail.com

📄Citation

Nasef M.Aly. (2025). Nasef2017/Bathymetrix-AI: Bathymetrix_AI V3.0 (v3.0). Zenodo.
https://doi.org/10.5281/zenodo.18097758

📄 License

Copyright 2025 Nasef M. Aly

Licensed under the Apache License, Version 2.0 (the "License");

you may not use this file except in compliance with the License.

You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software

distributed under the License is distributed on an "AS IS" BASIS,

WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and

limitations under the License.





Acknowledgements and Tool Development
The development of Bathymetrix-AI  plugin was significantly accelerated by leveraging advanced AI language models.
Google's Gemini Pro: Utilized for its strong capabilities in code generation, logical structuring of complex workflows, and debugging.

