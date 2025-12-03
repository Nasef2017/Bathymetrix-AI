# MasterWorkflowAlgorithm.py

from qgis.core import (
    QgsProcessing, QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer, QgsProcessingParameterVectorLayer,
    QgsProcessingParameterFile, QgsProcessingParameterField,
    QgsProcessingParameterString, QgsProcessingParameterNumber,
    QgsProcessingParameterBand, QgsProcessingParameterFolderDestination,
    QgsProcessingParameterBoolean, QgsRasterLayer, QgsProject, QgsProcessingUtils,
    QgsApplication, QgsFeatureRequest, QgsVectorLayer
)
import os
import time
import numpy as np
import pandas as pd
import processing
import rasterio
from rasterio.mask import mask
from rasterio.features import rasterize
from sklearn.linear_model import LinearRegression

try:
    from scipy.ndimage import median_filter, binary_opening, binary_closing
    scipy_installed = True
except ImportError:
    scipy_installed = False

class MasterWorkflowAlgorithm(QgsProcessingAlgorithm):
    INPUT_RASTER = 'INPUT_RASTER'; APPLY_SUNGLINT = 'APPLY_SUNGLINT'
    NIR_BAND_SUNGLINT = 'NIR_BAND_SUNGLINT'; DEEP_WATER_POLYGON = 'DEEP_WATER_POLYGON'
    SUNGLINT_PERCENTILE = 'SUNGLINT_PERCENTILE'; APPLY_WATER_MASK = 'APPLY_WATER_MASK'
    GREEN_BAND = 'GREEN_BAND'; NIR_BAND = 'NIR_BAND'; SWIR_BAND = 'SWIR_BAND'
    INPUT_TRAINING_VEC = 'INPUT_TRAINING_VEC'; DEPTH_FIELD_TRAINING = 'DEPTH_FIELD_TRAINING'
    INPUT_TESTING_VEC = 'INPUT_TESTING_VEC'; DEPTH_FIELD_TESTING = 'DEPTH_FIELD_TESTING'
    RUN_RF = 'RUN_RF'; RUN_GB = 'RUN_GB'; RUN_ET = 'RUN_ET'; RUN_MLP = 'RUN_MLP'
    RUN_SVR = 'RUN_SVR'; RUN_KNN = 'RUN_KNN'; RUN_DT = 'RUN_DT'; RUN_ELASTIC = 'RUN_ELASTIC'
    RUN_RIDGE = 'RUN_RIDGE'; RUN_LASSO = 'RUN_LASSO'; RUN_LINEAR = 'RUN_LINEAR'
    RUN_BANDRATIO = 'RUN_BANDRATIO'; BANDRATIO_HIGH = 'BANDRATIO_HIGH'; BANDRATIO_LOW = 'BANDRATIO_LOW'
    N_ITERATIONS = 'N_ITERATIONS'; MEDIAN_FILTER_SIZE = 'MEDIAN_FILTER_SIZE'
    SAVE_MODELS = 'SAVE_MODELS'; OUTPUT_FOLDER = 'OUTPUT_FOLDER'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_RASTER, 'Input Satellite Image'))
        self.addParameter(QgsProcessingParameterBoolean(self.APPLY_SUNGLINT, 'Apply Sunglint Correction (Hedley et al.)', defaultValue=False))
        self.addParameter(QgsProcessingParameterBand(self.NIR_BAND_SUNGLINT, 'Sunglint - NIR Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=8))
        self.addParameter(QgsProcessingParameterVectorLayer(self.DEEP_WATER_POLYGON, 'Sunglint - Deep Water ROI Polygon (Manual - Recommended)', optional=True))
        self.addParameter(QgsProcessingParameterNumber(self.SUNGLINT_PERCENTILE, 'Sunglint - Auto-Method: Darkest Pixels Percentile (%)', type=QgsProcessingParameterNumber.Double, defaultValue=1.0, minValue=0.1, maxValue=10.0))
        self.addParameter(QgsProcessingParameterBoolean(self.APPLY_WATER_MASK, 'Apply Water Mask (Recommended)', defaultValue=True))
        self.addParameter(QgsProcessingParameterBand(self.GREEN_BAND, 'Water Masking - Green Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=3))
        self.addParameter(QgsProcessingParameterBand(self.NIR_BAND, 'Water Masking - NIR Band (for NDWI)', parentLayerParameterName=self.INPUT_RASTER, optional=True, defaultValue=8))
        self.addParameter(QgsProcessingParameterBand(self.SWIR_BAND, 'Water Masking - SWIR Band (for MNDWI)', parentLayerParameterName=self.INPUT_RASTER, optional=True))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_TRAINING_VEC, 'Training Points'))
        self.addParameter(QgsProcessingParameterField(self.DEPTH_FIELD_TRAINING, 'Depth Field (Training)', parentLayerParameterName=self.INPUT_TRAINING_VEC, type=QgsProcessingParameterField.Numeric))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_TESTING_VEC, 'Unseen Testing Points'))
        self.addParameter(QgsProcessingParameterField(self.DEPTH_FIELD_TESTING, 'Depth Field (Testing)', parentLayerParameterName=self.INPUT_TESTING_VEC, type=QgsProcessingParameterField.Numeric))
        self.addParameter(QgsProcessingParameterBoolean(self.RUN_RF, 'Run: RandomForest', defaultValue=True)); self.addParameter(QgsProcessingParameterBoolean(self.RUN_GB, 'Run: Gradient Boosting', defaultValue=True))
        self.addParameter(QgsProcessingParameterBoolean(self.RUN_ET, 'Run: Extra Trees', defaultValue=True)); self.addParameter(QgsProcessingParameterBoolean(self.RUN_MLP, 'Run: MLP (Neural Network)', defaultValue=False))
        self.addParameter(QgsProcessingParameterBoolean(self.RUN_SVR, 'Run: SVR (can be very slow)', defaultValue=False)); self.addParameter(QgsProcessingParameterBoolean(self.RUN_KNN, 'Run: KNN', defaultValue=True))
        self.addParameter(QgsProcessingParameterBoolean(self.RUN_DT, 'Run: Decision Tree', defaultValue=False)); self.addParameter(QgsProcessingParameterBoolean(self.RUN_ELASTIC, 'Run: ElasticNet', defaultValue=False))
        self.addParameter(QgsProcessingParameterBoolean(self.RUN_RIDGE, 'Run: Ridge', defaultValue=False)); self.addParameter(QgsProcessingParameterBoolean(self.RUN_LASSO, 'Run: Lasso', defaultValue=False))
        self.addParameter(QgsProcessingParameterBoolean(self.RUN_LINEAR, 'Run: Linear Regression', defaultValue=True))
        self.addParameter(QgsProcessingParameterBoolean(self.RUN_BANDRATIO, 'Run: Band Ratio (Stumpf)', defaultValue=True))
        self.addParameter(QgsProcessingParameterBand(self.BANDRATIO_HIGH, 'Band Ratio - High Reflectance Band (e.g., Green)', parentLayerParameterName=self.INPUT_RASTER, defaultValue=3))
        self.addParameter(QgsProcessingParameterBand(self.BANDRATIO_LOW, 'Band Ratio - Low Reflectance Band (e.g., Blue)', parentLayerParameterName=self.INPUT_RASTER, defaultValue=2))
        self.addParameter(QgsProcessingParameterNumber(self.N_ITERATIONS, 'Search Iterations (for complex models)', type=QgsProcessingParameterNumber.Integer, defaultValue=20, minValue=10))
        self.addParameter(QgsProcessingParameterNumber(self.MEDIAN_FILTER_SIZE, 'Apply Median Filter to ALL Results (0 to disable)', type=QgsProcessingParameterNumber.Integer, defaultValue=3, minValue=0))
        self.addParameter(QgsProcessingParameterBoolean(self.SAVE_MODELS, 'Save Trained Models', defaultValue=False))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, 'Main Output Folder'))

    def name(self): 
        return 'sdb_master_workflow'
    def displayName(self): 
        return 'SDB Master Workflow'
    def group(self): 
        return ''
    def groupId(self): 
        return ''
    def createInstance(self): 
        return MasterWorkflowAlgorithm()
    
    def shortHelpString(self):
        help_text = """
        <p><b>SDB Master Workflow for Automated Comparison</b></p>
        <p>This tool automates the entire SDB process, from optional preprocessing to running and evaluating multiple algorithms.</p>
        <p><b>Workflow Steps:</b></p>
        <ul>
        <li><b>1. Sunglint Correction (Optional):</b> Corrects for sun glint using the <b>Hedley et al. (2009)</b> method. Requires either a user-drawn deep water polygon (recommended) or uses an automatic method based on the darkest pixels in the selected NIR band.</li>
        <li><b>2. Water Masking (Optional):</b> Isolates water bodies using MNDWI (Green/SWIR bands - preferred) or NDWI (Green/NIR bands). The threshold to separate land and water is determined automatically using <b>Otsu's method</b>. The final mask is saved in the output folder.</li>
        <li><b>3. Model Training & Comparison:</b> The processed image is used to train and test selected algorithms.
            <p>For complex models (e.g., RandomForest, SVR), the tool performs an intelligent hyperparameter search using <b>Bayesian optimization with Gaussian Processes</b> to find the best model settings. The number of steps for this search is controlled by the 'Search Iterations' parameter.</p>
            The available algorithms include:
            <ul>
            <li>Machine Learning models with auto-tuning (e.g., RandomForest, SVR)</li>
            <li>Simple Linear Regression</li>
            <li>The log-ratio model by <b>Stumpf et al. (2003)</b></li>
            </ul>
        </li>
        <li><b>4. Evaluation & Output:</b> All models are evaluated against unseen test data. A final report is generated ranking the models, and the best-performing bathymetry raster is automatically loaded into the QGIS project.</li>
        </ul>
        """
        return help_text

    def processAlgorithm(self, parameters, context, feedback):
        provider_id = 'sdb_tools'
        provider = QgsApplication.processingRegistry().providerById(provider_id)
        if not provider:
            raise RuntimeError(f"Could not find SDB Tools provider ('{provider_id}'). Is the plugin enabled and loaded correctly?")
        
        input_raster_path = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context).source()
        training_points = self.parameterAsVectorLayer(parameters, self.INPUT_TRAINING_VEC, context)
        testing_points = self.parameterAsVectorLayer(parameters, self.INPUT_TESTING_VEC, context)
        depth_field_training = self.parameterAsString(parameters, self.DEPTH_FIELD_TRAINING, context)
        depth_field_testing = self.parameterAsString(parameters, self.DEPTH_FIELD_TESTING, context)
        n_iterations = self.parameterAsInt(parameters, self.N_ITERATIONS, context)
        main_output_folder = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        green_band_idx = self.parameterAsInt(parameters, self.GREEN_BAND, context)
        nir_band_idx = self.parameterAsInt(parameters, self.NIR_BAND, context)
        swir_band_idx = self.parameterAsInt(parameters, self.SWIR_BAND, context)
        filter_size = self.parameterAsInt(parameters, self.MEDIAN_FILTER_SIZE, context)
        br_high_idx = self.parameterAsInt(parameters, self.BANDRATIO_HIGH, context)
        br_low_idx = self.parameterAsInt(parameters, self.BANDRATIO_LOW, context)
        start_time = time.time()
        apply_sunglint = self.parameterAsBool(parameters, self.APPLY_SUNGLINT, context)
        nir_band_sunglint_idx = self.parameterAsInt(parameters, self.NIR_BAND_SUNGLINT, context)
        deep_water_layer = self.parameterAsVectorLayer(parameters, self.DEEP_WATER_POLYGON, context)
        sunglint_percentile = self.parameterAsDouble(parameters, self.SUNGLINT_PERCENTILE, context)
        apply_water_mask = self.parameterAsBool(parameters, self.APPLY_WATER_MASK, context)
        save_models = self.parameterAsBool(parameters, self.SAVE_MODELS, context)

        if (filter_size > 1 or apply_water_mask) and not scipy_installed:
            raise RuntimeError("Required library 'scipy' is not installed, but it's needed for water mask cleaning or median filtering.")
        
        os.makedirs(main_output_folder, exist_ok=True)
        processing_raster_path = input_raster_path
        water_mask_path = os.path.join(main_output_folder, 'final_water_mask.tif')

        if apply_sunglint:
            feedback.pushInfo("Step 0: Applying Sunglint Correction...")
            temp_folder = QgsProcessingUtils.tempFolder()
            glint_corrected_path = os.path.join(temp_folder, 'glint_corrected_image.tif')

            with rasterio.open(input_raster_path) as src:
                profile = src.profile
                all_bands = {i: src.read(i).astype(np.float32) for i in range(1, src.count + 1)}
                nir_band_data = all_bands[nir_band_sunglint_idx]
                sunglint_pixel_map = np.zeros(src.shape, dtype=np.int16)

                if deep_water_layer and deep_water_layer.featureCount() > 0:
                    feedback.pushInfo("Using manual polygon for deep water ROI.")
                    geoms = [feat.geometry() for feat in deep_water_layer.getFeatures()]
                    geoms_json = [g.asJson() for g in geoms]
                    deep_water_mask = rasterize(geoms_json, out_shape=src.shape, transform=src.transform, all_touched=True).astype(bool)
                    nir_deep_water_samples = nir_band_data[deep_water_mask]
                else:
                    feedback.pushInfo(f"No manual polygon provided. Using automatic method with {sunglint_percentile}% darkest pixels.")
                    green_band_for_mask = all_bands.get(green_band_idx)
                    nir_for_mask = all_bands.get(nir_band_idx)
                    if green_band_for_mask is None or nir_for_mask is None:
                        raise RuntimeError("Bands for automatic water masking (for sunglint) are not available.")
                    
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ndwi = (green_band_for_mask - nir_for_mask) / (green_band_for_mask + nir_for_mask)
                    water_mask = (ndwi > 0) & np.isfinite(nir_band_data)
                    nir_water_pixels = nir_band_data[water_mask]
                    if nir_water_pixels.size == 0:
                        raise RuntimeError("No water pixels found for automatic sunglint correction.")
                    threshold = np.percentile(nir_water_pixels, sunglint_percentile)
                    feedback.pushInfo(f"Darkest pixel threshold (NIR value): {threshold:.4f}")
                    deep_water_mask = (nir_band_data <= threshold) & water_mask
                    nir_deep_water_samples = nir_band_data[deep_water_mask]
                
                sunglint_pixel_map[deep_water_mask] = 1

                if len(nir_deep_water_samples) < 50:
                    raise RuntimeError(f"Could not find enough deep water pixels ({len(nir_deep_water_samples)} found). Try a larger polygon or higher percentile.")
                
                feedback.pushInfo(f"Using {len(nir_deep_water_samples)} deep water pixels for regression.")
                nir_min = np.min(nir_deep_water_samples)
                
                with rasterio.open(glint_corrected_path, 'w', **profile) as dst:
                    for i, band_data in all_bands.items():
                        if i == nir_band_sunglint_idx:
                            dst.write(band_data, i)
                            continue
                        feedback.pushInfo(f"  - Correcting Band {i}...")
                        band_deep_water_samples = band_data[deep_water_mask]
                        min_len = min(len(nir_deep_water_samples), len(band_deep_water_samples))
                        model = LinearRegression()
                        model.fit(nir_deep_water_samples[:min_len].reshape(-1, 1), band_deep_water_samples[:min_len])
                        slope = model.coef_[0]
                        if slope > 0:
                            corrected_band = band_data - slope * (nir_band_data - nir_min)
                            corrected_band[corrected_band < 0] = 0
                            dst.write(corrected_band, i)
                        else:
                            feedback.pushWarning(f"    Slope for Band {i} is not positive ({slope:.4f}). Writing original band.")
                            dst.write(band_data, i)

            feedback.pushInfo(f"Sunglint correction complete. Corrected image at: {glint_corrected_path}")
            processing_raster_path = glint_corrected_path
            
            sunglint_pixels_path = os.path.join(main_output_folder, 'sunglint_source_pixels.tif')
            sunglint_profile = profile.copy()
            sunglint_profile.update(dtype=rasterio.int16, count=1, compress='lzw', nodata=0)
            with rasterio.open(sunglint_pixels_path, 'w', **sunglint_profile) as dst:
                dst.write(sunglint_pixel_map, 1)
            feedback.pushInfo(f"Sunglint source pixels map saved to: {sunglint_pixels_path}")
            
        if apply_water_mask:
            feedback.pushInfo("Step 1: Creating and cleaning Water Mask...")
            
            with rasterio.open(processing_raster_path) as src:
                profile = src.profile
                green_band = src.read(green_band_idx).astype('float32')
                
                if swir_band_idx > 0:
                    feedback.pushInfo("Using MNDWI for water masking.")
                    swir_band = src.read(swir_band_idx).astype('float32')
                    numerator, denominator = green_band - swir_band, green_band + swir_band
                elif nir_band_idx > 0:
                    feedback.pushInfo("Using NDWI for water masking.")
                    nir_band = src.read(nir_band_idx).astype('float32')
                    numerator, denominator = green_band - nir_band, green_band + nir_band
                else: 
                    raise RuntimeError("Please provide either a NIR or SWIR band for water masking.")
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    water_index = numerator / denominator
                
                valid_pixels = water_index[np.isfinite(water_index)]
                hist, bin_edges = np.histogram(valid_pixels, bins=256, range=(-1.0, 1.0))
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
                weight1 = np.cumsum(hist)
                weight2 = np.cumsum(hist[::-1])[::-1]
                mean1 = np.cumsum(hist * bin_centers) / np.where(weight1 == 0, 1, weight1)
                mean2_rev = np.cumsum((hist * bin_centers)[::-1]) / np.where(weight2[::-1] == 0, 1, weight2[::-1])
                mean2 = mean2_rev[::-1]
                variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:])**2
                threshold = bin_centers[np.nanargmax(variance)]

                water_mask_raw = (water_index > threshold)
                structure = np.ones((3,3), dtype=bool)
                mask_opened = binary_opening(water_mask_raw, structure=structure)
                mask_cleaned = binary_closing(mask_opened, structure=structure).astype('int16')
                
                mask_profile = profile.copy()
                mask_profile.update(dtype=rasterio.int16, count=1, compress='lzw', nodata=0)
                with rasterio.open(water_mask_path, 'w', **mask_profile) as dst:
                    dst.write(mask_cleaned, 1)

            feedback.pushInfo(f"Final water mask saved to: {water_mask_path}")
            
            feedback.pushInfo("Applying water mask to the image...")
            masked_image_path = os.path.join(main_output_folder, 'masked_image.tif')
            with rasterio.open(processing_raster_path) as src:
                with rasterio.open(water_mask_path) as mask_src:
                    out_profile = src.profile.copy()
                    out_profile['dtype'] = 'float32'
                    nodata_val = -9999.0
                    out_profile['nodata'] = nodata_val
                    with rasterio.open(masked_image_path, 'w', **out_profile) as dst:
                        for i in range(1, src.count + 1):
                            band_data = src.read(i).astype('float32')
                            mask_data = mask_src.read(1)
                            band_data[mask_data == 0] = nodata_val
                            dst.write(band_data, i)
            
            processing_raster_path = masked_image_path
        else:
            feedback.pushInfo("Skipping water masking step.")
        
        all_algorithms = {
            self.RUN_RF: ('autopredict', f'{provider_id}:sdb_autopredict_rf', 'rf', 'RandomForest'),
            self.RUN_GB: ('autopredict', f'{provider_id}:sdb_autopredict_gb', 'gb', 'Gradient Boosting'),
            self.RUN_ET: ('autopredict', f'{provider_id}:sdb_autopredict_extratrees', 'extratrees', 'Extra Trees'),
            self.RUN_MLP: ('autopredict', f'{provider_id}:sdb_autopredict_mlp', 'mlp', 'MLP'),
            self.RUN_SVR: ('autopredict', f'{provider_id}:sdb_autopredict_svr', 'svr', 'SVR'),
            self.RUN_KNN: ('autopredict', f'{provider_id}:sdb_autopredict_knn', 'knn', 'KNN'),
            self.RUN_DT: ('autopredict', f'{provider_id}:sdb_autopredict_decisiontree', 'dt', 'Decision Tree'),
            self.RUN_ELASTIC: ('autopredict', f'{provider_id}:sdb_autopredict_elasticnet', 'elasticnet', 'ElasticNet'),
            self.RUN_RIDGE: ('autopredict', f'{provider_id}:sdb_autopredict_ridge', 'ridge', 'Ridge'),
            self.RUN_LASSO: ('autopredict', f'{provider_id}:sdb_autopredict_lasso', 'lasso', 'Lasso'),
            self.RUN_LINEAR: ('simple', f'{provider_id}:ml_linear', 'linear', 'Linear Regression'), 
            self.RUN_BANDRATIO: ('simple', f'{provider_id}:bandratio', 'bandratio', 'Band Ratio')
        }
        selected_algs_to_run = [info for param, info in all_algorithms.items() if self.parameterAsBool(parameters, param, context)]
        
        final_summary = []
        total_algs = len(selected_algs_to_run)
        if total_algs == 0: 
            raise RuntimeError("No algorithms were selected to run.")
        progress_per_alg = 85.0 / total_algs

        for i, (alg_type, alg_id, alg_name, display_name) in enumerate(selected_algs_to_run):
            if feedback.isCanceled(): break
            feedback.pushInfo(f"\n--- Running Algorithm {i+1}/{total_algs}: {display_name.upper()} ---")
            feedback.setProgress(int(10 + (i * progress_per_alg)))
            alg_output_folder = os.path.join(main_output_folder, alg_name)
            os.makedirs(alg_output_folder, exist_ok=True)
            params = {
                'INPUT_RASTER': processing_raster_path, 
                'INPUT_SAMPLES_VEC': training_points, 
                'DEPTH_FIELD_VEC': depth_field_training, 
                'OUTPUT_FOLDER': alg_output_folder
            }
            
            if alg_type == 'autopredict':
                params['N_ITERATIONS'] = n_iterations
                predicted_raster_filename = f"autopredict_{alg_name}_depth.tif"
                internal_report_filename = f"autopredict_{alg_name}_report.txt"
                if save_models:
                    model_output_path = os.path.join(alg_output_folder, f"trained_{alg_name}_model.joblib")
                    params['OUTPUT_MODEL'] = model_output_path
            else:
                predicted_raster_filename = f"{alg_name}_depth.tif"
                internal_report_filename = f"{alg_name}_report.txt"
                if 'bandratio' in alg_id:
                    if br_high_idx == 0 or br_low_idx == 0: 
                        feedback.pushWarning("Band Ratio was selected but its bands were not specified. Skipping.")
                        continue
                    params['BAND_HIGH_REF'] = br_high_idx 
                    params['BAND_LOW_REF'] = br_low_idx
            
            try: 
                processing.run(alg_id, params, context=context, feedback=feedback, is_child_algorithm=True)
            except Exception as e: 
                feedback.pushWarning(f"Algorithm {display_name} failed. Skipping. Error: {e}")
                continue
            
            predicted_raster_path_original = os.path.join(alg_output_folder, predicted_raster_filename)
            
            if apply_water_mask and os.path.exists(predicted_raster_path_original):
                feedback.pushInfo("Post-processing: Re-applying final water mask to prediction...")
                try:
                    with rasterio.open(water_mask_path) as mask_src:
                        water_mask_array = mask_src.read(1)
                    
                    with rasterio.open(predicted_raster_path_original, 'r+') as pred_src:
                        pred_profile = pred_src.profile
                        pred_data = pred_src.read(1)
                        nodata_val = pred_profile.get('nodata')
                        if nodata_val is None:
                            nodata_val = -9999.0
                            feedback.pushWarning(f"Predicted raster for {display_name} had no nodata value set. Using default -9999.0.")
                        
                        pred_data[water_mask_array == 0] = nodata_val
                        pred_src.write(pred_data, 1)
                except Exception as e:
                    feedback.pushWarning(f"Could not re-apply water mask to the result of {display_name}. Error: {e}")

            path_for_evaluation = predicted_raster_path_original
            if filter_size > 1 and os.path.exists(predicted_raster_path_original):
                feedback.pushInfo(f"Applying {filter_size}x{filter_size} Median Filter...")
                filtered_name = predicted_raster_filename.replace('.tif', f'_filtered.tif')
                predicted_raster_path_filtered = os.path.join(alg_output_folder, filtered_name)
                try:
                    with rasterio.open(predicted_raster_path_original) as src:
                        profile = src.profile
                        depth_raster = src.read(1)
                        nodata_value = profile.get('nodata', np.nan)
                        valid_data_mask = ~np.isnan(depth_raster)
                        if not np.isnan(nodata_value): valid_data_mask &= (depth_raster != nodata_value)
                        filtered_data = median_filter(depth_raster, size=filter_size, mode='reflect')
                        filtered_raster = depth_raster.copy()
                        filtered_raster[valid_data_mask] = filtered_data[valid_data_mask]
                        with rasterio.open(predicted_raster_path_filtered, 'w', **profile) as dst: 
                            dst.write(filtered_raster, 1)
                        path_for_evaluation = predicted_raster_path_filtered
                except Exception as e: 
                    feedback.pushWarning(f"Could not apply median filter. Evaluating original result. Error: {e}")

            eval_report_path = os.path.join(alg_output_folder, "FINAL_EVALUATION_REPORT.txt")
            summary_entry = {'Algorithm': display_name.upper(), 'Internal_R2': np.nan, 'Internal_RMSE': np.nan, 'Unseen_Data_R2': np.nan, 'Unseen_Data_RMSE': np.nan, 'Raster_Path': path_for_evaluation}
            internal_report_path = os.path.join(alg_output_folder, internal_report_filename)
            if os.path.exists(internal_report_path):
                with open(internal_report_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if "R-squared (R2):" in line: summary_entry['Internal_R2'] = float(line.split(':')[1].strip())
                        if "Root Mean Squared Error (RMSE):" in line: summary_entry['Internal_RMSE'] = float(line.split(':')[1].strip())
            
            if os.path.exists(path_for_evaluation):
                eval_params = {
                    'INPUT_PREDICTED_RASTER': path_for_evaluation, 
                    'INPUT_UNSEEN_VEC': testing_points, 
                    'DEPTH_FIELD_VEC': depth_field_testing, 
                    'OUTPUT_FILE': eval_report_path
                }
                try:
                    processing.run('sdb_tools:sdb_evaluate_model', eval_params, context=context, feedback=feedback, is_child_algorithm=True)
                    if os.path.exists(eval_report_path):
                        with open(eval_report_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                if "R-squared (R2):" in line: summary_entry['Unseen_Data_R2'] = float(line.split(':')[1].strip())
                                if "Root Mean Squared Error (RMSE):" in line: summary_entry['Unseen_Data_RMSE'] = float(line.split(':')[1].strip())
                except Exception as e: 
                    feedback.pushWarning(f"Evaluation failed for {display_name}. Error: {e}")
            else: 
                feedback.pushWarning(f"Predicted raster not found for {display_name}. Skipping evaluation.")
            
            final_summary.append(summary_entry)

        feedback.setProgress(95)
        feedback.pushInfo("\n--- WORKFLOW COMPLETE ---")
        summary_report_path = os.path.join(main_output_folder, "MASTER_SUMMARY_REPORT.txt")
        if not final_summary: 
            feedback.pushWarning("No algorithms were successfully run.")
            return {}
            
        summary_df = pd.DataFrame(final_summary)
        summary_df.dropna(subset=['Unseen_Data_R2', 'Unseen_Data_RMSE'], inplace=True)
        if summary_df.empty:
            feedback.pushWarning("No models were successfully evaluated on unseen data.")
            with open(summary_report_path, 'w', encoding='utf-8') as f: f.write("No models were successfully evaluated on unseen data.")
            return {}
            
        r2_range = summary_df['Unseen_Data_R2'].max() - summary_df['Unseen_Data_R2'].min()
        rmse_range = summary_df['Unseen_Data_RMSE'].max() - summary_df['Unseen_Data_RMSE'].min()
        if r2_range == 0 or np.isnan(r2_range): 
            r2_norm = pd.Series([1.0] * len(summary_df), index=summary_df.index)
        else: 
            r2_norm = (summary_df['Unseen_Data_R2'] - summary_df['Unseen_Data_R2'].min()) / r2_range
            
        if rmse_range == 0 or np.isnan(rmse_range): 
            rmse_norm = pd.Series([1.0] * len(summary_df), index=summary_df.index)
        else: 
            rmse_norm = 1 - ((summary_df['Unseen_Data_RMSE'] - summary_df['Unseen_Data_RMSE'].min()) / rmse_range)
            
        summary_df['Final_Score'] = 0.7 * r2_norm + 0.3 * rmse_norm
        summary_df = summary_df.sort_values(by='Final_Score', ascending=False, na_position='last')
        
        column_order = ['Algorithm', 'Final_Score', 'Unseen_Data_R2', 'Unseen_Data_RMSE', 'Internal_R2', 'Internal_RMSE', 'Raster_Path']
        existing_columns = [col for col in column_order if col in summary_df.columns]
        summary_df = summary_df[existing_columns]
        
        with open(summary_report_path, 'w', encoding='utf-8') as f:
            f.write("====================================================\n")
            f.write(" SDB Master Workflow - Full Comparison Report\n")
            f.write("====================================================\n\n")
            f.write(f"Workflow completed on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total time: {(time.time() - start_time)/60:.2f} minutes\n")
            f.write(f"Search iterations for complex models: {n_iterations}\n\n")
            f.write("Internal R2/RMSE: Performance on the model's internal test set.\n")
            f.write("Unseen Data R2/RMSE: Performance on the separate, unseen validation points.\n")
            f.write("Final_Score: Combined metric (70% R2, 30% RMSE) to rank models. Higher is better.\n\n")
            f.write(summary_df.to_string(index=False, float_format="%.4f"))
            
        feedback.pushInfo(f"All processes finished. Master summary report saved to: {summary_report_path}")
        
        if not summary_df.empty:
            best_result = summary_df.iloc[0]
            if pd.notna(best_result['Final_Score']) and os.path.exists(best_result['Raster_Path']):
                best_alg_name = best_result['Algorithm']
                best_raster_path = best_result['Raster_Path']
                feedback.pushInfo(f"\nLoading the best result into QGIS: {best_alg_name} (Final Score = {best_result['Final_Score']:.4f})")
                layer_name = f"Best Result - {best_alg_name}"
                rlayer = QgsRasterLayer(best_raster_path, layer_name)
                if rlayer.isValid(): 
                    QgsProject.instance().addMapLayer(rlayer)
                else: 
                    feedback.pushWarning(f"Could not load the best raster layer: {best_raster_path}")
            else: 
                feedback.pushWarning("Could not determine or find the best result to load.")
        
        feedback.setProgress(100)
        return {}