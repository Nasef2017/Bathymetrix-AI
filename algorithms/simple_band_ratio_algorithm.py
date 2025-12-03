# simple_band_ratio_algorithm.py

from qgis.core import (
    QgsProcessing, QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer, QgsProcessingParameterVectorLayer,
    QgsProcessingParameterBand, QgsProcessingParameterField,
    QgsProcessingParameterFolderDestination,
    QgsProject, QgsCoordinateTransform, QgsPointXY, QgsCoordinateReferenceSystem
)
import os
import numpy as np
import pandas as pd
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

class SimpleBandRatioAlgorithm(QgsProcessingAlgorithm):
    INPUT_RASTER = 'INPUT_RASTER'
    INPUT_SAMPLES_VEC = 'INPUT_SAMPLES_VEC'
    DEPTH_FIELD_VEC = 'DEPTH_FIELD_VEC'
    BAND_HIGH_REF = 'BAND_HIGH_REF'
    BAND_LOW_REF = 'BAND_LOW_REF'
    OUTPUT_FOLDER = 'OUTPUT_FOLDER'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_RASTER, 'Input Raster'))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_SAMPLES_VEC, 'Training Points'))
        self.addParameter(QgsProcessingParameterField(self.DEPTH_FIELD_VEC, 'Depth Field', parentLayerParameterName=self.INPUT_SAMPLES_VEC, type=QgsProcessingParameterField.Numeric))
        self.addParameter(QgsProcessingParameterBand(self.BAND_HIGH_REF, 'High Reflectance Band (e.g., Green)', parentLayerParameterName=self.INPUT_RASTER))
        self.addParameter(QgsProcessingParameterBand(self.BAND_LOW_REF, 'Low Reflectance Band (e.g., Blue)', parentLayerParameterName=self.INPUT_RASTER))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, 'Output Folder'))

    def name(self):
        return 'bandratio'

    def displayName(self):
        return 'SDB Band Ratio (Stumpf)'

    def group(self):
        return 'SDB Tools-Algorithms'

    def groupId(self):
        return 'sdb_tools_algorithms'

    def createInstance(self):
        return SimpleBandRatioAlgorithm()

    def shortHelpString(self):
        return "Applies the Stumpf log-ratio algorithm using manually selected bands."

    def processAlgorithm(self, parameters, context, feedback):
        raster_layer = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        samples_layer = self.parameterAsVectorLayer(parameters, self.INPUT_SAMPLES_VEC, context)
        depth_field = self.parameterAsString(parameters, self.DEPTH_FIELD_VEC, context)
        band_high_idx = self.parameterAsInt(parameters, self.BAND_HIGH_REF, context)
        band_low_idx = self.parameterAsInt(parameters, self.BAND_LOW_REF, context)
        out_folder = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)

        os.makedirs(out_folder, exist_ok=True)
        
        # --- FIX: Get CRS from QGIS layer objects before anything else ---
        sample_crs = samples_layer.crs()
        raster_crs = raster_layer.crs()

        # Create transformation object only if CRS are different
        tr = None
        if sample_crs != raster_crs:
            feedback.pushInfo(f"Sample CRS ({sample_crs.authid()}) differs from Raster CRS ({raster_crs.authid()}). Creating transformation.")
            tr = QgsCoordinateTransform(sample_crs, raster_crs, QgsProject.instance())
        else:
            feedback.pushInfo("Sample and Raster CRS match. No transformation needed.")

        feedback.pushInfo("Reading raster bands and sample points...")
        with rasterio.open(raster_layer.source()) as src:
            profile = src.profile
            transform = src.transform
            band_high_data = src.read(band_high_idx).astype('float32')
            band_low_data = src.read(band_low_idx).astype('float32')

        X_vals, y_vals = [], []
        
        for feature in samples_layer.getFeatures():
            geom = feature.geometry()
            point = geom.asPoint()
            
            # Apply transformation if it was created
            if tr:
                point = tr.transform(point)

            col, row = ~transform * (point.x(), point.y())
            r_idx, c_idx = int(row), int(col)

            if 0 <= r_idx < profile['height'] and 0 <= c_idx < profile['width']:
                val_high = band_high_data[r_idx, c_idx]
                val_low = band_low_data[r_idx, c_idx]
                
                # Using Stumpf 2003 original formula: log(n*Ref_i) / log(n*Ref_j)
                # For simplicity and common practice, we use log(Ref_i) / log(Ref_j) or simple ratio
                # Let's stick to the log-transformed ratio which is more robust
                if val_high > 0 and val_low > 0:
                    log_ratio = np.log(val_high) / np.log(val_low)
                    if np.isfinite(log_ratio):
                        X_vals.append([log_ratio])
                        y_vals.append(feature[depth_field])

        if len(y_vals) < 10:
            raise RuntimeError(f"Could not extract enough valid sample points ({len(y_vals)} found). Check band selection and point locations.")

        feedback.pushInfo(f"Training model with {len(y_vals)} points...")
        X, y = np.array(X_vals), np.array(y_vals)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        m1 = model.coef_[0]
        m0 = model.intercept_
        
        feedback.pushInfo(f"Model trained. R²: {r2:.4f}, RMSE: {rmse:.4f}")

        feedback.pushInfo("Applying model to the entire raster...")
        with np.errstate(divide='ignore', invalid='ignore'):
            band_high_safe = np.where(band_high_data > 0, band_high_data, np.nan)
            band_low_safe = np.where(band_low_data > 0, band_low_data, np.nan)
            
            log_ratio_raster = np.log(band_high_safe) / np.log(band_low_safe)

        depth_raster = (m1 * log_ratio_raster) + m0
        
        raster_out_path = os.path.join(out_folder, 'bandratio_depth.tif')
        report_path = os.path.join(out_folder, 'bandratio_report.txt')

        profile.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=-9999.0)
        depth_raster[~np.isfinite(depth_raster)] = -9999.0
        with rasterio.open(raster_out_path, 'w', **profile) as dst:
            dst.write(depth_raster.astype(rasterio.float32), 1)
            
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("--- SDB Band Ratio (Stumpf) - Report ---\n\n")
            f.write(f"High Reflectance Band: {band_high_idx}\n")
            f.write(f"Low Reflectance Band: {band_low_idx}\n\n")
            f.write("--- Model Coefficients ---\n")
            f.write(f"m1 (slope): {m1:.4f}\n")
            f.write(f"m0 (intercept): {m0:.4f}\n\n")
            f.write("--- Performance on Internal Test Set ---\n")
            f.write(f"R-squared (R2): {r2:.4f}\n")
            f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
            
        feedback.pushInfo(f"Processing complete. Depth raster saved to: {raster_out_path}")
        return {}