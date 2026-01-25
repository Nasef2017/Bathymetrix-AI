# SDB_01_Preprocessing.py
# ---------------------------------------------------------------------------
# MODULE 01: COMPREHENSIVE PRE-PROCESSING (ALL ORIGINAL BANDS + FEATURES)
# Part of SDB Master Workflow
# ---------------------------------------------------------------------------
# Operations:
# 1. Sunglint Correction (Hedley) - Now includes Coastal Band.
# 2. Robust Water Masking (Otsu or Manual) - USER CHOICE ADDED.
# 3. Feature Generation (Logs, Ratios) - Stacks ALL original bands + New Features.
# 4. Exports INDIVIDUAL bands & FINAL STACK.
# ---------------------------------------------------------------------------

import os
import numpy as np
import rasterio
import warnings
import shutil

# Ignore warnings
warnings.filterwarnings("ignore")

from qgis.core import (
    QgsProcessing, QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer, QgsProcessingParameterVectorLayer,
    QgsProcessingParameterBand, QgsProcessingParameterFolderDestination,
    QgsProcessingParameterBoolean, QgsRasterLayer, QgsProject,
    QgsProcessingException, QgsProcessingParameterEnum, QgsProcessingParameterNumber
)

try:
    from scipy.ndimage import binary_opening, binary_closing
    scipy_is_available = True
except ImportError:
    scipy_is_available = False

class SDBPhase1Preprocessing(QgsProcessingAlgorithm):
    # --- INPUTS ---
    INPUT_RASTER = 'INPUT_RASTER'
    OUTPUT_FOLDER = 'OUTPUT_FOLDER'
    
    COASTAL_BAND = 'COASTAL_BAND' 
    BLUE_BAND = 'BLUE_BAND'; GREEN_BAND = 'GREEN_BAND'
    RED_BAND = 'RED_BAND'; NIR_BAND = 'NIR_BAND'

    APPLY_SUNGLINT = 'APPLY_SUNGLINT'; NIR_BAND_SUNGLINT = 'NIR_BAND_SUNGLINT'
    DEEP_WATER_POLY = 'DEEP_WATER_POLY'
    
    # --- NEW PARAMETERS FOR MASKING ---
    MASKING_METHOD = 'MASKING_METHOD'
    MANUAL_THRESHOLD = 'MANUAL_THRESHOLD'
    
    MASK_METHODS = ['Otsu (Automatic)', 'Manual Threshold']

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_RASTER, 'Input Satellite Image (Raw)'))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, 'Output Folder (For Phase 1 Results)'))
        
        # Band Selection (For calculation purposes)
        self.addParameter(QgsProcessingParameterBand(self.COASTAL_BAND, 'Coastal/Aerosol Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=1))
        self.addParameter(QgsProcessingParameterBand(self.BLUE_BAND, 'Blue Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=2))
        self.addParameter(QgsProcessingParameterBand(self.GREEN_BAND, 'Green Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=3))
        self.addParameter(QgsProcessingParameterBand(self.RED_BAND, 'Red Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=4))
        self.addParameter(QgsProcessingParameterBand(self.NIR_BAND, 'NIR Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=8))

        # Correction Config
        self.addParameter(QgsProcessingParameterBoolean(self.APPLY_SUNGLINT, 'Apply Sunglint Correction (Hedley)', defaultValue=True))
        self.addParameter(QgsProcessingParameterBand(self.NIR_BAND_SUNGLINT, 'Sunglint NIR Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=8))
        self.addParameter(QgsProcessingParameterVectorLayer(self.DEEP_WATER_POLY, 'Deep Water ROI (Optional)', optional=True))
        
        # --- MODIFIED: Masking Config ---
        self.addParameter(QgsProcessingParameterEnum(self.MASKING_METHOD, 'Water Masking Method', options=self.MASK_METHODS, defaultValue=0))
        self.addParameter(QgsProcessingParameterNumber(self.MANUAL_THRESHOLD, 'Manual NDWI Threshold (if Manual selected)', type=QgsProcessingParameterNumber.Double, defaultValue=0.0, optional=True))

    def name(self): return 'sdb_phase1_preprocessing'
    def displayName(self): return '1. SDB Phase 1: Pre-processing (All Bands Support)'
    def group(self): return 'SDB Research Tools'
    def groupId(self): return 'sdb_tools'

    def createInstance(self): return SDBPhase1Preprocessing()

    def processAlgorithm(self, parameters, context, feedback):
        out_dir = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        os.makedirs(out_dir, exist_ok=True)
        
        # Sub-folder for individual review bands
        review_dir = os.path.join(out_dir, '1_Review_Intermediate_Bands')
        os.makedirs(review_dir, exist_ok=True)

        # Paths
        p_glint = os.path.join(out_dir, '1_Sunglint_Corrected.tif')
        p_mask  = os.path.join(out_dir, '2_Water_Mask.tif')
        p_stack = os.path.join(out_dir, '3_Features_Stack.tif')

        # Inputs
        input_layer = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        if not input_layer: raise QgsProcessingException("Input Raster Missing!")
        curr_img = input_layer.source()
        
        c_idx = self.parameterAsInt(parameters, self.COASTAL_BAND, context)
        b_idx = self.parameterAsInt(parameters, self.BLUE_BAND, context)
        g_idx = self.parameterAsInt(parameters, self.GREEN_BAND, context)
        r_idx = self.parameterAsInt(parameters, self.RED_BAND, context)
        n_idx = self.parameterAsInt(parameters, self.NIR_BAND, context)

        feedback.pushInfo("\n>>> MODULE 01: STARTING PRE-PROCESSING...")

        # 1. SUNGLINT
        if self.parameterAsBool(parameters, self.APPLY_SUNGLINT, context):
            feedback.pushInfo("   [1/3] Applying Hedley Sunglint Correction (Main Bands)...")
            nir_g = self.parameterAsInt(parameters, self.NIR_BAND_SUNGLINT, context)
            deep_v = self.parameterAsVectorLayer(parameters, self.DEEP_WATER_POLY, context)
            self.run_hedley(curr_img, p_glint, nir_g, deep_v, c_idx, b_idx, g_idx, r_idx, n_idx)
            curr_img = p_glint
            QgsProject.instance().addMapLayer(QgsRasterLayer(p_glint, "1_Sunglint_Corrected"))
        else:
            feedback.pushInfo("   [1/3] Sunglint Skipped.")

        # 2. WATER MASK (ROBUST OTSU OR MANUAL) --- MODIFIED LOGIC ---
        masking_choice = self.parameterAsInt(parameters, self.MASKING_METHOD, context)
        
        if self.MASK_METHODS[masking_choice] == 'Otsu (Automatic)':
            feedback.pushInfo("   [2/3] Generating Water Mask (Otsu Method)...")
            self.run_otsu_robust(curr_img, p_mask, g_idx, n_idx, feedback)
            QgsProject.instance().addMapLayer(QgsRasterLayer(p_mask, "2_Water_Mask"))
        
        elif self.MASK_METHODS[masking_choice] == 'Manual Threshold':
            manual_thresh_val = self.parameterAsDouble(parameters, self.MANUAL_THRESHOLD, context)
            feedback.pushInfo(f"   [2/3] Generating Water Mask (Manual Threshold: {manual_thresh_val})...")
            self.run_manual_mask(curr_img, p_mask, g_idx, n_idx, manual_thresh_val, feedback)
            QgsProject.instance().addMapLayer(QgsRasterLayer(p_mask, "2_Water_Mask"))
            
        else: # Fallback in case of an issue
            feedback.pushInfo("   [2/3] Masking Skipped or Method not recognized.")
            self.create_dummy_mask(curr_img, p_mask)


        # 3. FEATURES (EXPORT ALL BANDS + CALCULATED FEATURES)
        feedback.pushInfo("   [3/3] Generating Features Stack (All Bands + Ratios)...")
        self.generate_features(curr_img, p_stack, review_dir, c_idx, b_idx, g_idx, r_idx, n_idx)
        QgsProject.instance().addMapLayer(QgsRasterLayer(p_stack, "3_Features_Stack"))

        feedback.pushInfo(f">>> PHASE 1 COMPLETED. Check folder: {review_dir}")
        return {'OUTPUT_FEATURES': p_stack, 'OUTPUT_MASK': p_mask}

    # =========================================================================
    # INTERNAL LOGIC
    # =========================================================================

    def run_manual_mask(self, in_f, out_f, g_idx, n_idx, manual_threshold, fb):
        """ NEW FUNCTION for Manual Threshold Masking """
        with rasterio.open(in_f) as src:
            if g_idx > src.count or n_idx > src.count:
                raise QgsProcessingException(f"Band Index Error: Image has {src.count} bands, but Green={g_idx}, NIR={n_idx} requested.")

            g = src.read(g_idx).astype('float32')
            n = src.read(n_idx).astype('float32')
            denom = (g + n); denom[denom == 0] = 1e-6
            ndwi = (g - n) / denom
            valid_mask = (g > 0) & (n > 0) & (g != -9999) & (n != -9999) & (~np.isnan(ndwi)) & (~np.isinf(ndwi))
            
            fb.pushInfo(f"      Using Manual Threshold: {manual_threshold:.4f}")

            water_mask = np.zeros(ndwi.shape, dtype='uint8')
            water_mask[(ndwi > manual_threshold) & valid_mask] = 1
            
            if scipy_is_available:
                water_mask = binary_opening(water_mask, np.ones((3,3))).astype('uint8')
                water_mask = binary_closing(water_mask, np.ones((3,3))).astype('uint8')
            
            prof = src.profile; prof.update(count=1, dtype='uint8', nodata=0)
            with rasterio.open(out_f, 'w', **prof) as dst: dst.write(water_mask, 1)


    def run_otsu_robust(self, in_f, out_f, g_idx, n_idx, fb):
        with rasterio.open(in_f) as src:
            # Check indices to prevent out of bounds
            if g_idx > src.count or n_idx > src.count:
                raise QgsProcessingException(f"Band Index Error: Image has {src.count} bands, but Green={g_idx}, NIR={n_idx} requested.")

            g = src.read(g_idx).astype('float32')
            n = src.read(n_idx).astype('float32')
            denom = (g + n); denom[denom == 0] = 1e-6
            ndwi = (g - n) / denom
            valid_mask = (g > 0) & (n > 0) & (g != -9999) & (n != -9999) & (~np.isnan(ndwi)) & (~np.isinf(ndwi))
            valid_ndwi = ndwi[valid_mask]
            
            thresh = 0.0
            if valid_ndwi.size > 100:
                hist, bins = np.histogram(valid_ndwi, bins=256, range=(-1.0, 1.0))
                total = valid_ndwi.size
                sum_total = np.dot(np.arange(256), hist)
                sum_b, weight_b, max_var, thresh_idx = 0, 0, 0, 0
                for i in range(256):
                    weight_b += hist[i]
                    if weight_b == 0: continue
                    weight_f = total - weight_b
                    if weight_f == 0: break
                    sum_b += i * hist[i]
                    m_b = sum_b / weight_b; m_f = (sum_total - sum_b) / weight_f
                    var_b = weight_b * weight_f * (m_b - m_f)**2
                    if var_b > max_var: max_var = var_b; thresh_idx = i
                thresh = bins[thresh_idx]
                fb.pushInfo(f"      Otsu Threshold: {thresh:.4f}")
            else:
                fb.pushWarning("      Not enough valid pixels for Otsu. Defaulting to 0.0")

            water_mask = np.zeros(ndwi.shape, dtype='uint8')
            water_mask[(ndwi > thresh) & valid_mask] = 1
            
            if scipy_is_available:
                water_mask = binary_opening(water_mask, np.ones((3,3))).astype('uint8')
                water_mask = binary_closing(water_mask, np.ones((3,3))).astype('uint8')
            
            prof = src.profile; prof.update(count=1, dtype='uint8', nodata=0)
            with rasterio.open(out_f, 'w', **prof) as dst: dst.write(water_mask, 1)

    def run_hedley(self, in_f, out_f, nir_idx, poly_lyr, c, b, g, r, n):
        with rasterio.open(in_f) as src:
            prof = src.profile; prof.update(dtype='float32', nodata=-9999.0)
            d = src.read().astype('float32')
            
            # 1-based indices to 0-based
            nir_band_data = d[nir_idx-1]
            target_bands_idx = [c-1, b-1, g-1, r-1]
            
            # Identify deep water pixels
            valid_pixels = (nir_band_data > 0) & (nir_band_data != -9999)
            mask = np.zeros(nir_band_data.shape, dtype=bool)
            
            if np.any(valid_pixels):
                # Simple auto-detection of deep water (bottom 1% of NIR)
                threshold = np.percentile(nir_band_data[valid_pixels], 1.0)
                mask = (nir_band_data <= threshold) & valid_pixels
            
            nir_min = np.mean(nir_band_data[mask]) if np.any(mask) else 0
            
            d_corr = d.copy() # Copy all bands (even extra ones)
            
            # Only correct the specified SDB bands (C, B, G, R)
            for band_i in target_bands_idx:
                if band_i < d.shape[0]: # Check if band exists
                    if np.any(mask):
                        x = nir_band_data[mask]
                        y = d[band_i][mask]
                        if len(x) > 10:
                            slope = np.polyfit(x, y, 1)[0]
                            d_corr[band_i] = d[band_i] - slope * (nir_band_data - nir_min)
            
            # Ensure valid range for corrected bands
            for band_i in target_bands_idx:
                 if band_i < d.shape[0]:
                    d_corr[band_i][d_corr[band_i] < 0.0001] = 0.0001 
            
            with rasterio.open(out_f, 'w', **prof) as dst: dst.write(d_corr)

    def generate_features(self, in_f, out_f, review_dir, c, b, g, r, n):
        with rasterio.open(in_f) as s:
            nbands = s.count
            all_bands = [s.read(i).astype('float32') for i in range(1, nbands + 1)]
            
            c_val = s.read(c).astype('float32') if c <= nbands else np.zeros_like(all_bands[0])
            b_val = s.read(b).astype('float32') if b <= nbands else np.zeros_like(all_bands[0])
            g_val = s.read(g).astype('float32') if g <= nbands else np.zeros_like(all_bands[0])
            r_val = s.read(r).astype('float32') if r <= nbands else np.zeros_like(all_bands[0])
            n_val = s.read(n).astype('float32') if n <= nbands else np.zeros_like(all_bands[0])
            
            mask_valid = (c_val > 0) & (b_val > 0) & (g_val > 0) & (r_val > 0) & (n_val > 0)
            
            lc = np.full_like(c_val, 0.0); lc[mask_valid] = np.log(c_val[mask_valid])
            lb = np.full_like(b_val, 0.0); lb[mask_valid] = np.log(b_val[mask_valid])
            lg = np.full_like(g_val, 0.0); lg[mask_valid] = np.log(g_val[mask_valid])
            lr = np.full_like(r_val, 0.0); lr[mask_valid] = np.log(r_val[mask_valid])
            ln = np.full_like(n_val, 0.0); ln[mask_valid] = np.log(n_val[mask_valid])
            
            rbg = np.zeros_like(b_val); rbr = np.zeros_like(b_val); rcg = np.zeros_like(b_val)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                safe_g = (lg != 0); safe_r = (lr != 0)
                rbg[mask_valid & safe_g] = lb[mask_valid & safe_g] / lg[mask_valid & safe_g]
                rbr[mask_valid & safe_r] = lb[mask_valid & safe_r] / lr[mask_valid & safe_r]
                rcg[mask_valid & safe_g] = lc[mask_valid & safe_g] / lg[mask_valid & safe_g]
            
            for arr in [rbg, rbr, rcg]:
                arr[np.isinf(arr)] = 0.0
                arr[np.isnan(arr)] = 0.0

            features = {}
            for i, band_data in enumerate(all_bands):
                features[f"Band_{i+1}_Original"] = band_data
            
            calc_feats = {
                "Log_Coastal": lc, "Log_Blue": lb, "Log_Green": lg, "Log_Red": lr, "Log_NIR": ln,
                "Ratio_BG": rbg, "Ratio_BR": rbr, "Ratio_CG": rcg
            }
            features.update(calc_feats)
            
            prof = s.profile
            prof.update(count=len(features), dtype='float32', nodata=-9999.0)
            
            stack_list = []
            
            for name, data in features.items():
                save_data = data.copy()
                save_data[~mask_valid] = -9999.0
                stack_list.append(save_data)
                
                out_path = os.path.join(review_dir, f"{name}.tif")
                prof_ind = prof.copy(); prof_ind.update(count=1)
                with rasterio.open(out_path, 'w', **prof_ind) as dst_ind:
                    dst_ind.write(save_data, 1)

            stack_data = np.array(stack_list)
            with rasterio.open(out_f, 'w', **prof) as dst: dst.write(stack_data)

    def create_dummy_mask(self, img, out):
        with rasterio.open(img) as s:
            p = s.profile; p.update(dtype='uint8', count=1, nodata=0)
            with rasterio.open(out, 'w', **p) as d: d.write(np.ones((s.height, s.width), 'uint8'), 1)