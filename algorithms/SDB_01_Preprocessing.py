# SDB_01_Preprocessing.py
# ---------------------------------------------------------------------------
# MODULE 01: COMPREHENSIVE PRE-PROCESSING
# Features: Sunglint (Tunable), Masking (Otsu Tunable/Manual), Feature Gen
# Updates: +Feature Selection, +Band Math, +Mask Toggle, +Dynamic Raw Bands, +Select All Default
# Restored: Original Hedley & Original Otsu math for maximum accuracy
# ---------------------------------------------------------------------------

import os
import numpy as np
import rasterio
import warnings
from qgis.core import (
    QgsProcessing, QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer, QgsProcessingParameterVectorLayer,
    QgsProcessingParameterBand, QgsProcessingParameterFolderDestination,
    QgsProcessingParameterBoolean, QgsRasterLayer, QgsProject,
    QgsProcessingException, QgsProcessingParameterEnum, QgsProcessingParameterNumber,
    QgsProcessingParameterString, QgsProcessingParameterDefinition
)

try:
    from scipy.ndimage import binary_opening, binary_closing
    scipy_is_available = True
except ImportError:
    scipy_is_available = False

warnings.filterwarnings("ignore")

class SDBPhase1Preprocessing(QgsProcessingAlgorithm):
    INPUT_RASTER = 'INPUT_RASTER'
    OUTPUT_FOLDER = 'OUTPUT_FOLDER'
    
    COASTAL_BAND = 'COASTAL_BAND' 
    BLUE_BAND = 'BLUE_BAND'; GREEN_BAND = 'GREEN_BAND'
    RED_BAND = 'RED_BAND'; NIR_BAND = 'NIR_BAND'

    APPLY_SUNGLINT = 'APPLY_SUNGLINT'; NIR_BAND_SUNGLINT = 'NIR_BAND_SUNGLINT'
    DEEP_WATER_POLY = 'DEEP_WATER_POLY'
    
    SUNGLINT_PERCENTILE = 'SUNGLINT_PERCENTILE' 
    MASKING_METHOD = 'MASKING_METHOD'
    MANUAL_THRESHOLD = 'MANUAL_THRESHOLD'
    OTSU_ADJUSTMENT = 'OTSU_ADJUSTMENT'         
    MASK_KERNEL_SIZE = 'MASK_KERNEL_SIZE'       
    NUM_THREADS = 'NUM_THREADS'                 
    
    ENABLE_MASKING = 'ENABLE_MASKING'       
    FEATURE_SELECTION = 'FEATURE_SELECTION' 
    ENABLE_BAND_CALC = 'ENABLE_BAND_CALC'   
    BAND_MATH_FORMULA = 'BAND_MATH_FORMULA' 
    
    MASK_METHODS = ['Otsu (Automatic)', 'Manual Threshold']

    FEATURE_OPTIONS = [
        '[All Raw] All Bands from Input Image',  
        '[Log] Log(Coastal)',                    
        '[Log] Log(Blue)',                       
        '[Log] Log(Green)',                      
        '[Log] Log(Red)',                        
        '[Log] Log(NIR)',                        
        '[Ratio] Log(Blue) / Log(Green)',        
        '[Ratio] Log(Blue) / Log(Red)',          
        '[Ratio] Log(Coastal) / Log(Green)',     
        '[Custom] Band Math Calculator'          
    ]

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_RASTER, 'Input Satellite Image (Raw)'))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, 'Output Folder (For Module 1 Results)'))
        
        self.addParameter(QgsProcessingParameterBand(self.COASTAL_BAND, 'Coastal/Aerosol Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=1))
        self.addParameter(QgsProcessingParameterBand(self.BLUE_BAND, 'Blue Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=2))
        self.addParameter(QgsProcessingParameterBand(self.GREEN_BAND, 'Green Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=3))
        self.addParameter(QgsProcessingParameterBand(self.RED_BAND, 'Red Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=4))
        self.addParameter(QgsProcessingParameterBand(self.NIR_BAND, 'NIR Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=8))

        self.addParameter(QgsProcessingParameterBoolean(self.APPLY_SUNGLINT, 'Apply Sunglint Correction (Hedley)', defaultValue=True))
        self.addParameter(QgsProcessingParameterBand(self.NIR_BAND_SUNGLINT, 'Sunglint NIR Band', parentLayerParameterName=self.INPUT_RASTER, defaultValue=8))
        self.addParameter(QgsProcessingParameterVectorLayer(self.DEEP_WATER_POLY, 'Deep Water ROI (Optional)', optional=True))
        self.addParameter(QgsProcessingParameterNumber(self.SUNGLINT_PERCENTILE, 'Sunglint Deep Water Percentile', type=QgsProcessingParameterNumber.Double, defaultValue=1.0))
        
        self.addParameter(QgsProcessingParameterBoolean(self.ENABLE_MASKING, 'Enable Water Masking', defaultValue=True))
        self.addParameter(QgsProcessingParameterEnum(self.MASKING_METHOD, 'Water Masking Method', options=self.MASK_METHODS, defaultValue=0))
        self.addParameter(QgsProcessingParameterNumber(self.MANUAL_THRESHOLD, 'Manual Threshold', type=QgsProcessingParameterNumber.Double, defaultValue=0.0, optional=True))
        self.addParameter(QgsProcessingParameterNumber(self.OTSU_ADJUSTMENT, 'Otsu Threshold Adjustment', type=QgsProcessingParameterNumber.Double, defaultValue=0.0))
        self.addParameter(QgsProcessingParameterNumber(self.MASK_KERNEL_SIZE, 'Mask Cleanup Kernel Size', type=QgsProcessingParameterNumber.Integer, defaultValue=3))

        num_options = len(self.FEATURE_OPTIONS)
        default_selection = list(range(num_options))
        self.addParameter(QgsProcessingParameterEnum(self.FEATURE_SELECTION, 'Output Feature Stack Selection', options=self.FEATURE_OPTIONS, allowMultiple=True, defaultValue=default_selection, optional=True))
        self.addParameter(QgsProcessingParameterBoolean(self.ENABLE_BAND_CALC, 'Enable Custom Band Math', defaultValue=True))
        self.addParameter(QgsProcessingParameterString(self.BAND_MATH_FORMULA, 'Band Math Formula (e.g. (B2-B3)/(B2+B3))', defaultValue='', optional=True))
        self.addParameter(QgsProcessingParameterNumber(self.NUM_THREADS, 'Processing Threads', type=QgsProcessingParameterNumber.Integer, defaultValue=4))

    def name(self): return 'sdb_phase1_preprocessing'
    def displayName(self): return '1. SDB Module 01: Pre-processing'
    def group(self): return 'SDB Research Tools'
    def groupId(self): return 'sdb_tools'
    def createInstance(self): return SDBPhase1Preprocessing()
    def shortHelpString(self): return "<p><b>Feature Selection:</b> All features are selected by default. The '[All Raw]' option will add every band from the input image to the stack.</p>"
    def helpString(self): return self.shortHelpString()

    def processAlgorithm(self, parameters, context, feedback):
        n_threads = self.parameterAsInt(parameters, self.NUM_THREADS, context)
        os.environ['GDAL_NUM_THREADS'] = str(n_threads)
        
        out_dir = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        os.makedirs(out_dir, exist_ok=True)
        review_dir = os.path.join(out_dir, '1_Review_Intermediate_Bands'); os.makedirs(review_dir, exist_ok=True)
        
        p_glint = os.path.join(out_dir, '1_Sunglint_Corrected.tif')
        p_mask  = os.path.join(out_dir, '2_Water_Mask.tif')
        p_stack = os.path.join(out_dir, '3_Features_Stack.tif')

        input_layer = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        if not input_layer: raise QgsProcessingException("Input Raster Missing!")
        curr_img = input_layer.source()
        
        c_idx = self.parameterAsInt(parameters, self.COASTAL_BAND, context)
        b_idx = self.parameterAsInt(parameters, self.BLUE_BAND, context)
        g_idx = self.parameterAsInt(parameters, self.GREEN_BAND, context)
        r_idx = self.parameterAsInt(parameters, self.RED_BAND, context)
        n_idx = self.parameterAsInt(parameters, self.NIR_BAND, context)

        feedback.pushInfo(f"\n>>> MODULE 01 START (Threads: {n_threads})...")

        if self.parameterAsBool(parameters, self.APPLY_SUNGLINT, context):
            nir_g = self.parameterAsInt(parameters, self.NIR_BAND_SUNGLINT, context)
            deep_v = self.parameterAsVectorLayer(parameters, self.DEEP_WATER_POLY, context)
            perc = self.parameterAsDouble(parameters, self.SUNGLINT_PERCENTILE, context)
            feedback.pushInfo("   [1/3] Sunglint Correction...")
            self.run_hedley(curr_img, p_glint, nir_g, deep_v, c_idx, b_idx, g_idx, r_idx, n_idx, perc)
            curr_img = p_glint
            QgsProject.instance().addMapLayer(QgsRasterLayer(p_glint, "1_Sunglint_Corrected"))
        else:
            feedback.pushInfo("   [1/3] Sunglint Skipped.")

        if self.parameterAsBool(parameters, self.ENABLE_MASKING, context):
            masking_choice = self.parameterAsInt(parameters, self.MASKING_METHOD, context)
            k_size = self.parameterAsInt(parameters, self.MASK_KERNEL_SIZE, context)
            if self.MASK_METHODS[masking_choice] == 'Otsu (Automatic)':
                adj = self.parameterAsDouble(parameters, self.OTSU_ADJUSTMENT, context)
                feedback.pushInfo(f"   [2/3] Water Mask (Otsu)...")
                self.run_otsu_robust(curr_img, p_mask, g_idx, n_idx, adj, k_size, feedback)
            else:
                manual_val = self.parameterAsDouble(parameters, self.MANUAL_THRESHOLD, context)
                feedback.pushInfo(f"   [2/3] Water Mask (Manual)...")
                self.run_manual_mask(curr_img, p_mask, g_idx, n_idx, manual_val, k_size, feedback)
        else:
            feedback.pushInfo("   [2/3] Water Mask Disabled (Creating dummy mask)...")
            self.create_dummy_mask(curr_img, p_mask)
        
        QgsProject.instance().addMapLayer(QgsRasterLayer(p_mask, "2_Water_Mask"))

        feedback.pushInfo("   [3/3] Generating Features Stack...")
        selected_feats = self.parameterAsEnums(parameters, self.FEATURE_SELECTION, context)
        do_calc = self.parameterAsBool(parameters, self.ENABLE_BAND_CALC, context)
        calc_formula = self.parameterAsString(parameters, self.BAND_MATH_FORMULA, context)
        self.generate_features(curr_img, p_stack, review_dir, c_idx, b_idx, g_idx, r_idx, n_idx, selected_feats, do_calc, calc_formula, feedback)
                             
        QgsProject.instance().addMapLayer(QgsRasterLayer(p_stack, "3_Features_Stack"))
        feedback.pushInfo(f">>> MODULE 1 COMPLETED.")
        return {'OUTPUT_FEATURES': p_stack, 'OUTPUT_MASK': p_mask}

    def run_manual_mask(self, in_f, out_f, g_idx, n_idx, threshold, k_size, fb):
        with rasterio.open(in_f) as src:
            g = src.read(g_idx).astype('float32')
            n = src.read(n_idx).astype('float32')
            denom = (g + n); denom[denom == 0] = 1e-6
            ndwi = (g - n) / denom
            valid_mask = (g > 0) & (n > 0) & (g != -9999) & (n != -9999)
            water_mask = np.zeros(ndwi.shape, dtype='uint8')
            water_mask[(ndwi > threshold) & valid_mask] = 1
            if scipy_is_available and k_size > 0:
                kernel = np.ones((k_size, k_size))
                water_mask = binary_opening(water_mask, kernel).astype('uint8')
                water_mask = binary_closing(water_mask, kernel).astype('uint8')
            prof = src.profile; prof.update(count=1, dtype='uint8', nodata=0)
            with rasterio.open(out_f, 'w', **prof) as dst: dst.write(water_mask, 1)

    # --- RESTORED: Original Otsu Logic (from your very first code) ---
    def run_otsu_robust(self, in_f, out_f, g_idx, n_idx, adjustment, k_size, fb):
        with rasterio.open(in_f) as src:
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
                fb.pushInfo(f"      Calculated Otsu: {thresh:.4f} | Adj: {adjustment}")
                thresh += adjustment
            else:
                fb.pushWarning("      Not enough pixels for Otsu.")

            water_mask = np.zeros(ndwi.shape, dtype='uint8')
            water_mask[(ndwi > thresh) & valid_mask] = 1
            
            if scipy_is_available and k_size > 0:
                kernel = np.ones((k_size, k_size))
                water_mask = binary_opening(water_mask, kernel).astype('uint8')
                water_mask = binary_closing(water_mask, kernel).astype('uint8')
            
            prof = src.profile; prof.update(count=1, dtype='uint8', nodata=0)
            with rasterio.open(out_f, 'w', **prof) as dst: dst.write(water_mask, 1)

    # --- RESTORED: Original Hedley Logic (from your very first code) ---
    def run_hedley(self, in_f, out_f, nir_idx, poly_lyr, c, b, g, r, n, percentile):
        with rasterio.open(in_f) as src:
            prof = src.profile; prof.update(dtype='float32', nodata=-9999.0)
            d = src.read().astype('float32')
            
            nir_band_data = d[nir_idx-1]
            target_bands_idx = [c-1, b-1, g-1, r-1]
            
            valid_pixels = (nir_band_data > 0) & (nir_band_data != -9999)
            mask = np.zeros(nir_band_data.shape, dtype=bool)
            
            if np.any(valid_pixels):
                threshold = np.percentile(nir_band_data[valid_pixels], percentile)
                mask = (nir_band_data <= threshold) & valid_pixels
            
            nir_min = np.mean(nir_band_data[mask]) if np.any(mask) else 0
            d_corr = d.copy() 
            
            for band_i in target_bands_idx:
                if band_i < d.shape[0]:
                    if np.any(mask):
                        x = nir_band_data[mask]
                        y = d[band_i][mask]
                        if len(x) > 10:
                            slope = np.polyfit(x, y, 1)[0]
                            d_corr[band_i] = d[band_i] - slope * (nir_band_data - nir_min)
            
            for band_i in target_bands_idx:
                 if band_i < d.shape[0]:
                    d_corr[band_i][d_corr[band_i] < 0.0001] = 0.0001 
            
            with rasterio.open(out_f, 'w', **prof) as dst: dst.write(d_corr)

    # --- RETAINED FIX: SCALE=1000 to prevent log(negative) crashes in Stumpf ---
    def generate_features(self, in_f, out_f, review_dir, c, b, g, r, n, selected_indices_str, do_calc, formula, fb):
        with rasterio.open(in_f) as s:
            nbands = s.count
            selected_indices = [int(i) for i in selected_indices_str]
            
            c_val = s.read(c).astype('float32')
            b_val = s.read(b).astype('float32')
            g_val = s.read(g).astype('float32')
            r_val = s.read(r).astype('float32')
            n_val = s.read(n).astype('float32')
            mask_valid = (b_val > 0) & (g_val > 0)

            SCALE = 1000.0
            lc = np.log(np.clip(c_val * SCALE, 1e-6, None))
            lb = np.log(np.clip(b_val * SCALE, 1e-6, None))
            lg = np.log(np.clip(g_val * SCALE, 1e-6, None))
            lr = np.log(np.clip(r_val * SCALE, 1e-6, None))
            ln = np.log(np.clip(n_val * SCALE, 1e-6, None))
            
            with np.errstate(divide='ignore', invalid='ignore'):
                rbg = lb / lg; rbr = lb / lr; rcg = lc / lg
            
            custom_band = np.zeros_like(b_val)
            if do_calc and formula and 9 in selected_indices:
                try:
                    band_dict = {}
                    for i in range(1, nbands + 1):
                        band_dict[f"B{i}"] = s.read(i).astype('float32')
                    band_dict['np'] = np; band_dict['log'] = np.log
                    fb.pushInfo(f"      Calculating: {formula}")
                    res = eval(formula, {"__builtins__": None}, band_dict)
                    if isinstance(res, np.ndarray):
                        custom_band = res
                        custom_band[~mask_valid] = 0
                        custom_band[np.isinf(custom_band)] = 0
                    else: custom_band[:] = res
                except Exception as e: fb.pushWarning(f"Calc Error: {e}")

            calculated_feats_map = {
                1: lc, 2: lb, 3: lg, 4: lr, 5: ln,
                6: rbg, 7: rbr, 8: rcg, 9: custom_band
            }
            
            final_stack = []
            
            if 0 in selected_indices:
                fb.pushInfo(f"      Adding all {nbands} raw bands to stack...")
                for i in range(1, nbands + 1):
                    raw_band_data = s.read(i).astype('float32')
                    raw_band_data[~mask_valid] = -9999.0
                    final_stack.append(raw_band_data)
                    p_ind = s.profile; p_ind.update(count=1, dtype='float32', nodata=-9999.0)
                    with rasterio.open(os.path.join(review_dir, f"Raw_Band_{i}.tif"), 'w', **p_ind) as dst:
                        dst.write(raw_band_data, 1)

            for idx in selected_indices:
                if idx in calculated_feats_map:
                    data = calculated_feats_map[idx].copy()
                    data[np.isinf(data)] = 0; data[np.isnan(data)] = 0
                    data[~mask_valid] = -9999.0
                    final_stack.append(data)
                    name = self.FEATURE_OPTIONS[idx].replace("[", "").replace("]", "").replace(" ", "_").replace("/", "")
                    p_ind = s.profile; p_ind.update(count=1, dtype='float32', nodata=-9999.0)
                    with rasterio.open(os.path.join(review_dir, f"{name}.tif"), 'w', **p_ind) as dst:
                        dst.write(data, 1)

            if not final_stack:
                fb.pushWarning("Stack is empty! Please check selections.")
                return

            stack_arr = np.array(final_stack)
            prof = s.profile
            prof.update(count=len(final_stack), dtype='float32', nodata=-9999.0)
            with rasterio.open(out_f, 'w', **prof) as dst: dst.write(stack_arr)
            
    def create_dummy_mask(self, img, out):
        with rasterio.open(img) as s:
            p = s.profile; p.update(dtype='uint8', count=1, nodata=0)
            with rasterio.open(out, 'w', **p) as d: d.write(np.ones((s.height, s.width), 'uint8'), 1)