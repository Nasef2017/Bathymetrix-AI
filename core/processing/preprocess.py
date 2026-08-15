import json
import os
import ast
import operator

import numpy as np
import rasterio

from rasterio.features import geometry_mask

from qgis.core import (
    QgsProcessingException,
    QgsCoordinateTransform,
    QgsProject,
)

_SAFE_OP = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(expr, variables):
    """Safely evaluate a mathematical expression from string using AST."""
    node = ast.parse(expr, mode="eval").body

    def _eval_node(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif getattr(ast, "Num", None) and isinstance(node, getattr(ast, "Num")):
            return node.n
        elif isinstance(node, ast.Name):
            if node.id in variables:
                return variables[node.id]
            raise ValueError(f"Variable '{node.id}' not permitted")
        elif isinstance(node, ast.BinOp):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            op = _SAFE_OP.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op)}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = _eval_node(node.operand)
            op = _SAFE_OP.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op)}")
            return op(operand)
        elif isinstance(node, ast.Call):
            func = _eval_node(node.func)
            if not callable(func):
                raise ValueError("Invalid function call")
            args = [_eval_node(a) for a in node.args]
            return func(*args)
        elif isinstance(node, ast.Attribute):
            val = _eval_node(node.value)
            if val is np:
                if node.attr in [
                    "log",
                    "exp",
                    "sin",
                    "cos",
                    "tan",
                    "sqrt",
                    "power",
                    "where",
                    "clip",
                    "arcsin",
                    "arccos",
                    "arctan",
                    "pi",
                    "e",
                ]:
                    return getattr(np, node.attr)
            raise ValueError(f"Unsupported attribute access: {node.attr}")
        else:
            raise ValueError(f"Unsupported expression node: {type(node)}")

    return _eval_node(node)


try:
    from scipy.ndimage import binary_opening, binary_closing

    scipy_is_available = True
except ImportError:
    scipy_is_available = False

MASK_METHODS = ["Otsu (Automatic NDWI)", "Manual NDWI Threshold", "3 Indices Equation (NDWI, MNDWI, NWI)", "Smart Hybrid (Dynamic Auto)"]

FEATURE_OPTIONS = [
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


def run_polygon_mask(in_f, out_f, raster_layer, poly_layer, fb):
    source_crs = poly_layer.crs()
    dest_crs = raster_layer.crs()
    transform = QgsCoordinateTransform(source_crs, dest_crs, QgsProject.instance())

    geoms = []
    for feat in poly_layer.getFeatures():
        geom = feat.geometry()
        geom.transform(transform)
        geoms.append(json.loads(geom.asJson()))

    if not geoms:
        raise QgsProcessingException("The provided water polygon layer is empty!")

    with rasterio.open(in_f) as src:
        mask_arr = geometry_mask(
            geoms,
            out_shape=(src.height, src.width),
            transform=src.transform,
            invert=True,
        ).astype("uint8")
        prof = src.profile
        prof.update(count=1, dtype="uint8", nodata=0)
        with rasterio.open(out_f, "w", **prof) as dst:
            dst.write(mask_arr, 1)


def run_watermasking_plugin(in_f, out_f, b_idx, g_idx, r_idx, n_idx, s_idx, k_size, fb):
    import numpy as np
    import rasterio
    from rasterio.features import sieve
    
    with rasterio.open(in_f) as src:
        nbands = src.count
        b_safe = min(max(1, b_idx), nbands)
        g_safe = min(max(1, g_idx), nbands)
        r_safe = min(max(1, r_idx), nbands)
        n_safe = min(max(1, n_idx), nbands)
        s_safe = min(max(1, s_idx), nbands) if s_idx <= nbands else n_safe

        b = src.read(b_safe).astype(np.float32)
        g = src.read(g_safe).astype(np.float32)
        r = src.read(r_safe).astype(np.float32)
        n = src.read(n_safe).astype(np.float32)
        s = src.read(s_safe).astype(np.float32)

        # Handle NoData properly: A pixel is NoData only if ALL bands equal the nodata value
        nodata = src.nodata if src.nodata is not None else -9999.0
        invalid_mask = (b == nodata) & (g == nodata) & (r == nodata) & (n == nodata) & (s == nodata)
        if nodata == 0:
            # For nodata=0, sometimes just b,g,r being 0 is enough to confidently say it's off-image
            invalid_mask = (b == 0) & (g == 0) & (r == 0)
        valid_mask = ~invalid_mask

        # Clip values to 1e-6 to prevent division by zero AND prevent sign-flip bugs from negative reflectances
        b_c = np.clip(b, 1e-6, None)
        g_c = np.clip(g, 1e-6, None)
        r_c = np.clip(r, 1e-6, None)
        n_c = np.clip(n, 1e-6, None)
        s_c = np.clip(s, 1e-6, None)

        # 1. NDVI (NIR - Red) / (NIR + Red)
        denom_ndvi = n_c + r_c
        ndvi = (n_c - r_c) / denom_ndvi

        # 2. NDWI / MNDWI (Green - SWIR) / (Green + SWIR)
        denom_mndwi = g_c + s_c
        mndwi = (g_c - s_c) / denom_mndwi

        # 3. NWI
        vis_mean = (b_c + g_c + r_c) / 3.0
        ir_mean = (n_c + s_c) / 2.0
        denom_nwi = vis_mean + ir_mean
        nwi = (vis_mean - ir_mean) / denom_nwi

        # Final Mask Equation
        water_mask = (mndwi > 0) & (nwi > 0) & (ndvi < 0.1) & valid_mask

        # Cleanup using Sieve (removes noise polygons smaller than k_size)
        mask_uint8 = water_mask.astype(np.uint8)
        if k_size > 0:
            mask_uint8 = sieve(mask_uint8, size=k_size, connectivity=8)

        # Separate NoData from valid land (0)
        mask_uint8[~valid_mask] = 255

        prof = src.profile
        prof.update(count=1, dtype="uint8", nodata=255)
        with rasterio.open(out_f, "w", **prof) as dst:
            dst.write(mask_uint8, 1)


def run_smart_hybrid_masking(in_f, out_f, b_idx, g_idx, r_idx, n_idx, s_idx, k_size, fb):
    import rasterio
    import numpy as np
    from rasterio.features import sieve
    
    with rasterio.open(in_f) as src:
        nbands = src.count
        b_safe = min(max(1, b_idx), nbands)
        g_safe = min(max(1, g_idx), nbands)
        r_safe = min(max(1, r_idx), nbands)
        n_safe = min(max(1, n_idx), nbands)
        s_safe = min(max(1, s_idx), nbands) if s_idx <= nbands else n_safe

        b = src.read(b_safe).astype(np.float32)
        g = src.read(g_safe).astype(np.float32)
        r = src.read(r_safe).astype(np.float32)
        n = src.read(n_safe).astype(np.float32)
        s = src.read(s_safe).astype(np.float32)

        valid_mask = (g > 0) & (n > 0) & (r > 0) & (s > 0) & (g != -9999) & (n != -9999)
        
        denom_ndwi = g + n
        denom_ndwi[denom_ndwi == 0] = 1e-6
        ndwi = (g - n) / denom_ndwi
        
        denom_mndwi = g + s
        denom_mndwi[denom_mndwi == 0] = 1e-6
        mndwi = (g - s) / denom_mndwi
        
        denom_ndvi = n + r
        denom_ndvi[denom_ndvi == 0] = 1e-6
        ndvi = (n - r) / denom_ndvi
        
        valid_ndwi = ndwi[valid_mask]
        water_mask = np.zeros(ndwi.shape, dtype="uint8")
        
        if valid_ndwi.size > 100:
            std_dev = np.std(valid_ndwi)
            fb.pushInfo(f"      [Smart Hybrid] NDWI StdDev: {std_dev:.4f}")
            
            if std_dev < 0.05:
                fb.pushInfo("      [Smart Hybrid] Scene is heavily unimodal. Using Physical thresholds.")
                water_mask[(mndwi > -0.05) & (ndvi < 0.15) & valid_mask] = 1
            else:
                hist, bins = np.histogram(valid_ndwi, bins=256, range=(-1.0, 1.0))
                total = valid_ndwi.size
                
                try:
                    from scipy.ndimage import gaussian_filter1d
                    hist_smoothed = gaussian_filter1d(hist.astype(float), sigma=2)
                except ImportError:
                    hist_smoothed = hist.astype(float)
                
                sum_total = np.dot(np.arange(256), hist)
                max_val = 0
                thresh_idx = 0
                sum_b = 0
                weight_b = 0
                
                for i in range(256):
                    weight_b += hist[i]
                    if weight_b == 0: continue
                    weight_f = total - weight_b
                    if weight_f == 0: break
                    
                    sum_b += i * hist[i]
                    m_b = sum_b / weight_b
                    m_f = (sum_total - sum_b) / weight_f
                    var_b = weight_b * weight_f * (m_b - m_f) ** 2
                    
                    p_t = hist_smoothed[i] / total
                    valley_weight = 1.0 - p_t
                    score = valley_weight * var_b
                    
                    if score > max_val:
                        max_val = score
                        thresh_idx = i
                        
                otsu_thresh = bins[thresh_idx]
                fb.pushInfo(f"      [Smart Hybrid] Valley-Emphasis Otsu NDWI Threshold: {otsu_thresh:.4f}")
                
                water_condition = ((ndwi > otsu_thresh) | (mndwi > -0.05)) & (ndvi < 0.15)
                water_mask[water_condition & valid_mask] = 1
                
        else:
            fb.pushWarning("      [Smart Hybrid] Not enough valid pixels. Defaulting to MNDWI > 0.")
            water_mask[(mndwi > 0) & (ndvi < 0.1) & valid_mask] = 1
            
        if k_size > 0:
            try:
                from scipy import ndimage
                struct = ndimage.generate_binary_structure(2, 2)
                water_mask_closed = ndimage.binary_closing(water_mask, structure=struct, iterations=1).astype("uint8")
                water_mask = sieve(water_mask_closed, size=k_size, connectivity=8)
            except ImportError:
                water_mask = sieve(water_mask, size=k_size, connectivity=8)
                
        water_mask[~valid_mask] = 255
        
        prof = src.profile
        prof.update(count=1, dtype="uint8", nodata=255)
        with rasterio.open(out_f, "w", **prof) as dst:
            dst.write(water_mask.astype('uint8'), 1)


def run_hedley(in_f, out_f, nir_idx, target_bands_idx, mask_f, percentile, fb):
    with rasterio.open(in_f) as src:
        prof = src.profile
        cols = src.width
        rows = src.height
        band_count = src.count

        nir_safe = min(max(1, nir_idx), band_count)
        nir = src.read(nir_safe).astype(np.float64)

        nodata_val = src.nodata
        if nodata_val is not None:
            valid_pixels = (nir != nodata_val) & np.isfinite(nir)
        else:
            valid_pixels = np.isfinite(nir)

        if mask_f and os.path.exists(mask_f):
            with rasterio.open(mask_f) as m_src:
                water_mask = (m_src.read(1) == 1) & valid_pixels
        else:
            water_mask = valid_pixels

        nir_water = nir[water_mask]

        if len(nir_water) == 0:
            fb.pushWarning(
                "      Warning: Mask contains no water pixels! Falling back to full image valid pixels."
            )
            water_mask = valid_pixels
            nir_water = nir[water_mask]

        nir_min = np.percentile(nir_water, percentile)

        prof.update(dtype=rasterio.float32, nodata=np.nan)
        out_data = np.zeros((band_count, rows, cols), dtype=np.float32)

        for b in range(1, band_count + 1):
            band = src.read(b).astype(np.float64)

            if b in target_bands_idx:
                # Ensure we only use pixels where BOTH NIR and the target band are finite
                valid_band_mask = water_mask & np.isfinite(band)
                band_water = band[valid_band_mask]
                nir_water_b = nir[valid_band_mask]

                if len(nir_water_b) > 1:
                    var_nir = np.var(nir_water_b)
                    if var_nir == 0:
                        slope = 0.0
                    else:
                        cov_matrix = np.cov(nir_water_b, band_water)
                        slope = cov_matrix[0, 1] / var_nir
                else:
                    slope = 0.0

                fb.pushInfo(f"      Band {b} slope = {slope:.5f}")
                corrected = band - slope * (nir - nir_min)
                # Prevent negative or exact zero values from causing NaN/mask drops downstream
                corrected = np.clip(corrected, 1e-4, None)
            else:
                corrected = band

            corrected[~valid_pixels] = np.nan
            out_data[b - 1] = corrected.astype(np.float32)

        with rasterio.open(out_f, "w", **prof) as dst:
            dst.write(out_data)
            dst.descriptions = src.descriptions


def run_manual_mask(in_f, out_f, g_idx, n_idx, threshold, k_size, fb):
    with rasterio.open(in_f) as src:
        g_safe = min(max(1, g_idx), src.count)
        n_safe = min(max(1, n_idx), src.count)
        g = src.read(g_safe).astype("float32")
        n = src.read(n_safe).astype("float32")
        denom = g + n
        denom[denom == 0] = 1e-6
        ndwi = (g - n) / denom
        valid_mask = (g > 0) & (n > 0) & (g != -9999) & (n != -9999) & (~np.isnan(ndwi))
        water_mask = np.zeros(ndwi.shape, dtype="uint8")
        water_mask[(ndwi > threshold) & valid_mask] = 1
        if k_size > 0:
            from rasterio.features import sieve
            water_mask = sieve(water_mask, size=k_size, connectivity=8)
        
        water_mask[~valid_mask] = 255
        
        prof = src.profile
        prof.update(count=1, dtype="uint8", nodata=255)
        with rasterio.open(out_f, "w", **prof) as dst:
            dst.write(water_mask, 1)


def run_otsu_robust(in_f, out_f, g_idx, n_idx, adjustment, k_size, fb):
    with rasterio.open(in_f) as src:
        g_safe = min(max(1, g_idx), src.count)
        n_safe = min(max(1, n_idx), src.count)
        g = src.read(g_safe).astype("float32")
        n = src.read(n_safe).astype("float32")
        denom = g + n
        denom[denom == 0] = 1e-6
        ndwi = (g - n) / denom
        valid_mask = (g > 0) & (n > 0) & (g != -9999) & (n != -9999)
        valid_mask &= (~np.isnan(ndwi)) & (~np.isinf(ndwi))
        valid_ndwi = ndwi[valid_mask]

        thresh = 0.0
        if valid_ndwi.size > 100:
            hist, bins = np.histogram(valid_ndwi, bins=256, range=(-1.0, 1.0))
            total = valid_ndwi.size
            sum_total = np.dot(np.arange(256), hist)
            sum_b, weight_b, max_var, thresh_idx = 0, 0, 0, 0
            for i in range(256):
                weight_b += hist[i]
                if weight_b == 0:
                    continue
                weight_f = total - weight_b
                if weight_f == 0:
                    break
                sum_b += i * hist[i]
                m_b = sum_b / weight_b
                m_f = (sum_total - sum_b) / weight_f
                var_b = weight_b * weight_f * (m_b - m_f) ** 2
                if var_b > max_var:
                    max_var = var_b
                    thresh_idx = i
            thresh = bins[thresh_idx]
            fb.pushInfo(f"      Calculated Otsu NDWI: {thresh:.4f} | Adj: {adjustment}")
            thresh += adjustment
        else:
            fb.pushWarning("      Not enough pixels for Otsu.")

        water_mask = np.zeros(ndwi.shape, dtype="uint8")
        water_mask[(ndwi > thresh) & valid_mask] = 1

        if k_size > 0:
            from rasterio.features import sieve
            water_mask = sieve(water_mask, size=k_size, connectivity=8)

        water_mask[~valid_mask] = 255

        prof = src.profile
        prof.update(count=1, dtype="uint8", nodata=255)
        with rasterio.open(out_f, "w", **prof) as dst:
            dst.write(water_mask, 1)


def generate_features(
    in_f,
    out_f,
    review_dir,
    c,
    b,
    g,
    r,
    n,
    selected_indices_str,
    do_calc,
    formula,
    mask_f,
    fb,
):
    with rasterio.open(in_f) as s:
        nbands = s.count
        selected_indices = [int(i) for i in selected_indices_str]

        b_safe = min(max(1, b), nbands)
        g_safe = min(max(1, g), nbands)
        r_safe = min(max(1, r), nbands)
        n_safe = min(max(1, n), nbands)
        c_safe = min(max(1, c), nbands) if (c and 1 <= c <= nbands) else b_safe

        c_val = s.read(c_safe).astype("float32")
        b_val = s.read(b_safe).astype("float32")
        g_val = s.read(g_safe).astype("float32")
        r_val = s.read(r_safe).astype("float32")
        n_val = s.read(n_safe).astype("float32")

        # Relaxed valid mask to avoid dropping valid water pixels with negative or exact zero values
        # A pixel is only NoData if it's -9999.0, NaN, or completely black (all bands 0)
        nodata_mask = (b_val == 0) & (g_val == 0) & (r_val == 0)
        mask_valid = (~nodata_mask) & (b_val != -9999.0) & np.isfinite(b_val)

        if mask_f and os.path.exists(mask_f):
            with rasterio.open(mask_f) as m_src:
                water_mask_array = m_src.read(1) == 1
            mask_valid = mask_valid & water_mask_array

        SCALE = 1000.0
        lc = np.log(np.clip(c_val * SCALE, 1e-6, None))
        lb = np.log(np.clip(b_val * SCALE, 1e-6, None))
        lg = np.log(np.clip(g_val * SCALE, 1e-6, None))
        lr = np.log(np.clip(r_val * SCALE, 1e-6, None))
        ln = np.log(np.clip(n_val * SCALE, 1e-6, None))

        with np.errstate(divide="ignore", invalid="ignore"):
            rbg = lb / lg
            rbr = lb / lr
            rcg = lc / lg
            rgn = lg / ln
            rrn = lr / ln
            ndwi = (g_val - n_val) / (g_val + n_val)

        custom_band = np.zeros_like(b_val)
        if do_calc and formula and 12 in selected_indices:
            try:
                band_dict = {}
                for i in range(1, nbands + 1):
                    band_dict[f"B{i}"] = s.read(i).astype("float32")
                band_dict["np"] = np
                band_dict["log"] = np.log
                fb.pushInfo(f"      Calculating Custom Formula: {formula}")
                res = _safe_eval(formula, band_dict)
                if isinstance(res, np.ndarray):
                    custom_band = res
                    custom_band[~mask_valid] = 0
                    custom_band[np.isinf(custom_band)] = 0
                else:
                    custom_band[:] = res
            except Exception as e:
                fb.pushWarning(f"Calc Error: {e}")

        calculated_feats_map = {
            1: lc,
            2: lb,
            3: lg,
            4: lr,
            5: ln,
            6: rbg,
            7: rbr,
            8: rcg,
            9: rgn,
            10: rrn,
            11: ndwi,
            12: custom_band,
        }

        final_stack = []
        final_descriptions = []

        if 0 in selected_indices:
            fb.pushInfo(f"      Adding {nbands} raw bands to stack...")
            
            # Map index to name based on user inputs
            band_names = {}
            if c > 0: band_names[c] = "Coastal"
            if b > 0: band_names[b] = "Blue"
            if g > 0: band_names[g] = "Green"
            if r > 0: band_names[r] = "Red"
            if n > 0: band_names[n] = "NIR"
            
            for i in range(1, nbands + 1):
                raw_band_data = s.read(i).astype("float32")
                
                # Fix NaN/Inf or original nodata in valid pixels so ML phase doesn't drop them
                invalid_data = np.isnan(raw_band_data) | np.isinf(raw_band_data) | (raw_band_data <= -9999.0)
                raw_band_data[invalid_data & mask_valid] = 0.0
                
                raw_band_data[~mask_valid] = -9999.0
                final_stack.append(raw_band_data)
                
                # Assign a sensible name
                if i in band_names:
                    band_name_str = f"Raw_{band_names[i]}"
                else:
                    existing_desc = s.descriptions[i-1]
                    if existing_desc:
                        band_name_str = existing_desc
                    else:
                        band_name_str = f"B{i}"
                final_descriptions.append(band_name_str)
                
                p_ind = s.profile
                p_ind.update(count=1, dtype="float32", nodata=-9999.0)
                with rasterio.open(
                    os.path.join(review_dir, f"{band_name_str}.tif"), "w", **p_ind
                ) as dst:
                    dst.write(raw_band_data, 1)

        for idx in selected_indices:
            if idx in calculated_feats_map:
                data = calculated_feats_map[idx].copy()
                data[np.isinf(data)] = 0
                data[np.isnan(data)] = 0
                data[~mask_valid] = -9999.0
                final_stack.append(data)
                name = (
                    FEATURE_OPTIONS[idx]
                    .replace("[Log] ", "")
                    .replace("[Ratio] ", "Ratio_")
                    .replace("[Custom] ", "")
                    .replace(" ", "")
                    .replace("/", "_")
                    .replace("(", "")
                    .replace(")", "")
                )
                final_descriptions.append(name)
                p_ind = s.profile
                p_ind.update(count=1, dtype="float32", nodata=-9999.0)
                with rasterio.open(
                    os.path.join(review_dir, f"{name}.tif"), "w", **p_ind
                ) as dst:
                    dst.write(data, 1)

        if not final_stack:
            fb.pushWarning("Stack is empty! Please check selections.")
            return

        stack_arr = np.array(final_stack)
        prof = s.profile
        prof.update(count=len(final_stack), dtype="float32", nodata=-9999.0)
        with rasterio.open(out_f, "w", **prof) as dst:
            dst.write(stack_arr)
            dst.descriptions = tuple(final_descriptions)


def apply_deepwater_mask(
    base_img_path, out_dir,
    b_idx, g_idx, n_idx,
    apply_dw, dw_method, dw_roi, nir_perc,
    median_size, fill_holes, extract_polygon,
    context, feedback
):
    import processing
    from qgis.core import QgsProcessingException, QgsProcessingUtils
    import numpy as np
    import rasterio

    if apply_dw:
        fallback_to_auto = False
        if dw_method == 0:  # Manual
            if not dw_roi:
                feedback.pushWarning("⚠️ Deep Water Filter is ON (Manual mode), but no ROI provided. Falling back to Automatic NIR.")
                fallback_to_auto = True
            else:
                from qgis.core import QgsRasterLayer
                raster_layer = QgsRasterLayer(base_img_path, "base_img")
                if raster_layer.isValid() and dw_roi.crs() != raster_layer.crs():
                    feedback.pushInfo(f"      Reprojecting Deep Water ROI to {raster_layer.crs().authid()}...")
                    reproject_alg = processing.run("native:reprojectlayer", {
                        'INPUT': dw_roi,
                        'TARGET_CRS': raster_layer.crs(),
                        'OUTPUT': 'TEMPORARY_OUTPUT'
                    }, context=context, feedback=feedback, is_child_algorithm=True)
                    dw_roi_final = QgsProcessingUtils.mapLayerFromString(reproject_alg['OUTPUT'], context)
                else:
                    dw_roi_final = dw_roi
                    
                feedback.pushInfo("      Calculating Deep Water Stats (Manual ROI)...")
                
                stats_alg = processing.run("native:zonalstatisticsfb", {
                    'INPUT': dw_roi_final,
                    'INPUT_RASTER': base_img_path,
                    'RASTER_BAND': b_idx,
                    'COLUMN_PREFIX': 'b_',
                    'STATISTICS': [2, 4], # Mean, StDev
                    'OUTPUT': 'TEMPORARY_OUTPUT'
                }, context=context, feedback=feedback, is_child_algorithm=True)
                layer = QgsProcessingUtils.mapLayerFromString(stats_alg['OUTPUT'], context)
                feat = list(layer.getFeatures())[0]
                b_mean = feat['b_mean']
                b_std = feat['b_stdev']
                
                if b_mean is None or b_std is None or np.isnan(b_mean) or np.isnan(b_std):
                    feedback.pushWarning("⚠️ Calculated Blue stats are NULL. Manual ROI might not overlap valid raster data. Falling back to Auto NIR.")
                    fallback_to_auto = True
                else:
                    b_thresh = b_mean + (2 * b_std)
                    
                    stats_alg_g = processing.run("native:zonalstatisticsfb", {
                        'INPUT': dw_roi_final,
                        'INPUT_RASTER': base_img_path,
                        'RASTER_BAND': g_idx,
                        'COLUMN_PREFIX': 'g_',
                        'STATISTICS': [2, 4],
                        'OUTPUT': 'TEMPORARY_OUTPUT'
                    }, context=context, feedback=feedback, is_child_algorithm=True)
                    layer_g = QgsProcessingUtils.mapLayerFromString(stats_alg_g['OUTPUT'], context)
                    feat_g = list(layer_g.getFeatures())[0]
                    g_mean = feat_g['g_mean']
                    g_std = feat_g['g_stdev']
                    
                    if g_mean is None or g_std is None or np.isnan(g_mean) or np.isnan(g_std):
                        feedback.pushWarning("⚠️ Calculated Green stats are NULL. Falling back to Auto NIR.")
                        fallback_to_auto = True
                    else:
                        g_thresh = g_mean + (2 * g_std)
                        feedback.pushInfo(f"      Blue Thresh: {b_thresh:.4f}, Green Thresh: {g_thresh:.4f}")

                        stats_csv = os.path.join(out_dir, "04_DeepWater_Threshold_Stats.csv")
                        with open(stats_csv, "w") as f:
                            f.write("Band,Mean,SD,Threshold\n")
                            f.write(f"Blue,{b_mean},{b_std},{b_thresh}\n")
                            f.write(f"Green,{g_mean},{g_std},{g_thresh}\n")
                        feedback.pushInfo(f"      Saved stats to: {stats_csv}")

        if dw_method == 2:  # Shallow Water Bound (OSW Polygon)
            if not dw_roi:
                feedback.pushWarning("⚠️ OSW Polygon method selected, but no ROI provided. Falling back to Automatic NIR.")
                fallback_to_auto = True
            else:
                from qgis.core import QgsRasterLayer
                raster_layer = QgsRasterLayer(base_img_path, "base_img")
                if raster_layer.isValid() and dw_roi.crs() != raster_layer.crs():
                    feedback.pushInfo(f"      Reprojecting OSW Polygon to {raster_layer.crs().authid()}...")
                    reproject_alg = processing.run("native:reprojectlayer", {
                        'INPUT': dw_roi,
                        'TARGET_CRS': raster_layer.crs(),
                        'OUTPUT': 'TEMPORARY_OUTPUT'
                    }, context=context, feedback=feedback, is_child_algorithm=True)
                    dw_roi_final = QgsProcessingUtils.mapLayerFromString(reproject_alg['OUTPUT'], context)
                else:
                    dw_roi_final = dw_roi
                
                feedback.pushInfo("      Rasterizing OSW Polygon to mask...")
                import json
                from rasterio import features
                geom_list = []
                for feat in dw_roi_final.getFeatures():
                    geom = feat.geometry()
                    if geom and not geom.isNull():
                        geom_list.append(json.loads(geom.asJson()))
                
                with rasterio.open(base_img_path) as src:
                    out_shape = (src.height, src.width)
                    out_transform = src.transform
                
                if geom_list:
                    not_deep_water = features.rasterize(geom_list, out_shape=out_shape, transform=out_transform, fill=0, default_value=1, dtype=np.uint8).astype(bool)
                else:
                    feedback.pushWarning("⚠️ OSW Polygon is empty. Falling back to Automatic NIR.")
                    fallback_to_auto = True

        if dw_method == 1 or fallback_to_auto:  # Automatic
            feedback.pushInfo(f"      Calculating Deep Water Stats (Auto NIR {nir_perc}%)...")
            with rasterio.open(base_img_path) as src:
                nir_arr = src.read(n_idx).astype(np.float32)
                b_arr = src.read(b_idx).astype(np.float32)
                g_arr = src.read(g_idx).astype(np.float32)
                
                valid = (b_arr > 0) & (g_arr > 0) & np.isfinite(b_arr) & np.isfinite(g_arr) & np.isfinite(nir_arr)
                nir_valid = nir_arr[valid]
                if len(nir_valid) == 0:
                    raise QgsProcessingException("No valid pixels found to calculate NIR percentile.")
                nir_thresh = np.percentile(nir_valid, nir_perc)
                
                dw_mask = (nir_arr <= nir_thresh) & valid
                b_deep = b_arr[dw_mask]
                g_deep = g_arr[dw_mask]
                
                if len(b_deep) == 0:
                    raise QgsProcessingException("No deep water pixels identified. Adjust NIR percentile.")
                    
                b_thresh = np.mean(b_deep) + (2 * np.std(b_deep))
                g_thresh = np.mean(g_deep) + (2 * np.std(g_deep))
                feedback.pushInfo(f"      NIR Thresh: {nir_thresh:.4f} | Blue Thresh: {b_thresh:.4f}, Green Thresh: {g_thresh:.4f}")

                stats_csv = os.path.join(out_dir, "04_DeepWater_Threshold_Stats.csv")
                with open(stats_csv, "w") as f:
                    f.write("Band,Mean,SD,Threshold\n")
                    f.write(f"Blue,{np.mean(b_deep)},{np.std(b_deep)},{b_thresh}\n")
                    f.write(f"Green,{np.mean(g_deep)},{np.std(g_deep)},{g_thresh}\n")
                feedback.pushInfo(f"      Saved stats to: {stats_csv}")

    with rasterio.open(base_img_path) as src:
        b_arr = src.read(b_idx).astype(np.float32)
        g_arr = src.read(g_idx).astype(np.float32)
        prof = src.profile
        
    if apply_dw:
        if dw_method == 0 or dw_method == 1 or fallback_to_auto:
            not_deep_water = (b_arr > b_thresh) | (g_arr > g_thresh)
        # if dw_method == 2, not_deep_water is already computed
    else:
        not_deep_water = np.ones_like(b_arr, dtype=bool)
        
    osw_mask = not_deep_water
    
    prof.update(count=1, dtype='uint8', nodata=0)
    dw_mask_path = os.path.join(out_dir, "05_ShallowWater_Pixel_Mask.tif")
    
    with rasterio.open(dw_mask_path, "w", **prof) as dst:
        dst.write(not_deep_water.astype('uint8'), 1)
        
    try:
        from scipy import ndimage
        if median_size > 0 and dw_method != 2:
            osw_mask = ndimage.median_filter(osw_mask.astype(np.uint8), size=median_size) > 0
        if fill_holes and dw_method != 2:
            osw_mask = ndimage.binary_fill_holes(osw_mask)
    except ImportError:
        pass
        
    if dw_method != 2:
        # =========================================================================
        # METHOD A: Elbow Point Detection (Dynamic / No Hardcoded Threshold)
        # =========================================================================
        try:
            from rasterio.features import sieve
            from scipy import ndimage

            labeled_array, num_features = ndimage.label(osw_mask)
            if num_features > 3:
                component_sizes = ndimage.sum_labels(osw_mask, labeled_array, range(1, num_features + 1))
                component_sizes = np.sort(component_sizes)[::-1]

                # Log-transformed scale for Elbow (Max Distance from Secant Line)
                log_sizes = np.log10(component_sizes + 1e-5)
                n_pts = len(log_sizes)
                x_norm = np.linspace(0, 1, n_pts)
                y_range = log_sizes[0] - log_sizes[-1]
                if y_range > 0:
                    y_norm = (log_sizes - log_sizes[-1]) / y_range

                    p1 = np.array([x_norm[0], y_norm[0]])
                    p2 = np.array([x_norm[-1], y_norm[-1]])
                    vec_line = p2 - p1
                    line_norm = np.linalg.norm(vec_line)

                    if line_norm > 0:
                        pts = np.column_stack((x_norm, y_norm))
                        distances = np.abs(np.cross(vec_line, p1 - pts)) / line_norm
                        elbow_idx = np.argmax(distances)
                        cutoff_size = int(component_sizes[elbow_idx])

                        if cutoff_size > 1:
                            feedback.pushInfo(f"      📊 [Elbow Filter] Dynamic Threshold: {cutoff_size} pixels ({num_features} components analyzed)")
                            osw_mask = sieve(osw_mask.astype(np.uint8), size=cutoff_size, connectivity=8).astype(bool)
        except Exception as err:
            feedback.pushWarning(f"⚠️ Elbow filter fallback warning: {str(err)}")
            # =========================================================================
            # METHOD B (ORIGINAL BASELINE - PRE-MODIFICATION):
            # Median Filter + Binary Fill Holes (without any Sieve / Area Filtering)
            # If you want to revert to original baseline, comment out Method A above.
            # (The baseline uses scipy.ndimage.median_filter & binary_fill_holes above).
            # =========================================================================
            pass
        
    osw_mask_path = os.path.join(out_dir, "06_Final_OSW_Mask.tif")
    with rasterio.open(osw_mask_path, "w", **prof) as dst:
        dst.write(osw_mask.astype('uint8'), 1)
        
    osw_poly_path = None
    if extract_polygon:
        if dw_method == 2 and dw_roi:
            osw_poly_path = dw_roi.source()
            feedback.pushInfo(f"      Using existing OSW Polygon: {osw_poly_path}")
        else:
            osw_poly_path = os.path.join(out_dir, "07_OSW_Boundary_Polygon.gpkg")
            poly_res = processing.run("gdal:polygonize", {
                'INPUT': osw_mask_path,
                'BAND': 1,
                'FIELD': 'DN',
                'EIGHT_CONNECTEDNESS': False,
                'OUTPUT': 'TEMPORARY_OUTPUT'
            }, context=context, feedback=feedback, is_child_algorithm=True)
            
            extracted = processing.run("native:extractbyexpression", {
                'INPUT': poly_res['OUTPUT'],
                'EXPRESSION': '"DN" = 1',
                'OUTPUT': 'TEMPORARY_OUTPUT'
            }, context=context, feedback=feedback, is_child_algorithm=True)
            
            crs_wkt = None
            try:
                with rasterio.open(base_img_path) as src:
                    crs_wkt = src.crs.to_wkt() if src.crs else None
            except Exception:  # nosec B110
                pass
                
            from qgis.core import QgsCoordinateReferenceSystem
            qgis_crs = QgsCoordinateReferenceSystem.fromWkt(crs_wkt) if crs_wkt else None
            
            if qgis_crs and qgis_crs.isValid():
                processing.run("native:reprojectlayer", {
                    'INPUT': extracted['OUTPUT'],
                    'TARGET_CRS': qgis_crs,
                    'OUTPUT': osw_poly_path
                }, context=context, feedback=feedback, is_child_algorithm=True)
            else:
                processing.run("native:extractbyexpression", {
                    'INPUT': poly_res['OUTPUT'],
                    'EXPRESSION': '"DN" = 1',
                    'OUTPUT': osw_poly_path
                }, context=context, feedback=feedback, is_child_algorithm=True)
                
            feedback.pushInfo(f"      Saved OSW Polygon to: {osw_poly_path}")

    return osw_poly_path


def run_phase01_preprocessing(algorithm, parameters, context, feedback):
    """Body of SDBPhase1Preprocessing.processAlgorithm."""
    n_threads = algorithm.parameterAsInt(parameters, algorithm.NUM_THREADS, context)
    os.environ["GDAL_NUM_THREADS"] = str(n_threads)

    out_dir = algorithm.parameterAsString(parameters, algorithm.OUTPUT_FOLDER, context)
    os.makedirs(out_dir, exist_ok=True)
    review_dir = os.path.join(out_dir, "1_Review_Intermediate_Bands")
    os.makedirs(review_dir, exist_ok=True)

    p_mask = os.path.join(out_dir, "01_Land_Water_Mask.tif")
    p_glint = os.path.join(out_dir, "02_Sunglint_Corrected.tif")
    p_stack = os.path.join(out_dir, "03_Feature_Stack.tif")

    input_layer = algorithm.parameterAsRasterLayer(
        parameters, algorithm.INPUT_RASTER, context
    )
    if not input_layer:
        raise QgsProcessingException("Input Raster Missing!")
    curr_img = input_layer.source()
    raw_img_path = curr_img

    c_idx = algorithm.parameterAsInt(parameters, algorithm.COASTAL_BAND, context)
    b_idx = algorithm.parameterAsInt(parameters, algorithm.BLUE_BAND, context)
    g_idx = algorithm.parameterAsInt(parameters, algorithm.GREEN_BAND, context)
    r_idx = algorithm.parameterAsInt(parameters, algorithm.RED_BAND, context)
    n_idx = algorithm.parameterAsInt(parameters, algorithm.NIR_BAND, context)
    s_idx = algorithm.parameterAsInt(parameters, algorithm.SWIR_BAND, context)

    # feedback.pushInfo(f"\n>>> MODULE 01 START (Threads: {n_threads})...")

    water_poly_layer = algorithm.parameterAsVectorLayer(
        parameters, algorithm.INPUT_WATER_POLY, context
    )
    enable_auto_mask = algorithm.parameterAsBool(
        parameters, algorithm.ENABLE_MASKING, context
    )

    final_mask_path = None

    if water_poly_layer is not None:
        feedback.pushInfo("      → [1/3] Using provided Vector Polygon for Water Masking...")
        run_polygon_mask(curr_img, p_mask, input_layer, water_poly_layer, feedback)
        final_mask_path = p_mask

    elif enable_auto_mask:
        masking_choice = algorithm.parameterAsInt(
            parameters, algorithm.MASKING_METHOD, context
        )
        k_size = algorithm.parameterAsInt(
            parameters, algorithm.MASK_KERNEL_SIZE, context
        )
        if MASK_METHODS[masking_choice] == "Otsu (Automatic NDWI)":
            adj = algorithm.parameterAsDouble(
                parameters, algorithm.OTSU_ADJUSTMENT, context
            )
            feedback.pushInfo("      → [1/3] Generating Water Mask (Otsu NDWI)...")
            run_otsu_robust(curr_img, p_mask, g_idx, n_idx, adj, k_size, feedback)
        elif MASK_METHODS[masking_choice] == "Manual NDWI Threshold":
            manual_val = algorithm.parameterAsDouble(
                parameters, algorithm.MANUAL_THRESHOLD, context
            )
            feedback.pushInfo("      → [1/3] Generating Water Mask (Manual NDWI)...")
            run_manual_mask(
                curr_img, p_mask, g_idx, n_idx, manual_val, k_size, feedback
            )
        elif MASK_METHODS[masking_choice] == "Smart Hybrid (Dynamic Auto)":
            feedback.pushInfo("      → [1/3] Generating Water Mask (Smart Hybrid Auto)...")
            run_smart_hybrid_masking(curr_img, p_mask, b_idx, g_idx, r_idx, n_idx, s_idx, k_size, feedback)
        else:
            feedback.pushInfo("      → [1/3] Generating Water Mask (3 Indices)...")
            run_watermasking_plugin(curr_img, p_mask, b_idx, g_idx, r_idx, n_idx, s_idx, k_size, feedback)
        final_mask_path = p_mask

    else:
        feedback.pushInfo(
            "      → [1/3] Masking is completely Disabled. Proceeding with the entire image."
        )
        final_mask_path = None

    if algorithm.parameterAsBool(parameters, algorithm.APPLY_SUNGLINT, context):
        nir_band_idx = algorithm.parameterAsInt(
            parameters, algorithm.NIR_BAND, context
        )
        perc = algorithm.parameterAsDouble(
            parameters, algorithm.SUNGLINT_PERCENTILE, context
        )
        target_bands_idx = [c_idx, b_idx, g_idx, r_idx]

        mask_status = "Mask" if final_mask_path else "Full Image"
        feedback.pushInfo(
            f"      → [2/3] Sunglint Correction (Float64 Math | Target: {mask_status})..."
        )
        run_hedley(
            curr_img, p_glint, nir_band_idx, target_bands_idx, final_mask_path, perc, feedback
        )
        curr_img = p_glint
    else:
        feedback.pushInfo("      → [2/3] Sunglint Correction Skipped.")

    feedback.pushInfo("      → [3/3] Generating Features Stack...")
    selected_feats = algorithm.parameterAsEnums(
        parameters, algorithm.FEATURE_SELECTION, context
    )
    do_calc = algorithm.parameterAsBool(parameters, algorithm.ENABLE_BAND_CALC, context)
    calc_formula = algorithm.parameterAsString(
        parameters, algorithm.BAND_MATH_FORMULA, context
    )
    generate_features(
        curr_img,
        p_stack,
        review_dir,
        c_idx,
        b_idx,
        g_idx,
        r_idx,
        n_idx,
        selected_feats,
        do_calc,
        calc_formula,
        final_mask_path,
        feedback,
    )

    feedback.pushInfo("      → [4/4] Applying OSW Filter (Deep Water)...")
    
    apply_dw = algorithm.parameterAsBool(parameters, algorithm.APPLY_DEEPWATER, context)
    dw_method = algorithm.parameterAsInt(parameters, algorithm.DEEPWATER_METHOD, context)
    try:
        dw_roi = algorithm.parameterAsVectorLayer(parameters, algorithm.DEEPWATER_ROI, context)
    except Exception:
        dw_roi = None
    nir_perc = algorithm.parameterAsDouble(parameters, algorithm.NIR_PERCENTILE_OSW, context)

    osw_median = algorithm.parameterAsInt(parameters, algorithm.OSW_MEDIAN_SIZE, context)
    fill_holes = algorithm.parameterAsBool(parameters, algorithm.FILL_INTERNAL_HOLES, context)
    extract_poly = algorithm.parameterAsBool(parameters, algorithm.EXTRACT_POLYGON, context)

    p_osw_stack = os.path.join(out_dir, "4_OSW_Filtered_Stack.tif")

    osw_poly = None
    if apply_dw:
        # Use raw_img_path as the base image to ensure thresholds match raw data exactly
        osw_poly = apply_deepwater_mask(
            base_img_path=raw_img_path,
            out_dir=out_dir,
            b_idx=b_idx,
            g_idx=g_idx,
            n_idx=n_idx,
            apply_dw=apply_dw,
            dw_method=dw_method,
            dw_roi=dw_roi,
            nir_perc=nir_perc,
            median_size=osw_median,
            fill_holes=fill_holes,
            extract_polygon=True, # Always extract polygon for Phase 03/4 clipping
            context=context,
            feedback=feedback
        )
        
        osw_mask_path = os.path.join(out_dir, "06_Final_OSW_Mask.tif")
        if os.path.exists(osw_mask_path) and os.path.exists(p_stack):
            feedback.pushInfo("      → [4/4] Masking Feature Stack with OSW...")
            with rasterio.open(osw_mask_path) as msrc:
                osw_mask = msrc.read(1)
            
            with rasterio.open(p_stack, "r+") as src:
                stack_data = src.read()
                nodata_val = src.nodata if src.nodata is not None else np.nan
                for b_i in range(stack_data.shape[0]):
                    stack_data[b_i, osw_mask == 0] = nodata_val
                src.write(stack_data)
                
            # Create a combined mask for prediction
            if final_mask_path and os.path.exists(final_mask_path):
                feedback.pushInfo("      → [4/4] Generating Combined Prediction Mask (Water + OSW)...")
                combined_mask_path = os.path.join(out_dir, "08_Combined_Prediction_Mask.tif")
                with rasterio.open(final_mask_path) as lw_src:
                    lw_mask = lw_src.read(1)
                    prof = lw_src.profile
                
                # Combine: 1 only if BOTH are 1
                combined = (lw_mask == 1) & (osw_mask == 1)
                with rasterio.open(combined_mask_path, "w", **prof) as dst:
                    dst.write(combined.astype(np.uint8), 1)
                
                final_mask_path = combined_mask_path
    else:
        feedback.pushInfo("      → OSW Filter is completely disabled.")

    output_dict = {"OUTPUT_FEATURES": p_stack}
    if final_mask_path:
        output_dict["OUTPUT_MASK"] = final_mask_path
    if osw_poly:
        output_dict["OUTPUT_OSW_POLY"] = osw_poly

    return output_dict
