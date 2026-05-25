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

MASK_METHODS = ["Otsu (Automatic NDWI)", "Manual NDWI Threshold"]

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


def run_hedley(in_f, out_f, nir_idx, target_bands_idx, mask_f, percentile, fb):
    with rasterio.open(in_f) as src:
        prof = src.profile
        cols = src.width
        rows = src.height
        band_count = src.count

        nir = src.read(nir_idx).astype(np.float64)

        if mask_f and os.path.exists(mask_f):
            with rasterio.open(mask_f) as m_src:
                water_mask = m_src.read(1) == 1
        else:
            water_mask = np.isfinite(nir)

        nir_water = nir[water_mask]

        if len(nir_water) == 0:
            fb.pushWarning(
                "      Warning: Mask contains no water pixels! Falling back to full image valid pixels."
            )
            water_mask = np.isfinite(nir)
            nir_water = nir[water_mask]

        nir_min = np.percentile(nir_water, percentile)

        prof.update(dtype=rasterio.float32, nodata=np.nan)
        out_data = np.zeros((band_count, rows, cols), dtype=np.float32)

        for b in range(1, band_count + 1):
            band = src.read(b).astype(np.float64)

            if b in target_bands_idx:
                band_water = band[water_mask]

                var_nir = np.var(nir_water)
                if var_nir == 0:
                    slope = 0.0
                else:
                    cov_matrix = np.cov(nir_water, band_water)
                    slope = cov_matrix[0, 1] / var_nir

                fb.pushInfo(f"      Band {b} slope = {slope:.5f}")
                corrected = band - slope * (nir - nir_min)
            else:
                corrected = band

            out_data[b - 1] = corrected.astype(np.float32)

        with rasterio.open(out_f, "w", **prof) as dst:
            dst.write(out_data)


def run_manual_mask(in_f, out_f, g_idx, n_idx, threshold, k_size, fb):
    with rasterio.open(in_f) as src:
        g = src.read(g_idx).astype("float32")
        n = src.read(n_idx).astype("float32")
        denom = g + n
        denom[denom == 0] = 1e-6
        ndwi = (g - n) / denom
        valid_mask = (g > 0) & (n > 0) & (g != -9999) & (n != -9999) & (~np.isnan(ndwi))
        water_mask = np.zeros(ndwi.shape, dtype="uint8")
        water_mask[(ndwi > threshold) & valid_mask] = 1
        if scipy_is_available and k_size > 0:
            kernel = np.ones((k_size, k_size))
            water_mask = binary_opening(water_mask, kernel).astype("uint8")
            water_mask = binary_closing(water_mask, kernel).astype("uint8")
        prof = src.profile
        prof.update(count=1, dtype="uint8", nodata=0)
        with rasterio.open(out_f, "w", **prof) as dst:
            dst.write(water_mask, 1)


def run_otsu_robust(in_f, out_f, g_idx, n_idx, adjustment, k_size, fb):
    with rasterio.open(in_f) as src:
        g = src.read(g_idx).astype("float32")
        n = src.read(n_idx).astype("float32")
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

        if scipy_is_available and k_size > 0:
            kernel = np.ones((k_size, k_size))
            water_mask = binary_opening(water_mask, kernel).astype("uint8")
            water_mask = binary_closing(water_mask, kernel).astype("uint8")

        prof = src.profile
        prof.update(count=1, dtype="uint8", nodata=0)
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

        c_val = s.read(c).astype("float32")
        b_val = s.read(b).astype("float32")
        g_val = s.read(g).astype("float32")
        r_val = s.read(r).astype("float32")
        n_val = s.read(n).astype("float32")

        mask_valid = (b_val > 0) & (g_val > 0) & np.isfinite(b_val)

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

        custom_band = np.zeros_like(b_val)
        if do_calc and formula and 9 in selected_indices:
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
            9: custom_band,
        }

        final_stack = []

        if 0 in selected_indices:
            fb.pushInfo(f"      Adding {nbands} raw bands to stack...")
            for i in range(1, nbands + 1):
                raw_band_data = s.read(i).astype("float32")
                raw_band_data[~mask_valid] = -9999.0
                final_stack.append(raw_band_data)
                p_ind = s.profile
                p_ind.update(count=1, dtype="float32", nodata=-9999.0)
                with rasterio.open(
                    os.path.join(review_dir, f"Raw_Band_{i}.tif"), "w", **p_ind
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
                    .replace("[", "")
                    .replace("]", "")
                    .replace(" ", "_")
                    .replace("/", "")
                )
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


def run_phase01_preprocessing(algorithm, parameters, context, feedback):
    """Body of SDBPhase1Preprocessing.processAlgorithm."""
    n_threads = algorithm.parameterAsInt(parameters, algorithm.NUM_THREADS, context)
    os.environ["GDAL_NUM_THREADS"] = str(n_threads)

    out_dir = algorithm.parameterAsString(parameters, algorithm.OUTPUT_FOLDER, context)
    os.makedirs(out_dir, exist_ok=True)
    review_dir = os.path.join(out_dir, "1_Review_Intermediate_Bands")
    os.makedirs(review_dir, exist_ok=True)

    p_mask = os.path.join(out_dir, "1_Water_Mask.tif")
    p_glint = os.path.join(out_dir, "2_Sunglint_Corrected.tif")
    p_stack = os.path.join(out_dir, "3_Features_Stack.tif")

    input_layer = algorithm.parameterAsRasterLayer(
        parameters, algorithm.INPUT_RASTER, context
    )
    if not input_layer:
        raise QgsProcessingException("Input Raster Missing!")
    curr_img = input_layer.source()

    c_idx = algorithm.parameterAsInt(parameters, algorithm.COASTAL_BAND, context)
    b_idx = algorithm.parameterAsInt(parameters, algorithm.BLUE_BAND, context)
    g_idx = algorithm.parameterAsInt(parameters, algorithm.GREEN_BAND, context)
    r_idx = algorithm.parameterAsInt(parameters, algorithm.RED_BAND, context)
    n_idx = algorithm.parameterAsInt(parameters, algorithm.NIR_BAND, context)

    feedback.pushInfo(f"\n>>> MODULE 01 START (Threads: {n_threads})...")

    water_poly_layer = algorithm.parameterAsVectorLayer(
        parameters, algorithm.INPUT_WATER_POLY, context
    )
    enable_auto_mask = algorithm.parameterAsBool(
        parameters, algorithm.ENABLE_MASKING, context
    )

    final_mask_path = None

    if water_poly_layer is not None:
        feedback.pushInfo("   [1/3] Using provided Vector Polygon for Water Masking...")
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
            feedback.pushInfo("   [1/3] Generating Water Mask (Otsu NDWI)...")
            run_otsu_robust(curr_img, p_mask, g_idx, n_idx, adj, k_size, feedback)
        else:
            manual_val = algorithm.parameterAsDouble(
                parameters, algorithm.MANUAL_THRESHOLD, context
            )
            feedback.pushInfo("   [1/3] Generating Water Mask (Manual NDWI)...")
            run_manual_mask(
                curr_img, p_mask, g_idx, n_idx, manual_val, k_size, feedback
            )
        final_mask_path = p_mask

    else:
        feedback.pushInfo(
            "   [1/3] Masking is completely Disabled. Proceeding with the entire image."
        )
        final_mask_path = None

    if algorithm.parameterAsBool(parameters, algorithm.APPLY_SUNGLINT, context):
        nir_g = algorithm.parameterAsInt(
            parameters, algorithm.NIR_BAND_SUNGLINT, context
        )
        perc = algorithm.parameterAsDouble(
            parameters, algorithm.SUNGLINT_PERCENTILE, context
        )
        target_bands_idx = [c_idx, b_idx, g_idx, r_idx]

        mask_status = "Mask" if final_mask_path else "Full Image"
        feedback.pushInfo(
            f"   [2/3] Sunglint Correction (Float64 Math | Target: {mask_status})..."
        )
        run_hedley(
            curr_img, p_glint, nir_g, target_bands_idx, final_mask_path, perc, feedback
        )
        curr_img = p_glint
    else:
        feedback.pushInfo("   [2/3] Sunglint Correction Skipped.")

    feedback.pushInfo("   [3/3] Generating Features Stack...")
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

    feedback.pushInfo(">>> MODULE 1 COMPLETED SUCCESSFULLY.")

    output_dict = {"OUTPUT_FEATURES": p_stack}
    if final_mask_path:
        output_dict["OUTPUT_MASK"] = final_mask_path
    return output_dict
