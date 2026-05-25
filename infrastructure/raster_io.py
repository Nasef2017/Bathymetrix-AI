# infrastructure/raster_io.py
import rasterio
import numpy as np
import processing


def remove_positive_pixels(in_path, out_path, nodata_val=-9999.0, feedback=None):
    if feedback:
        feedback.pushInfo(f">>> Removing positive values (>= 0) from: {in_path}")
    try:
        with rasterio.open(in_path) as src:
            data = src.read(1)
            meta = src.profile
            nodata_val_src = src.nodata if src.nodata is not None else nodata_val

        mask = (data >= 0) & (data != nodata_val_src)
        data[mask] = nodata_val_src

        meta.update(dtype="float32", nodata=nodata_val_src, count=1)

        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(data.astype(np.float32), 1)

        return out_path
    except Exception as e:
        if feedback:
            feedback.pushWarning(f"Failed to remove positive pixels: {str(e)}")
        return in_path


def clean_depth_map(
    depth_raster, feature_stack_raster, max_depth, out_path, context=None, feedback=None
):
    if feedback:
        feedback.pushInfo(
            ">>> Cleaning Depth Map: Clamping edges and deep anomalies..."
        )
    deep_limit = (max_depth * 1.5) if max_depth < 0 else -100.0
    formula = (
        f"A * ((A >= {deep_limit}) * (B > -9000)) "
        f"+ (-9999.0) * (1 - ((A >= {deep_limit}) * (B > -9000)))"
    )
    calc_res = processing.run(
        "gdal:rastercalculator",
        {
            "INPUT_A": depth_raster,
            "BAND_A": 1,
            "INPUT_B": feature_stack_raster,
            "BAND_B": 1,
            "FORMULA": formula,
            "NO_DATA": -9999.0,
            "RTYPE": 5,
            "OUTPUT": out_path,
        },
        context=context,
        feedback=feedback,
        is_child_algorithm=True,
    )
    return calc_res["OUTPUT"]


def slope_filter_depth(
    depth_raster, slope_threshold, out_path, context=None, feedback=None
):
    if feedback:
        feedback.pushInfo(
            f">>> Applying slope filter (threshold={slope_threshold} degrees)..."
        )
    slope_temp = out_path.replace(".tif", "_slope.tif")

    processing.run(
        "gdal:slope",
        {
            "INPUT": depth_raster,
            "BAND": 1,
            "SCALE": 1.0,
            "AS_PERCENT": False,
            "COMPUTE_EDGES": True,
            "ZEVENBERGEN": False,
            "OUTPUT": slope_temp,
        },
        context=context,
        feedback=feedback,
        is_child_algorithm=True,
    )

    formula = f"A * (B <= {slope_threshold}) + (-9999.0) * (B > {slope_threshold})"
    result = processing.run(
        "gdal:rastercalculator",
        {
            "INPUT_A": depth_raster,
            "BAND_A": 1,
            "INPUT_B": slope_temp,
            "BAND_B": 1,
            "FORMULA": formula,
            "NO_DATA": -9999.0,
            "RTYPE": 5,
            "OUTPUT": out_path,
        },
        context=context,
        feedback=feedback,
        is_child_algorithm=True,
    )
    return result["OUTPUT"]
