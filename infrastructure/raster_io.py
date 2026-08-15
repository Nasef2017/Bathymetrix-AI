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
    import rasterio
    import numpy as np

    abs_max = abs(float(max_depth)) if max_depth is not None and max_depth != 0 else 50.0
    deep_limit_neg = -abs_max * 1.5
    deep_limit_pos = abs_max * 1.5

    with rasterio.open(depth_raster) as src_depth, rasterio.open(feature_stack_raster) as src_feat:
        depth_data = src_depth.read(1)
        nodata_depth = src_depth.nodata if src_depth.nodata is not None else -9999.0
        
        # Ensure feat_data matches shape (just in case)
        if src_feat.shape != src_depth.shape:
            from rasterio.warp import reproject, Resampling
            feat_data = np.zeros_like(depth_data, dtype=np.float32)
            reproject(
                source=src_feat.read(1),
                destination=feat_data,
                src_transform=src_feat.transform,
                src_crs=src_feat.crs,
                dst_transform=src_depth.transform,
                dst_crs=src_depth.crs,
                resampling=Resampling.nearest
            )
        else:
            feat_data = src_feat.read(1)
        
        # Exclude nodata pixels
        is_nodata = (depth_data == nodata_depth) | (depth_data < -9000) | np.isnan(depth_data)
        valid_depth = ~is_nodata
        valid_feat = (feat_data > -9000) & (~np.isnan(feat_data))
        
        # Depth filter that respects both negative & positive depths
        valid_range = (depth_data >= deep_limit_neg) & (depth_data <= deep_limit_pos)
        
        valid_mask = valid_depth & valid_feat & valid_range
        clean_data = np.where(valid_mask, depth_data, -9999.0)
        
        profile = src_depth.profile.copy()
        profile.update(dtype=rasterio.float32, nodata=-9999.0)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(clean_data.astype(np.float32), 1)
    return out_path


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
