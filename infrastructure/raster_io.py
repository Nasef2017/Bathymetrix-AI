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
    try:
        import scipy.ndimage as ndi

        with rasterio.open(depth_raster) as src:
            data = src.read(1).astype(np.float32)
            meta = src.profile.copy()
            nodata_val = src.nodata if src.nodata is not None else -9999.0

            # Pixel resolution in meters
            dx = abs(float(src.transform[0]))
            dy = abs(float(src.transform[4]))
            if dx <= 0: dx = 10.0
            if dy <= 0: dy = 10.0

            valid_mask = (data != nodata_val) & (data > -9000) & (~np.isnan(data))

            if not np.any(valid_mask):
                meta.update(dtype="float32", nodata=-9999.0, count=1)
                with rasterio.open(out_path, "w", **meta) as dst:
                    dst.write(np.full_like(data, -9999.0, dtype=np.float32), 1)
                return out_path

            # Fill nodata with nearest valid values to avoid false steep edges at shoreline/nodata boundary
            filled_data = data.copy()
            invalid_mask = ~valid_mask
            if np.any(invalid_mask):
                dist, inds = ndi.distance_transform_edt(invalid_mask, return_indices=True)
                filled_data[invalid_mask] = data[inds[0][invalid_mask], inds[1][invalid_mask]]

            # Standard Horn's 3x3 slope kernels (used by GDAL)
            kernel_x = np.array([[-1.0, 0.0, 1.0],
                                 [-2.0, 0.0, 2.0],
                                 [-1.0, 0.0, 1.0]], dtype=np.float32) / (8.0 * dx)
            kernel_y = np.array([[-1.0, -2.0, -1.0],
                                 [ 0.0,  0.0,  0.0],
                                 [ 1.0,  2.0,  1.0]], dtype=np.float32) / (8.0 * dy)

            dz_dx = ndi.convolve(filled_data, kernel_x, mode="nearest")
            dz_dy = ndi.convolve(filled_data, kernel_y, mode="nearest")

            slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
            slope_deg = np.degrees(slope_rad)

            # Filter: Keep valid pixels where slope <= slope_threshold, else set to nodata
            thresh = float(slope_threshold)
            filtered_depth = np.where(valid_mask & (slope_deg <= thresh), data, -9999.0)

            meta.update(dtype="float32", nodata=-9999.0, count=1)
            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(filtered_depth.astype(np.float32), 1)

        return out_path
    except Exception as e:
        if feedback:
            feedback.pushWarning(f"Failed to apply slope filter: {str(e)}")
        return depth_raster
