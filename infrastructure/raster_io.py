# infrastructure/raster_io.py
import rasterio
import numpy as np
try:
    import processing
except ImportError:
    processing = None


def remove_positive_pixels(in_path, out_path, nodata_val=-9999.0, feedback=None):
    try:
        with rasterio.open(in_path) as src:
            data = src.read(1)
            meta = src.profile
            nodata_val_src = src.nodata if src.nodata is not None else nodata_val

        valid_mask = np.isfinite(data) & (data != nodata_val_src) & (np.abs(data) < 15000)
        valid_vals = data[valid_mask]
        
        if len(valid_vals) > 0:
            median_val = float(np.nanmedian(valid_vals))
            # Only remove positive values if the dataset is using negative elevation convention (depth < 0)
            if median_val < 0:
                mask = (data > 0) & (data != nodata_val_src)
                data[mask] = nodata_val_src
                if feedback:
                    feedback.pushInfo(f">>> Negative depth convention detected (median: {median_val:.2f}m). Removed land/positive pixels (> 0).")
            else:
                if feedback:
                    feedback.pushInfo(f">>> Positive depth convention detected (median: {median_val:.2f}m). Preserved positive bathymetric depths.")

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


def get_raster_min_max(raster_path):
    """
    Robust calculation of raster min/max depths for bathymetry color ramp scaling.
    Excludes nodata, NaNs, and extreme outliers.
    """
    import os
    if not raster_path or not os.path.exists(raster_path):
        return -30.0, 0.0
    try:
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            nodata = src.nodata if src.nodata is not None else -9999.0
            valid = (data != nodata) & (data > -9000) & (~np.isnan(data))
            if not np.any(valid):
                return -30.0, 0.0
            v_data = data[valid]
            p2 = float(np.nanpercentile(v_data, 1.0))
            p98 = float(np.nanpercentile(v_data, 99.0))
            if p2 >= p98:
                p2 = float(np.min(v_data))
                p98 = float(np.max(v_data))
            if p2 >= p98:
                p2, p98 = p2 - 5.0, p2 + 1.0
            return p2, p98
    except Exception:
        return -30.0, 0.0


def write_qml_style(tif_path):
    """
    Writes a standardized QGIS Layer Style (.qml) alongside the given GeoTIFF raster.
    Creates a unified Single-Band Pseudocolor ocean bathymetry theme (Deep Navy -> Shallow Surf/White).
    """
    import os
    if not tif_path or not os.path.exists(tif_path):
        return None
    
    qml_path = os.path.splitext(tif_path)[0] + ".qml"
    min_d, max_d = get_raster_min_max(tif_path)
    
    step = (max_d - min_d) / 8.0
    if step <= 0:
        step = 1.0
    
    # Base palette: Deep Navy -> Pale White/Surf
    palette = [
        "#08306b", "#08519c", "#2171b5", "#4292c6",
        "#6baed6", "#9ecae1", "#c6dbef", "#deebf7", "#f7fbff"
    ]
    
    # Check if depths are positive convention (e.g. 0m shallow -> +30m deep)
    is_positive_convention = (min_d >= 0 and max_d > 0)
    
    if is_positive_convention:
        # min_d is Shallow, max_d is Deep -> reverse palette so max_d gets deep navy
        colors = list(reversed(palette))
        items_xml = []
        for i in range(9):
            val = min_d + step * i if i < 8 else max_d
            lbl = f"{val:.2f} m"
            if i == 0:
                lbl += " (Shallow)"
            elif i == 8:
                lbl += " (Deep)"
            items_xml.append(f'          <item alpha="255" value="{val}" label="{lbl}" color="{colors[i]}"/>')
    else:
        # min_d is Deep (e.g. -30m), max_d is Shallow (e.g. 0m)
        colors = palette
        items_xml = []
        for i in range(9):
            val = min_d + step * i if i < 8 else max_d
            lbl = f"{val:.2f} m"
            if i == 0:
                lbl += " (Deep)"
            elif i == 8:
                lbl += " (Shallow)"
            items_xml.append(f'          <item alpha="255" value="{val}" label="{lbl}" color="{colors[i]}"/>')
            
    items_block = "\n".join(items_xml)
    
    qml_content = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28.0" hasScaleBasedVisibilityFlag="0" minScale="1e+08" maxScale="0">
  <pipe>
    <provider>
      <resampling zoomedOutResamplingMethod="nearestNeighbour" maxOversampling="2" zoomedInResamplingMethod="nearestNeighbour" enabled="false"/>
    </provider>
    <rasterrenderer opacity="1" classificationMin="{min_d}" nodataColor="" alphaBand="-1" classificationMax="{max_d}" band="1" type="singlebandpseudocolor">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <rastershader>
        <colorrampshader classificationMode="1" colorRampType="INTERPOLATED" labelPrecision="4" clip="0">
{items_block}
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0"/>
    <huesaturation colorizeGreen="128" colorizeStrength="100" saturation="0" colorizeOn="0" grayscaleMode="0" colorizeRed="255" colorizeBlue="128"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
"""
    try:
        with open(qml_path, "w", encoding="utf-8") as f:
            f.write(qml_content)
        return qml_path
    except Exception:
        return None


try:
    from qgis.core import QgsProcessingLayerPostProcessorInterface

    class StylePostProcessor(QgsProcessingLayerPostProcessorInterface):
        """
        QGIS Processing Layer Post-Processor to apply a .qml style file
        automatically when a layer is loaded upon algorithm completion.
        Maintains an internal reference registry to prevent garbage collection by PyQGIS.
        """
        _instances = []

        def __init__(self, style_path):
            super().__init__()
            self.style_path = str(style_path) if style_path else ""
            self.__class__._instances.append(self)

        def postProcessLayer(self, layer, context, feedback):
            import os
            try:
                if layer and layer.isValid() and self.style_path and os.path.exists(self.style_path):
                    layer.loadNamedStyle(self.style_path)
                    layer.triggerRepaint()
                    if feedback:
                        feedback.pushInfo(f"🎨 Applied bathymetric symbology from: {self.style_path}")
            except Exception:
                pass

        @classmethod
        def create(cls, style_path):
            return cls(style_path)
except Exception:
    StylePostProcessor = None


