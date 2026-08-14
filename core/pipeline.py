import datetime
import os
import processing

from qgis.core import (
    QgsProcessingContext,
    QgsProcessingException,
    QgsProject,
    QgsRasterLayer,
    QgsProcessingFeedback,
    QgsProcessingLayerPostProcessorInterface,
    QgsVectorLayer,
    NULL,
)

class StylePostProcessor(QgsProcessingLayerPostProcessorInterface):
    def __init__(self, qml_path):
        super().__init__()
        self.qml_path = qml_path
        
    def postProcessLayer(self, layer, context, feedback):
        if layer and layer.isValid():
            layer.loadNamedStyle(self.qml_path)
            layer.triggerRepaint()

from ..infrastructure.logging import append_log
from ..infrastructure.raster_io import (
    clean_depth_map,
    remove_positive_pixels,
    slope_filter_depth,
)
from ..infrastructure.vector_io import filter_by_depth, reproject_layer_if_needed


class LoggingFeedback(QgsProcessingFeedback):
    def __init__(self, original_feedback, log_file_path):
        super().__init__()
        self.original = original_feedback
        self.log_path = log_file_path
        self.is_logging_feedback = True
        
    def setProgressText(self, text):
        if self.original:
            self.original.setProgressText(text)
        self.log_message(f"Progress: {text}")
        
    def pushInfo(self, info):
        if self.original:
            self.original.pushInfo(info)
        self.log_message(info)
        
    def pushWarning(self, warning):
        if self.original:
            self.original.pushWarning(warning)
        self.log_message(f"[Warning] {warning}")
        
    def pushError(self, error):
        if self.original:
            self.original.pushError(error)
        self.log_message(f"[ERROR] {error}")
        
    def reportError(self, error, fatal=False):
        if self.original:
            self.original.reportError(error, fatal)
        self.log_message(f"[CRITICAL ERROR] {error}")
        
    def setProgress(self, progress):
        if self.original:
            self.original.setProgress(progress)
            
    def isCanceled(self):
        if self.original:
            return self.original.isCanceled()
        return super().isCanceled()
        
    def log_message(self, message):
        if self.log_path:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(message + "\n")
            except Exception:  # nosec B110
                pass


def get_raster_min_max(raster_path):
    try:
        import rasterio
        import numpy as np
        with rasterio.open(raster_path) as src:
            data = src.read(1).astype(float)
            nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = np.nan
            data[~np.isfinite(data)] = np.nan
            valid = data[np.isfinite(data)]
            if len(valid) > 0:
                return float(np.nanmin(valid)), float(np.nanmax(valid))
    except Exception:  # nosec B110
        pass
    return -30.0, 0.0


def write_qml_style(tif_path):
    if not tif_path or not os.path.exists(tif_path):
        return
    
    qml_path = os.path.splitext(tif_path)[0] + ".qml"
    min_d, max_d = get_raster_min_max(tif_path)
    
    step = (max_d - min_d) / 8.0
    
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
          <item alpha="255" value="{min_d}" label="{min_d:.2f}" color="#08306b"/>
          <item alpha="255" value="{min_d + step * 1}" label="{(min_d + step * 1):.2f}" color="#08519c"/>
          <item alpha="255" value="{min_d + step * 2}" label="{(min_d + step * 2):.2f}" color="#2171b5"/>
          <item alpha="255" value="{min_d + step * 3}" label="{(min_d + step * 3):.2f}" color="#4292c6"/>
          <item alpha="255" value="{min_d + step * 4}" label="{(min_d + step * 4):.2f}" color="#6baed6"/>
          <item alpha="255" value="{min_d + step * 5}" label="{(min_d + step * 5):.2f}" color="#9ecae1"/>
          <item alpha="255" value="{min_d + step * 6}" label="{(min_d + step * 6):.2f}" color="#c6dbef"/>
          <item alpha="255" value="{min_d + step * 7}" label="{(min_d + step * 7):.2f}" color="#deebf7"/>
          <item alpha="255" value="{max_d}" label="{max_d:.2f}" color="#f7fbff"/>
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
    except Exception:  # nosec B110
        pass


def generate_3d_seabed_png(raster_path, out_png_path, feedback=None):
    if feedback:
        feedback.pushInfo(f"--- Generating static 3D Seabed Plot: {out_png_path}")
    try:
        import rasterio
        from rasterio.enums import Resampling
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        grid_size = 150
        with rasterio.open(raster_path) as src:
            data = src.read(
                1,
                out_shape=(grid_size, grid_size),
                resampling=Resampling.bilinear
            )
            nodata = src.nodata if src.nodata is not None else -9999.0
            transform = src.transform
            bounds = src.bounds
            
            # Real-world extents in meters
            x_extent = bounds.right - bounds.left
            y_extent = bounds.top - bounds.bottom
            
            data_float = data.astype(float)
            data_float[data_float == nodata] = np.nan
            
            # Create real-world coordinate grids (meters from origin)
            x_coords = np.linspace(0, x_extent, grid_size)
            y_coords = np.linspace(y_extent, 0, grid_size)
            X, Y = np.meshgrid(x_coords, y_coords)
            masked_Z = np.ma.masked_invalid(data_float)
            
            # Auto-calculate Vertical Exaggeration for visual clarity
            valid_z = data_float[np.isfinite(data_float)]
            if len(valid_z) > 0:
                z_range = float(np.nanmax(valid_z) - np.nanmin(valid_z))
                z_range = max(z_range, 0.1)
                horizontal_extent = max(x_extent, y_extent)
                # Target: Z should visually appear as ~35% of the longest horizontal axis
                target_z_fraction = 0.35
                ve = (target_z_fraction * horizontal_extent) / z_range
                ve = max(2.0, ve)  # minimum VE = 2x
            else:
                ve = 3.0
                z_range = 1.0
                horizontal_extent = max(x_extent, y_extent)
            
            fig = plt.figure(figsize=(12, 9), facecolor='#020617')
            ax = fig.add_subplot(111, projection='3d', facecolor='#020617')
            
            surf = ax.plot_surface(
                X, Y, masked_Z,
                cmap='Spectral_r',
                linewidth=0,
                antialiased=True,
                alpha=0.92,
                rcount=120, ccount=120
            )
            
            # Set normalized box aspect: x_norm, y_norm relative to max_h, z = target fraction
            max_h = max(x_extent, y_extent)
            try:
                ax.set_box_aspect([x_extent / max_h, y_extent / max_h, target_z_fraction])
            except AttributeError:
                pass
            
            ve_label = f"VE ≈ {ve:.1f}x"
            ax.set_title(f"3D Seabed Topography Model\n", color='white', fontsize=14, fontweight='bold', pad=15)
            ax.text2D(0.5, 0.93, ve_label, transform=ax.transAxes, ha='center', va='top',
                      fontsize=9, color='#60a5fa',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='#1e293b', edgecolor='#334155', alpha=0.85))
            
            # Format axis labels with real units
            if x_extent > 2000:
                ax.set_xlabel(f"Easting ({x_extent/1000:.1f} km)", color='#94a3b8', labelpad=10, fontsize=9)
            else:
                ax.set_xlabel(f"Easting ({x_extent:.0f} m)", color='#94a3b8', labelpad=10, fontsize=9)
            if y_extent > 2000:
                ax.set_ylabel(f"Northing ({y_extent/1000:.1f} km)", color='#94a3b8', labelpad=10, fontsize=9)
            else:
                ax.set_ylabel(f"Northing ({y_extent:.0f} m)", color='#94a3b8', labelpad=10, fontsize=9)
            ax.set_zlabel("Depth (m)", color='#94a3b8', labelpad=10, fontsize=9)
            
            ax.tick_params(colors='#94a3b8', labelsize=7)
            ax.xaxis.line.set_color('#334155')
            ax.yaxis.line.set_color('#334155')
            ax.zaxis.line.set_color('#334155')
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('#1e293b')
            ax.yaxis.pane.set_edgecolor('#1e293b')
            ax.zaxis.pane.set_edgecolor('#1e293b')
            ax.grid(True, color='#1e293b', linestyle='--', alpha=0.3)
            
            cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=14, pad=0.08)
            cbar.set_label('Depth (m)', color='white', fontsize=10, labelpad=10)
            cbar.ax.yaxis.set_tick_params(colors='#94a3b8', labelsize=8)
            cbar.ax.yaxis.label.set_color('white')
            
            ax.view_init(elev=30, azim=-55)
            
            plt.savefig(out_png_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
            plt.close(fig)
            if feedback:
                feedback.pushInfo(f"Static 3D Seabed Plot generated successfully. {ve_label}")
            return True
    except Exception as e:
        if feedback:
            feedback.pushWarning(f"Failed to generate static 3D Seabed PNG: {str(e)}")
        return False


def generate_3d_seabed_png(raster_path, out_png_path, feedback=None):
    if feedback:
        feedback.pushInfo(f"--- Generating static 3D Seabed Plot: {out_png_path}")
    try:
        import rasterio
        from rasterio.enums import Resampling
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        grid_size = 150
        with rasterio.open(raster_path) as src:
            data = src.read(
                1,
                out_shape=(grid_size, grid_size),
                resampling=Resampling.bilinear
            )
            nodata = src.nodata if src.nodata is not None else -9999.0
            transform = src.transform
            bounds = src.bounds
            
            # Real-world extents in meters
            x_extent = bounds.right - bounds.left
            y_extent = bounds.top - bounds.bottom
            
            data_float = data.astype(float)
            data_float[data_float == nodata] = np.nan
            
            # Create real-world coordinate grids (meters from origin)
            x_coords = np.linspace(0, x_extent, grid_size)
            y_coords = np.linspace(0, y_extent, grid_size)
            X, Y = np.meshgrid(x_coords, y_coords)
            masked_Z = np.ma.masked_invalid(data_float)
            
            # Auto-calculate Vertical Exaggeration for visual clarity
            valid_z = data_float[np.isfinite(data_float)]
            if len(valid_z) > 0:
                z_range = float(np.nanmax(valid_z) - np.nanmin(valid_z))
                z_range = max(z_range, 0.1)
                horizontal_extent = max(x_extent, y_extent)
                # Target: Z should visually appear as ~35% of the longest horizontal axis
                target_z_fraction = 0.35
                ve = (target_z_fraction * horizontal_extent) / z_range
                ve = max(2.0, ve)  # minimum VE = 2x
            else:
                ve = 3.0
                z_range = 1.0
                horizontal_extent = max(x_extent, y_extent)
            
            fig = plt.figure(figsize=(12, 9), facecolor='#020617')
            ax = fig.add_subplot(111, projection='3d', facecolor='#020617')
            
            surf = ax.plot_surface(
                X, Y, masked_Z,
                cmap='Spectral_r',
                linewidth=0,
                antialiased=True,
                alpha=0.92,
                rcount=120, ccount=120
            )
            
            # Set normalized box aspect: x_norm, y_norm relative to max_h, z = target fraction
            max_h = max(x_extent, y_extent)
            try:
                ax.set_box_aspect([x_extent / max_h, y_extent / max_h, target_z_fraction])
            except AttributeError:
                pass
            
            ve_label = f"VE ≈ {ve:.1f}x"
            ax.set_title(f"3D Seabed Topography Model\n", color='white', fontsize=14, fontweight='bold', pad=15)
            ax.text2D(0.5, 0.93, ve_label, transform=ax.transAxes, ha='center', va='top',
                      fontsize=9, color='#60a5fa',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='#1e293b', edgecolor='#334155', alpha=0.85))
            
            # Format axis labels with real units
            if x_extent > 2000:
                ax.set_xlabel(f"Easting ({x_extent/1000:.1f} km)", color='#94a3b8', labelpad=10, fontsize=9)
            else:
                ax.set_xlabel(f"Easting ({x_extent:.0f} m)", color='#94a3b8', labelpad=10, fontsize=9)
            if y_extent > 2000:
                ax.set_ylabel(f"Northing ({y_extent/1000:.1f} km)", color='#94a3b8', labelpad=10, fontsize=9)
            else:
                ax.set_ylabel(f"Northing ({y_extent:.0f} m)", color='#94a3b8', labelpad=10, fontsize=9)
            ax.set_zlabel("Depth (m)", color='#94a3b8', labelpad=10, fontsize=9)
            
            ax.tick_params(colors='#94a3b8', labelsize=7)
            ax.xaxis.line.set_color('#334155')
            ax.yaxis.line.set_color('#334155')
            ax.zaxis.line.set_color('#334155')
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('#1e293b')
            ax.yaxis.pane.set_edgecolor('#1e293b')
            ax.zaxis.pane.set_edgecolor('#1e293b')
            ax.grid(True, color='#1e293b', linestyle='--', alpha=0.3)
            
            cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=14, pad=0.08)
            cbar.set_label('Depth (m)', color='white', fontsize=10, labelpad=10)
            cbar.ax.yaxis.set_tick_params(colors='#94a3b8', labelsize=8)
            cbar.ax.yaxis.label.set_color('white')
            
            ax.view_init(elev=30, azim=-55)
            
            plt.savefig(out_png_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
            plt.close(fig)
            if feedback:
                feedback.pushInfo(f"Static 3D Seabed Plot generated successfully. {ve_label}")
            return True
    except Exception as e:
        if feedback:
            feedback.pushWarning(f"Failed to generate static 3D Seabed PNG: {str(e)}")
        return False


def generate_html_dashboard(out_dir, p3_dir, p4_dir=None, spatial_cv_p3=True, spatial_cv_p4=True, enable_ransac=False, filter_mode=0, field_depth=None, field_weight=None, collision_handling_idx=0, log_path=None, feedback=None, raster_name="Satellite Imagery", train_name="ICESat-2 (ATL24) LiDAR", test_name="In-situ Echosounder Surveys", final_raster_path=None, is_spatiospectral=False, p2_dir=None):
    import json
    import csv
    benchmark_csv = os.path.join(p3_dir, "3_All_Algorithms_Benchmark.csv")
    p4_benchmark_csv = os.path.join(p4_dir, "4_All_Algorithms_Benchmark.csv") if p4_dir else None
    
    cv_type_p3 = "Spatial K-Fold Cross Validation" if spatial_cv_p3 else "Standard Random K-Fold Cross Validation"
    cv_type_p4 = "Spatial K-Fold Cross Validation" if spatial_cv_p4 else "Standard Random K-Fold Cross Validation"
    
    folder_name = os.path.basename(out_dir)
    folder_url = f"file:///{out_dir.replace(chr(92), '/')}"
    
    # Downsample final depth map for 3D web rendering
    import json
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    
    p3_rel = os.path.relpath(p3_dir, out_dir).replace('\\', '/') if p3_dir else "Phase_03_Initial_Modeling"
    p4_rel = os.path.relpath(p4_dir, out_dir).replace('\\', '/') if p4_dir else "Phase_04_Adaptive_Refinement"
    
    if not final_raster_path:
        if p4_dir:
            for candidate in ["Phase04_Final_Depth_OSW_Clipped.tif", "Final_Depth_Cleaned.tif", "Phase4_Adaptive_Depth.tif", "4_Refined_Depth.tif"]:
                test_path = os.path.join(p4_dir, candidate)
                if os.path.exists(test_path):
                    final_raster_path = test_path
                    break
        if not final_raster_path:
            for candidate in ["Phase03_Depth_OSW_Clipped.tif", "Phase3_Depth_Cleaned.tif", "3_Initial_Global_Depth.tif", "3_Best_Depth_Map.tif"]:
                test_path = os.path.join(p3_dir, candidate)
                if os.path.exists(test_path):
                    final_raster_path = test_path
                    break

    z_data_json = "[]"
    x_coords_json = "[]"
    y_coords_json = "[]"
    plotly_aspect_json = "{}"
    plotly_x_title = "X (Grid)"
    plotly_y_title = "Y (Grid)"
    if final_raster_path and os.path.exists(final_raster_path):
        try:
            grid_size = 100
            with rasterio.open(final_raster_path) as src:
                data = src.read(
                    1,
                    out_shape=(grid_size, grid_size),
                    resampling=Resampling.bilinear
                )
                nodata = src.nodata if src.nodata is not None else -9999.0
                bounds = src.bounds
                x_extent = bounds.right - bounds.left
                y_extent = bounds.top - bounds.bottom
                
                data_float = data.astype(float)
                data_float[data_float == nodata] = np.nan
                grid_list = [[(val if np.isfinite(val) else None) for val in row] for row in data_float]
                z_data_json = json.dumps(grid_list)
                
                # Real-world coordinate arrays
                x_arr = np.linspace(0, x_extent, grid_size).tolist()
                y_arr = np.linspace(y_extent, 0, grid_size).tolist()
                x_coords_json = json.dumps([round(v, 1) for v in x_arr])
                y_coords_json = json.dumps([round(v, 1) for v in y_arr])
                
                # Auto Vertical Exaggeration & Plotly aspect ratio
                target_z_fraction = 0.35
                valid_z = data_float[np.isfinite(data_float)]
                if len(valid_z) > 0:
                    z_range = float(np.nanmax(valid_z) - np.nanmin(valid_z))
                    z_range = max(z_range, 0.1)
                    horizontal_extent = max(x_extent, y_extent)
                    ve = (target_z_fraction * horizontal_extent) / z_range
                    ve = max(2.0, ve)
                else:
                    z_range = 1.0
                    ve = 3.0
                    
                # Normalize aspect: longest horizontal side = 1, Z = target fraction
                max_h = max(x_extent, y_extent)
                aspect = {"x": round(x_extent / max_h, 4), "y": round(y_extent / max_h, 4), "z": round(target_z_fraction, 4)}
                plotly_aspect_json = json.dumps(aspect)
                
                if x_extent > 2000:
                    plotly_x_title = f"Easting ({x_extent/1000:.1f} km)"
                else:
                    plotly_x_title = f"Easting ({x_extent:.0f} m)"
                if y_extent > 2000:
                    plotly_y_title = f"Northing ({y_extent/1000:.1f} km)"
                else:
                    plotly_y_title = f"Northing ({y_extent:.0f} m)"
        except Exception as e:
            if feedback:
                feedback.pushWarning(f"Failed to extract 3D Seabed data for dashboard: {str(e)}")

    rows_p3_html = ""
    rows_p4_html = ""
    rows_strat_html = ""
    has_strat = False
    
    # Declare defaults for filtering stats (used in PDF report)
    pt_count = "N/A"
    depth_min = "N/A"
    depth_max = "N/A"
    weight_min = "N/A"
    weight_max = "N/A"
    actual_pt_count = "N/A"
    collision_handling = "Bypassed"
    filter_mode_name = "Disabled"
    has_weight_stats = False
    strat_rows = []
    
    best_algo = "N/A"
    best_r2 = -9999.0
    best_rmse = 9999.0
    
    import csv
    
    # -------------------------------------------------------------
    # Read and Sort Phase 03 Leaderboard
    # -------------------------------------------------------------
    p3_models = []
    if os.path.exists(benchmark_csv):
        try:
            with open(benchmark_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    algo = row.get("Algorithm", "Unknown")
                    try:
                        r2 = float(row.get("R2", 0.0))
                    except ValueError:
                        r2 = -9999.0
                    try:
                        rmse = float(row.get("RMSE", 0.0))
                    except ValueError:
                        rmse = 9999.0
                    try:
                        wmape = float(row.get("wMAPE", 0.0))
                    except ValueError:
                        wmape = 0.0
                    
                    p3_models.append({
                        "Algorithm": algo,
                        "R2": r2,
                        "RMSE": rmse,
                        "wMAPE": wmape
                    })
        except Exception:  # nosec B110
            pass
            
    p3_models.sort(key=lambda x: x["R2"], reverse=True)
    
    for m in p3_models:
        algo = m["Algorithm"]
        r2 = m["R2"]
        rmse = m["RMSE"]
        wmape = m["wMAPE"]
        
        if r2 > best_r2:
            best_r2 = r2
            best_algo = algo
            best_rmse = rmse
            
        algo_folder = algo.replace(" ", "_")
        if algo.startswith("Ensemble"):
            algo_folder = "Ensemble_Model"
            
        rows_p3_html += f"""
        <tr class="hover:bg-slate-700/50 transition-colors border-b border-slate-700/30">
            <td class="px-6 py-4 text-sm font-semibold text-slate-200">{algo}</td>
            <td class="px-6 py-4 text-sm font-medium text-emerald-400">{r2:.4f}</td>
            <td class="px-6 py-4 text-sm font-medium text-blue-400">{rmse:.2f}m</td>
            <td class="px-6 py-4 text-sm font-medium text-indigo-400">{wmape:.2f}%</td>
            <td class="px-6 py-4 text-sm">
                <a href="{p3_rel}/{algo_folder}/Validation_Scatter_Plot.png" target="_blank" class="text-xs text-sky-400 hover:text-sky-300 font-semibold underline">View Plot</a>
            </td>
        </tr>
        """
        
    # -------------------------------------------------------------
    # Read and Sort Phase 04 Leaderboard
    # -------------------------------------------------------------
    has_p4 = False
    p4_models = []
    if p4_benchmark_csv and os.path.exists(p4_benchmark_csv):
        has_p4 = True
        try:
            with open(p4_benchmark_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    algo = row.get("Algorithm", "Unknown")
                    try:
                        r2 = float(row.get("R2", 0.0))
                    except ValueError:
                        r2 = -9999.0
                    try:
                        rmse = float(row.get("RMSE", 0.0))
                    except ValueError:
                        rmse = 9999.0
                    try:
                        wmape = float(row.get("wMAPE", 0.0))
                    except ValueError:
                        wmape = 0.0
                        
                    p4_models.append({
                        "Algorithm": algo,
                        "R2": r2,
                        "RMSE": rmse,
                        "wMAPE": wmape
                    })
        except Exception:  # nosec B110
            pass
            
    p4_models.sort(key=lambda x: x["R2"], reverse=True)
    
    for m in p4_models:
        algo = m["Algorithm"]
        r2 = m["R2"]
        rmse = m["RMSE"]
        wmape = m["wMAPE"]
        
        rows_p4_html += f"""
        <tr class="hover:bg-slate-700/50 transition-colors border-b border-slate-700/30">
            <td class="px-6 py-4 text-sm font-semibold text-slate-200">{algo}</td>
            <td class="px-6 py-4 text-sm font-medium text-emerald-400">{r2:.4f}</td>
            <td class="px-6 py-4 text-sm font-medium text-blue-400">{rmse:.2f}m</td>
            <td class="px-6 py-4 text-sm font-medium text-indigo-400">{wmape:.2f}%</td>
        </tr>
        """

    # -------------------------------------------------------------
    # Phase 02 Filtering & Shapefile Reading
    # -------------------------------------------------------------
    html_p2_section = ""
    if enable_ransac:
        p2_dir_path = p2_dir if p2_dir else os.path.join(out_dir, "Phase_02_Filtering")
        clean_shp_path = os.path.join(p2_dir_path, "2_Cleaned_Training_Data.shp")
        actual_shp_path = os.path.join(p3_dir, "3_Actual_Model_Input_Points.shp")
        
        pt_count = "N/A"
        depth_min = "N/A"
        depth_max = "N/A"
        weight_min = "N/A"
        weight_max = "N/A"
        actual_pt_count = "N/A"
        
        has_weight_stats = False
        
        if os.path.exists(clean_shp_path):
            try:
                layer = QgsVectorLayer(clean_shp_path, "Cleaned Training Data", "ogr")
                if layer and layer.isValid():
                    pt_count = layer.featureCount()
                    
                    depth_vals = []
                    weight_vals = []
                    
                    fields = layer.fields()
                    depth_idx = -1
                    weight_idx = -1
                    
                    for idx in range(fields.count()):
                        f_name = fields.at(idx).name().lower()
                        if field_depth and f_name == field_depth.lower():
                            depth_idx = idx
                        elif field_weight and f_name == field_weight.lower():
                            weight_idx = idx
                            
                    for feat in layer.getFeatures():
                        if depth_idx != -1:
                            val = feat.attribute(depth_idx)
                            if val is not None and val != NULL:
                                try:
                                    depth_vals.append(float(val))
                                except (ValueError, TypeError):
                                    pass
                        if weight_idx != -1:
                            val = feat.attribute(weight_idx)
                            if val is not None and val != NULL:
                                try:
                                    weight_vals.append(float(val))
                                except (ValueError, TypeError):
                                    pass
                                    
                    if depth_vals:
                        depth_min = f"{min(depth_vals):.2f}"
                        depth_max = f"{max(depth_vals):.2f}"
                    if weight_vals:
                        weight_min = f"{min(weight_vals):.3f}"
                        weight_max = f"{max(weight_vals):.3f}"
                        has_weight_stats = True
            except Exception:  # nosec B110
                pass
                
        if os.path.exists(actual_shp_path):
            try:
                actual_layer = QgsVectorLayer(actual_shp_path, "Actual Input Points", "ogr")
                if actual_layer and actual_layer.isValid():
                    actual_pt_count = actual_layer.featureCount()
            except Exception:
                pass
                
        collision_list_names = ["Keep All Points", "Highest Confidence", "Closest to Pixel Center", "Hybrid", "Strict Center"]
        collision_handling = collision_list_names[collision_handling_idx] if 0 <= collision_handling_idx < len(collision_list_names) else "Unknown"
        
        filter_mode_name = "Unknown"
        
        if p2_dir:
            try:
                rel_p2 = os.path.relpath(p2_dir, out_dir).replace("\\", "/")
            except Exception:
                rel_p2 = "Phase_02_Filtering"
        else:
            rel_p2 = "Phase_02_Filtering"

        if filter_mode == 0:
            filter_mode_name = "Linear RANSAC"
        elif filter_mode == 1:
            filter_mode_name = "LS Variance Fit"
        elif filter_mode == 2:
            filter_mode_name = "Huber Variance Fit"
            
        weight_stat_html = ""
        if has_weight_stats:
            weight_stat_html = f"""
                        <div class="bg-slate-800/40 border border-slate-700/20 rounded-xl p-4">
                            <span class="text-xs text-slate-400 block mb-1">Confidence / Weight Range</span>
                            <span class="text-sm font-bold text-indigo-400">{weight_min} to {weight_max}</span>
                        </div>
            """
            
        plot_1 = f"""
            <a href="{rel_p2}/2_Plot_1_Trend.png" target="_blank" class="block w-full">
                <img src="{rel_p2}/2_Plot_1_Trend.png" alt="Trend Plot" class="w-full h-auto rounded-lg border border-slate-800 hover:opacity-90 transition-opacity" onerror="this.style.display='none'"/>
            </a>"""
        plot_2 = f"""
            <a href="{rel_p2}/2_Plot_2_Variance.png" target="_blank" class="block w-full">
                <img src="{rel_p2}/2_Plot_2_Variance.png" alt="Variance Plot" class="w-full h-auto rounded-lg border border-slate-800 hover:opacity-90 transition-opacity" onerror="this.style.display='none'"/>
            </a>"""
        plot_3 = f"""
            <a href="{rel_p2}/2_Plot_3_Envelope.png" target="_blank" class="block w-full">
                <img src="{rel_p2}/2_Plot_3_Envelope.png" alt="Envelope Plot" class="w-full h-auto rounded-lg border border-slate-800 hover:opacity-90 transition-opacity" onerror="this.style.display='none'"/>
            </a>"""
            
        inner_plots = plot_1
        if filter_mode == 1:
            inner_plots += plot_2
        else:
            inner_plots += plot_3
            
        plots_html = f"""
        <div class="bg-slate-900/50 rounded-xl p-4 border border-slate-800 flex flex-col justify-center items-center w-full space-y-4">
            <h4 class="text-xs font-semibold text-slate-400 mb-1 uppercase tracking-wider">Regression & Cleaned Data Fit</h4>
            {inner_plots}
        </div>
        """

        html_p2_section = f"""
        <!-- Phase 02 Filtering Details -->
        <div class="bg-slate-800/30 border border-slate-700/30 rounded-2xl p-6 mt-8">
            <div class="flex items-center justify-between mb-4 border-b border-slate-800 pb-4">
                <div>
                    <h2 class="text-lg font-bold text-slate-100">🧹 Phase 02: Training Dataset Filtering & Uncertainty</h2>
                    <p class="text-xs text-slate-400 mt-1">Robust outlier rejection and variance/trend analysis on training data</p>
                </div>
                <span class="px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">{filter_mode_name}</span>
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                <!-- Plot Column -->
                {plots_html}
                
                <!-- Metadata Column -->
                <div class="space-y-4">
                    <h4 class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Dataset statistics (Cleaned Training Data)</h4>
                    
                    <div class="grid grid-cols-2 gap-4">
                        <div class="bg-slate-800/40 border border-slate-700/20 rounded-xl p-4">
                            <span class="text-xs text-slate-400 block mb-1">Total Training Points</span>
                            <span class="text-xl font-bold text-slate-200">{pt_count}</span>
                        </div>
                        <div class="bg-slate-800/40 border border-slate-700/20 rounded-xl p-4">
                            <span class="text-xs text-slate-400 block mb-1">Depth Range</span>
                            <span class="text-sm font-bold text-sky-400">{depth_min}m to {depth_max}m</span>
                        </div>
                        {weight_stat_html}
                    </div>
                    
                    <div class="bg-slate-900/60 rounded-xl p-4 border border-slate-800 space-y-2">
                        <span class="text-xs font-semibold text-amber-400 block mb-1">💡 Data Processing & Model Inputs</span>
                        <div class="grid grid-cols-2 gap-2 text-xs">
                            <div class="text-slate-400">Collision Handling:</div>
                            <div class="text-slate-200 font-semibold">{collision_handling}</div>
                            
                            <div class="text-slate-400">Actual Model Input Points:</div>
                            <div class="text-slate-200 font-semibold">{actual_pt_count}</div>
                        </div>
                        <p class="text-[11px] text-slate-400 leading-relaxed font-normal pt-1 border-t border-slate-800">
                            The original cleaned training points are aggregated and processed using the selected collision handling method ({collision_handling}) to resolve duplicate points falling inside the same raster pixel. This results in {actual_pt_count} unique training samples used directly in model optimization and cross-validation.
                        </p>
                    </div>
                </div>
            </div>
        </div>
        """

    stratified_csv = os.path.join(out_dir, "5_Stratified_Error_Analysis.csv")
    if os.path.exists(stratified_csv):
        has_strat = True
        try:
            with open(stratified_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    model = row.get("Model", "Unknown")
                    depth_bin = row.get("Depth_Bin", "Unknown")
                    count = row.get("Count", "0")
                    mean_depth = float(row.get("Mean_Depth", 0.0))
                    rmse = float(row.get("RMSE", 0.0))
                    model_tvu = float(row.get("Model_TVU_95", 0.0))
                    iho_limit = float(row.get("IHO_TVU_Limit", 0.0))
                    iho_order = row.get("IHO_Order", "Unknown")
                    uses = row.get("Suggested_Uses", "")
                    
                    strat_rows.append({
                        "Model": model,
                        "Depth_Bin": depth_bin,
                        "Count": count,
                        "Mean_Depth": mean_depth,
                        "RMSE": rmse,
                        "Model_TVU_95": model_tvu,
                        "IHO_TVU_Limit": iho_limit,
                        "IHO_Order": iho_order,
                        "Suggested_Uses": uses
                    })
                    
                    if iho_order == "Special Order":
                        order_badge = '<span class="px-2 py-0.5 rounded text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">Special Order</span>'
                    elif iho_order == "Order 1a/1b":
                        order_badge = '<span class="px-2 py-0.5 rounded text-xs font-semibold bg-sky-500/10 text-sky-400 border border-sky-500/20">Order 1a/1b</span>'
                    elif iho_order == "Order 2":
                        order_badge = '<span class="px-2 py-0.5 rounded text-xs font-semibold bg-amber-500/10 text-amber-400 border border-emerald-500/20">Order 2</span>'
                    else:
                        order_badge = '<span class="px-2 py-0.5 rounded text-xs font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/20">Out of Spec</span>'
                        
                    rows_strat_html += f"""
                    <tr class="hover:bg-slate-700/50 transition-colors border-b border-slate-700/30">
                        <td class="px-6 py-4 text-xs font-semibold text-slate-300">{model}</td>
                        <td class="px-6 py-4 text-xs font-medium text-slate-200">{depth_bin}</td>
                        <td class="px-6 py-4 text-xs text-slate-400">{count}</td>
                        <td class="px-6 py-4 text-xs text-slate-400">{mean_depth:.2f}m</td>
                        <td class="px-6 py-4 text-xs text-slate-400">{rmse:.3f}m</td>
                        <td class="px-6 py-4 text-xs font-semibold text-sky-400">{model_tvu:.3f}m</td>
                        <td class="px-6 py-4 text-xs text-slate-500">{iho_limit:.3f}m</td>
                        <td class="px-6 py-4 text-xs">{order_badge}</td>
                        <td class="px-6 py-4 text-xs text-slate-400 max-w-xs truncate" title="{uses}">{uses}</td>
                    </tr>
                    """
        except Exception:  # nosec B110
            pass

    html_strat_section = ""
    if has_strat:
        html_strat_section = f"""
        <!-- IHO S-44 Standards Compliance & B-13 Applications -->
        <div class="bg-slate-800/30 border border-slate-700/30 rounded-2xl p-6 mt-8">
            <div class="flex items-center justify-between mb-4 border-b border-slate-800 pb-4">
                <div>
                    <h2 class="text-lg font-bold text-slate-100">🌊 IHO S-44 Hydrographic Standards & Industrial Uses</h2>
                    <p class="text-xs text-slate-400 mt-1">Conformity analysis of prediction uncertainty against International Hydrographic Organization (IHO) orders</p>
                </div>
            </div>
            
            <!-- Legal Disclaimer Warning Box -->
            <div class="p-4 bg-amber-500/10 border border-amber-500/20 text-amber-300 rounded-xl mb-6 text-xs leading-relaxed">
                <strong>⚠️ IMPORTANT LEGAL DISCLAIMER:</strong>
                This analysis, including IHO S-44 conformity assessment and recommended industrial applications, is generated automatically for reference, planning, and scientific guidance purposes only. It does not constitute official hydrographic data or a certified navigational product. This information <strong>must not</strong> be used for direct vessel navigation, marine safety operations, or any legal hydrographic charting applications. Always consult official charts published by authorized national hydrographic authorities.
            </div>
            
            <div class="overflow-x-auto">
                <table class="min-w-full divide-y divide-slate-800 text-left">
                    <thead class="bg-slate-800/20 text-slate-400 text-xs font-semibold uppercase tracking-wider">
                        <tr>
                            <th class="px-6 py-3">Model</th>
                            <th class="px-6 py-3">Depth Bin</th>
                            <th class="px-6 py-3">Samples</th>
                            <th class="px-6 py-3">Mean Depth</th>
                            <th class="px-6 py-3">RMSE</th>
                            <th class="px-6 py-3">Model TVU (95%)</th>
                            <th class="px-6 py-3">IHO Limit (95%)</th>
                            <th class="px-6 py-3">Achieved Order</th>
                            <th class="px-6 py-3">Suggested Applications</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-slate-800 bg-transparent text-xs text-slate-300">
                        {rows_strat_html}
                    </tbody>
                </table>
            </div>
        </div>
        """

    has_val_plots = os.path.exists(os.path.join(out_dir, "5_Plot_Scatter_Comparison.png"))
    main_grid_col_class = "lg:col-span-2" if has_val_plots else "lg:col-span-3"
    best_r2_str = f"{best_r2:.4f}" if best_r2 != -9999.0 else "N/A"
    best_rmse_str = f"{best_rmse:.2f}m" if best_rmse != 9999.0 else "N/A"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bathymetrix-AI Project Validation Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {{
            background-color: #0f172a;
        }}
    </style>
</head>
<body class="text-slate-100 font-sans antialiased min-h-screen">
    <div class="max-w-6xl mx-auto px-4 py-8">
        <!-- Header -->
        <header class="flex items-center justify-between mb-8 pb-6 border-b border-slate-800">
            <div>
                <h1 class="text-3xl font-extrabold tracking-tight bg-gradient-to-r from-sky-400 to-blue-500 bg-clip-text text-transparent">🛰&nbsp;Bathymetrix-AI</h1>
                <p class="text-slate-400 mt-1 text-sm font-medium">Satellite-Derived Bathymetry (SDB) Project Dashboard</p>
                <div class="mt-2 text-xs text-slate-400">
                    Project Folder: <a href="{folder_url}" target="_blank" class="text-sky-400 hover:underline font-mono text-sm font-semibold">{folder_name}</a>
                </div>
            </div>
            <div class="text-right flex flex-col items-end">
                <div class="text-xs text-slate-400 mb-1.5 font-medium">
                    Tool: <span class="text-sky-400 font-bold">SDB MasterFlow</span> | Developer: <span class="text-slate-200 font-semibold">Mohamed Aly Nasef</span>
                </div>
                <span class="px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 w-fit">Pipeline Completed</span>
            </div>
        </header>

        <!-- Quick Metrics -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8" style="display: {'none' if is_spatiospectral else 'grid'};">
            <div class="bg-slate-800/40 border border-slate-700/50 rounded-2xl p-6 backdrop-blur-sm">
                <h3 class="text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">Best Performing Algorithm</h3>
                <div class="text-2xl font-bold text-slate-100">{best_algo}</div>
            </div>
            <div class="bg-slate-800/40 border border-slate-700/50 rounded-2xl p-6 backdrop-blur-sm">
                <h3 class="text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">AutoML Best R²</h3>
                <div class="text-2xl font-bold text-emerald-400">{best_r2_str}</div>
            </div>
            <div class="bg-slate-800/40 border border-slate-700/50 rounded-2xl p-6 backdrop-blur-sm">
                <h3 class="text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">AutoML Best RMSE</h3>
                <div class="text-2xl font-bold text-blue-400">{best_rmse_str}</div>
            </div>
        </div>

        {html_p2_section}

        <!-- Main Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <!-- Left Column: Leaderboards -->
            <div class="{main_grid_col_class} space-y-8">
                <!-- Phase 03 Leaderboard -->
                <div class="bg-slate-800/30 border border-slate-700/30 rounded-2xl overflow-hidden" style="display: {'none' if is_spatiospectral else 'block'};">
                    <div class="px-6 py-5 border-b border-slate-800 bg-slate-800/50 flex justify-between items-center flex-wrap gap-2">
                        <div>
                            <h2 class="text-lg font-bold text-slate-100">🏆 Phase 03: Initial Modeling Leaderboard</h2>
                            <p class="text-xs text-slate-400 mt-1">Cross-Validation performance of benchmarked algorithms</p>
                        </div>
                        <span class="px-2 py-0.5 rounded text-[10px] font-semibold bg-sky-500/10 text-sky-400 border border-sky-500/20">{cv_type_p3}</span>
                    </div>
                    <div class="overflow-x-auto">
                        <table class="min-w-full divide-y divide-slate-800 text-left">
                            <thead class="bg-slate-800/20 text-slate-400 text-xs font-semibold uppercase tracking-wider">
                                <tr>
                                    <th class="px-6 py-3">Algorithm</th>
                                    <th class="px-6 py-3">R²</th>
                                    <th class="px-6 py-3">RMSE</th>
                                    <th class="px-6 py-3">wMAPE</th>
                                    <th class="px-6 py-3">Details</th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-slate-800 bg-transparent">
                                {rows_p3_html if rows_p3_html else '<tr><td colspan="5" class="px-6 py-4 text-center text-slate-400 text-sm">No Phase 03 results found.</td></tr>'}
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- Phase 04 Leaderboard (Conditionally Shown) -->
                {f'''<div class="bg-slate-800/30 border border-slate-700/30 rounded-2xl overflow-hidden">
                    <div class="px-6 py-5 border-b border-slate-800 bg-slate-800/50 flex justify-between items-center flex-wrap gap-2">
                        <div>
                            <h2 class="text-lg font-bold text-slate-100">⚡ Phase 04: Depth-Dependent Residual Calibration Leaderboard</h2>
                            <p class="text-xs text-slate-400 mt-1">Cross-Validation/Retraining performance after Depth-Dependent Residual Calibration</p>
                        </div>
                        <span class="px-2 py-0.5 rounded text-[10px] font-semibold bg-violet-500/10 text-violet-400 border border-violet-500/20">{cv_type_p4}</span>
                    </div>
                    <div class="overflow-x-auto">
                        <table class="min-w-full divide-y divide-slate-800 text-left">
                            <thead class="bg-slate-800/20 text-slate-400 text-xs font-semibold uppercase tracking-wider">
                                <tr>
                                    <th class="px-6 py-3">Algorithm</th>
                                    <th class="px-6 py-3">R²</th>
                                    <th class="px-6 py-3">RMSE</th>
                                    <th class="px-6 py-3">wMAPE</th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-slate-800 bg-transparent">
                                {rows_p4_html if rows_p4_html else '<tr><td colspan="4" class="px-6 py-4 text-center text-slate-400 text-sm">No Phase 04 results found.</td></tr>'}
                            </tbody>
                        </table>
                    </div>
                </div>''' if has_p4 and not is_spatiospectral else ''}
            </div>

            {f'''<!-- Right Column: Final Validation Outputs -->
            <div class="space-y-6">
                <div class="bg-slate-800/30 border border-slate-700/30 rounded-2xl p-6">
                    <h2 class="text-lg font-bold text-slate-100 mb-4">📈 Phase 05 Validation Plots</h2>
                    
                    <div class="space-y-4">
                           <div>
                               <h4 class="text-sm font-semibold text-slate-300 mb-2">Density Scatter Plot</h4>
                               <div class="bg-slate-900 rounded-lg overflow-hidden border border-slate-800">
                                   <a href="5_Plot_Scatter_Comparison.png" target="_blank">
                                       <img src="5_Plot_Scatter_Comparison.png" alt="Density Scatter Plot" class="w-full h-auto hover:opacity-90 transition-opacity" onerror="this.src='https://placehold.co/400x300/1e293b/94a3b8?text=Scatter+Plot+Not+Found'"/>
                                   </a>
                                </div>
                           </div>
                           
                           <div>
                               <h4 class="text-sm font-semibold text-slate-300 mb-2">Error Distribution Histogram</h4>
                               <div class="bg-slate-900 rounded-lg overflow-hidden border border-slate-800">
                                   <a href="5_Plot_Error_Histogram.png" target="_blank">
                                       <img src="5_Plot_Error_Histogram.png" alt="Error Histogram" class="w-full h-auto hover:opacity-90 transition-opacity" onerror="this.src='https://placehold.co/400x300/1e293b/94a3b8?text=Histogram+Not+Found'"/>
                                   </a>
                                </div>
                           </div>

                           <div>
                               <h4 class="text-sm font-semibold text-slate-300 mb-2">Residuals vs Depth Plot</h4>
                               <div class="bg-slate-900 rounded-lg overflow-hidden border border-slate-800">
                                   <a href="5_Plot_Residuals.png" target="_blank">
                                       <img src="5_Plot_Residuals.png" alt="Residuals Plot" class="w-full h-auto hover:opacity-90 transition-opacity" onerror="this.src='https://placehold.co/400x300/1e293b/94a3b8?text=Residuals+Plot+Not+Found'"/>
                                   </a>
                               </div>
                           </div>
                    </div>
                </div>
            </div>''' if has_val_plots else ''}
        </div>
        
        <!-- 3D Seabed Viewer & Plot Section -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-8">
            <!-- Left: Interactive 3D Viewer -->
            <div class="bg-slate-800/30 border border-slate-700/30 rounded-2xl p-6">
                <h2 class="text-lg font-bold text-slate-100 mb-2">🔮 Interactive 3D Seabed Viewer</h2>
                <p class="text-xs text-slate-400 mb-4">Click and drag to rotate, scroll to zoom. Double click to reset view.</p>
                <div id="seabed-3d-viewer" style="width: 100%; height: 450px;" class="rounded-xl overflow-hidden bg-slate-950 border border-slate-800"></div>
            </div>
            <!-- Right: Static 3D Seabed Plot (PNG) -->
            <div class="bg-slate-800/30 border border-slate-700/30 rounded-2xl p-6">
                <h2 class="text-lg font-bold text-slate-100 mb-2">📷 Static 3D Seabed Elevation Model</h2>
                <p class="text-xs text-slate-400 mb-4">High-resolution 3D surface plot generated using Spectral reversed colormap.</p>
                <div class="bg-slate-900 rounded-xl overflow-hidden border border-slate-800 h-[450px] flex items-center justify-center">
                    <a href="5_Plot_3D_Seabed.png" target="_blank" class="w-full h-full flex items-center justify-center p-2">
                        <img src="5_Plot_3D_Seabed.png" alt="Static 3D Seabed Plot" class="max-w-full max-h-full object-contain hover:opacity-95 transition-opacity" onerror="this.src='https://placehold.co/500x400/020617/94a3b8?text=3D+Seabed+Plot+Not+Found'"/>
                    </a>
                </div>
            </div>
        </div>
        
        {html_strat_section}
        
        <!-- Footer -->
        <footer class="mt-16 pt-6 border-t border-slate-800 text-center text-slate-500 text-xs">
            <p>Generated by SDB MasterFlow | Developer: Mohamed Aly Nasef</p>
        </footer>
    </div>
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <script>
        try {{
            var zData = {z_data_json};
            var xCoords = {x_coords_json};
            var yCoords = {y_coords_json};
            var aspectRatio = {plotly_aspect_json};
            if (zData && zData.length > 0) {{
                var data = [{{
                    z: zData,
                    x: xCoords.length > 0 ? xCoords : undefined,
                    y: yCoords.length > 0 ? yCoords : undefined,
                    type: 'surface',
                    colorscale: 'Spectral',
                    reversescale: true,
                    colorbar: {{ title: 'Depth (m)', titlefont: {{ color: '#94a3b8' }}, tickfont: {{ color: '#94a3b8' }} }},
                    contours: {{
                        z: {{ show: true, usecolormap: true, highlightcolor: "#ffffff", project: {{ z: true }} }}
                    }}
                }}];
                var sceneConfig = {{
                    xaxis: {{ title: '{plotly_x_title}', color: '#94a3b8', gridcolor: '#1e293b' }},
                    yaxis: {{ title: '{plotly_y_title}', color: '#94a3b8', gridcolor: '#1e293b' }},
                    zaxis: {{ title: 'Depth (m)', color: '#94a3b8', gridcolor: '#1e293b' }}
                }};
                if (aspectRatio && aspectRatio.x) {{
                    sceneConfig.aspectmode = 'manual';
                    sceneConfig.aspectratio = {{ x: aspectRatio.x, y: aspectRatio.y, z: aspectRatio.z }};
                }}
                var layout = {{
                    margin: {{ l: 0, r: 0, b: 0, t: 0 }},
                    paper_bgcolor: '#020617',
                    plot_bgcolor: '#020617',
                    scene: sceneConfig
                }};
                Plotly.newPlot('seabed-3d-viewer', data, layout);
            }} else {{
                document.getElementById('seabed-3d-viewer').innerHTML = '<div class="flex items-center justify-center h-full text-slate-500 text-sm">3D Seabed Data Not Available</div>';
            }}
        }} catch (e) {{
            console.error(e);
            document.getElementById('seabed-3d-viewer').innerHTML = '<div class="flex items-center justify-center h-full text-rose-500 text-sm">Failed to render 3D Viewer</div>';
        }}
    </script>
</body>
</html>
"""
    try:
        dashboard_path = os.path.join(out_dir, "SDB_Validation_Dashboard.html")
        with open(dashboard_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    except Exception:  # nosec B110
        pass

    # Generate HTML Technical Report
    try:
        generate_pdf_report(
            out_dir=out_dir,
            p3_models=p3_models,
            p4_models=p4_models,
            has_p4=has_p4,
            enable_ransac=enable_ransac,
            pt_count=pt_count,
            depth_min=depth_min,
            depth_max=depth_max,
            has_weight_stats=has_weight_stats,
            weight_min=weight_min,
            weight_max=weight_max,
            actual_pt_count=actual_pt_count,
            collision_handling=collision_handling,
            filter_mode_name=filter_mode_name,
            strat_rows=strat_rows,
            log_path=log_path,
            feedback=feedback,
            raster_name=raster_name,
            train_name=train_name,
            test_name=test_name,
            is_spatiospectral=is_spatiospectral,
            p2_dir=p2_dir
        )
    except Exception as e:
        msg = f"Failed to invoke HTML report generation: {str(e)}\n{__import__('traceback').format_exc()}"
        if log_path:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')
        if feedback is not None:
            feedback.pushWarning(msg)
        elif log_path:
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"[Warning] {msg}\n")
            except Exception:
                pass


def run_master_pipeline(algorithm, parameters, context, feedback):
    """Execute SDB Master orchestration; `algorithm` is SDBMasterOrchestrator."""
    import time
    start_time = time.time()
    start_str = time.strftime('%H:%M:%S', time.localtime(start_time))
    
    out_dir = algorithm.parameterAsString(parameters, algorithm.OUTPUT_FOLDER, context)
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "SDB_Full_Log.txt")
    
    with open(log_path, "w", encoding="utf-8") as f:
        pass
        
    feedback = LoggingFeedback(feedback, log_path)
    
    append_log("════════════════════════════════════════════════════════════", log_path, feedback)
    append_log("SDB Single Masterflow".center(60), log_path, feedback)
    append_log("════════════════════════════════════════════════════════════", log_path, feedback)
    append_log(f"Started: {start_str}", log_path, feedback)
    append_log("════════════════════════════════════════════════════════════\n", log_path, feedback)
    
    # --- Pre-Scan Validation: Check if Depth Field exists ---
    field_depth = algorithm.parameterAsString(parameters, algorithm.FIELD_DEPTH, context)
    if field_depth:
        from qgis.core import QgsVectorLayer, QgsProcessingException
        append_log("→ Validating Depth Fields...", log_path, feedback)
        
        train_val = algorithm.parameterAsVectorLayer(parameters, algorithm.INPUT_TRAIN, context)
        test_val = algorithm.parameterAsVectorLayer(parameters, algorithm.INPUT_TEST, context)
        
        try:
            field_test_depth = algorithm.parameterAsString(parameters, algorithm.FIELD_TEST_DEPTH, context)
        except Exception:
            field_test_depth = None
            
        try:
            adaptive_val = algorithm.parameterAsVectorLayer(parameters, algorithm.INPUT_ADAPTIVE_TRAIN, context)
            field_adaptive_depth = algorithm.parameterAsString(parameters, algorithm.FIELD_ADAPTIVE_DEPTH, context)
        except Exception:
            adaptive_val = None
            field_adaptive_depth = None
            
        checks = []
        if train_val:
            checks.append((train_val, "Training", field_depth))
        if test_val:
            checks.append((test_val, "Validation", field_test_depth if field_test_depth else field_depth))
        if adaptive_val:
            checks.append((adaptive_val, "Adaptive", field_adaptive_depth if field_adaptive_depth else field_depth))
        
        for layer, name, expected_field in checks:
            if layer and layer.isValid() and expected_field:
                fields = layer.fields().names()
                matched = False
                for f in fields:
                    if f.lower() == expected_field.lower() or f.lower() == expected_field.lower()[:10] or expected_field.lower().startswith(f.lower()[:8]):
                        matched = True
                        break
                if not matched:
                    err_msg = f"✗ ERROR: Depth field '{expected_field}' not found in {name} dataset."
                    append_log(err_msg, log_path, feedback)
                    raise QgsProcessingException(err_msg)
        append_log("✓ Validation completed\n", log_path, feedback)
    # --- End Pre-Scan ---
    
    p1_dir = os.path.join(out_dir, "Phase_01_Preprocessing")
    p2_dir = os.path.join(out_dir, "Phase_02_Filtering")
    p3_dir = os.path.join(out_dir, "Phase_03_Initial_Modeling")
    p4_dir = os.path.join(out_dir, "Phase_04_Adaptive_Refinement")
    
    os.makedirs(p1_dir, exist_ok=True)
    os.makedirs(p2_dir, exist_ok=True)
    os.makedirs(p3_dir, exist_ok=True)
    os.makedirs(p4_dir, exist_ok=True)

    input_raster = algorithm.parameterAsRasterLayer(
        parameters, algorithm.INPUT_RASTER, context
    )
    target_crs = input_raster.crs()
    crs_id = target_crs.authid() if (target_crs and target_crs.isValid() and target_crs.authid()) else target_crs.toWkt()
    
    corr_thresh_opt = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    corr_thresh_p4_opt = ["Use Phase 03 (-1.0)", "0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    
    p3_thresh_val = parameters.get(algorithm.FEATURE_CORR_THRESHOLD, 2)
    if isinstance(p3_thresh_val, (int, float)) and not isinstance(p3_thresh_val, bool):
        if isinstance(p3_thresh_val, float):
            p3_thresh = p3_thresh_val
        else:
            p3_thresh = float(corr_thresh_opt[p3_thresh_val]) if 0 <= p3_thresh_val < len(corr_thresh_opt) else 0.2
    else:
        p3_thresh = 0.2
        
    final_water_mask = None

    max_depth = algorithm.parameterAsDouble(
        parameters, algorithm.MAX_DEPTH_THRESHOLD, context
    )
    shrink_dist = algorithm.parameterAsDouble(
        parameters, algorithm.SHRINK_EDGE_DIST, context
    )

    remove_positives_flag = algorithm.parameterAsBool(
        parameters, algorithm.REMOVE_POSITIVES, context
    )
    apply_slope_filter = algorithm.parameterAsBool(
        parameters, algorithm.ENABLE_SLOPE_FILTER, context
    )
    slope_threshold_val = algorithm.parameterAsDouble(
        parameters, algorithm.SLOPE_THRESHOLD, context
    )

    water_mask_poly = algorithm.parameterAsVectorLayer(
        parameters, algorithm.WATER_MASK_POLY, context
    )
    if water_mask_poly:
        append_log("  [Phase 01] Pre-processing", log_path, feedback)
        append_log("      → Applying Ready-made Water Mask Polygon...", log_path, feedback)

        temp_mask_path = os.path.join(p1_dir, "temp_water_mask.gpkg")
        final_water_mask = reproject_layer_if_needed(
            water_mask_poly, target_crs, temp_mask_path, context, feedback
        )

        fixed_mask_path = os.path.join(p1_dir, "temp_water_mask_fixed.gpkg")
        fix_res = processing.run(
            "native:fixgeometries",
            {"INPUT": final_water_mask, "OUTPUT": fixed_mask_path},
            context=context,
            feedback=feedback,
            is_child_algorithm=True,
        )
        final_water_mask = fix_res["OUTPUT"]

        if shrink_dist < 0:
            append_log(
                f"      → Shrinking Water Polygon by {shrink_dist} units...",
                log_path,
                feedback,
            )
            shrunk_path = os.path.join(p1_dir, "temp_water_mask_shrunk.gpkg")
            buffer_res = processing.run(
                "native:buffer",
                {
                    "INPUT": final_water_mask,
                    "DISTANCE": shrink_dist,
                    "SEGMENTS": 5,
                    "END_CAP_STYLE": 0,
                    "JOIN_STYLE": 0,
                    "MITER_LIMIT": 2,
                    "DISSOLVE": False,
                    "OUTPUT": shrunk_path,
                },
                context=context,
                feedback=feedback,
                is_child_algorithm=True,
            )
            final_water_mask = buffer_res["OUTPUT"]



    field_depth = algorithm.parameterAsString(
        parameters, algorithm.FIELD_DEPTH, context
    )
    temp_train = os.path.join(p2_dir, "temp_reprojected_train.gpkg")

    final_train = reproject_layer_if_needed(
        algorithm.parameterAsVectorLayer(parameters, algorithm.INPUT_TRAIN, context),
        target_crs,
        temp_train,
        context,
        feedback,
    )
    final_train = filter_by_depth(
        final_train, field_depth, max_depth, context, feedback
    )

    enable_val = algorithm.parameterAsBool(
        parameters, algorithm.ENABLE_VALIDATION, context
    )
    final_test = None

    if enable_val:
        t_layer = algorithm.parameterAsVectorLayer(
            parameters, algorithm.INPUT_TEST, context
        )
        if t_layer:
            temp_test = os.path.join(out_dir, "temp_reprojected_test.gpkg")
            final_test = reproject_layer_if_needed(
                t_layer, target_crs, temp_test, context, feedback
            )
            final_test = filter_by_depth(
                final_test,
                algorithm.parameterAsString(
                    parameters, algorithm.FIELD_TEST_DEPTH, context
                ),
                max_depth,
                context,
                feedback,
            )
        else:
            enable_val = False

    enable_preproc = algorithm.parameterAsBool(
        parameters, algorithm.ENABLE_PREPROCESSING, context
    )

    if enable_preproc:
        append_log("  [Phase 01] Pre-processing", log_path, feedback)
        append_log("      → Running extraction and masking...", log_path, feedback)
        p1 = processing.run(
            "sdb_tools:sdb_phase1_preprocessing",
            {
                "INPUT_RASTER": input_raster,
                "COASTAL_BAND": parameters[algorithm.COASTAL_BAND],
                "BLUE_BAND": parameters[algorithm.BLUE_BAND],
                "GREEN_BAND": parameters[algorithm.GREEN_BAND],
                "RED_BAND": parameters[algorithm.RED_BAND],
                "NIR_BAND": parameters[algorithm.NIR_BAND],
                "SWIR_BAND": parameters[algorithm.SWIR_BAND],
                "APPLY_SUNGLINT": parameters[algorithm.APPLY_SUNGLINT],
                "SUNGLINT_PERCENTILE": parameters[algorithm.SUNGLINT_PERCENTILE],
                "INPUT_WATER_POLY": final_water_mask if water_mask_poly else None,
                "ENABLE_MASKING": parameters[algorithm.ENABLE_MASKING],
                "MASKING_METHOD": parameters[algorithm.MASKING_METHOD],
                "MANUAL_THRESHOLD": parameters[algorithm.MANUAL_THRESHOLD],
                "OTSU_ADJUSTMENT": parameters[algorithm.OTSU_ADJUSTMENT],
                "MASK_KERNEL_SIZE": parameters[algorithm.MASK_KERNEL_SIZE],
                "FEATURE_SELECTION": parameters[algorithm.FEATURE_SELECTION],
                "ENABLE_BAND_CALC": parameters[algorithm.ENABLE_BAND_CALC],
                "BAND_MATH_FORMULA": parameters[algorithm.BAND_MATH_FORMULA],
                "APPLY_DEEPWATER": parameters[algorithm.APPLY_DEEPWATER],
                "DEEPWATER_METHOD": parameters[algorithm.DEEPWATER_METHOD],
                "DEEPWATER_ROI": parameters.get(algorithm.DEEPWATER_ROI, None),
                "NIR_PERCENTILE_OSW": parameters[algorithm.NIR_PERCENTILE_OSW],
                "OSW_MEDIAN_SIZE": parameters[algorithm.OSW_MEDIAN_SIZE],
                "FILL_INTERNAL_HOLES": parameters.get(algorithm.FILL_INTERNAL_HOLES, True),
                "EXTRACT_POLYGON": parameters.get(algorithm.EXTRACT_POLYGON, True),
                "NUM_THREADS": parameters[algorithm.NUM_THREADS],
                "OUTPUT_FOLDER": p1_dir,
            },
            context=context,
            feedback=feedback,
            is_child_algorithm=True,
        )
    else:
        append_log("  [Phase 01] Pre-processing", log_path, feedback)
        append_log("      → Skipped by User.", log_path, feedback)
        p1 = {
            "OUTPUT_FEATURES": input_raster.source(),
            "OUTPUT_MASK": None,
            "OUTPUT_OSW_POLY": None
        }


    path_clean = final_train
    if algorithm.parameterAsBool(parameters, algorithm.ENABLE_RANSAC, context):
        append_log("  [Phase 02] Filtering", log_path, feedback)
        append_log("      → Removing outliers...", log_path, feedback)
        p2 = processing.run(
            "sdb_tools:sdb_02_filtering",
            {
                "INPUT_STACK": p1["OUTPUT_FEATURES"],
                "INPUT_POINTS": final_train,
                "FIELD_DEPTH": field_depth,
                "BLUE_BAND": parameters.get(getattr(algorithm, "FILTER_NUMERATOR_BAND", "BLUE_BAND"), parameters.get("BLUE_BAND")),
                "GREEN_BAND": parameters.get(getattr(algorithm, "FILTER_DENOMINATOR_BAND", "GREEN_BAND"), parameters.get("GREEN_BAND")),
                "FILTER_MODE": parameters[algorithm.FILTER_MODE],
                "RESIDUAL_THRESHOLD": parameters[algorithm.RANSAC_THRESHOLD],
                "RANSAC_MAX_TRIALS": parameters[algorithm.RANSAC_MAX_TRIALS],
                "OUTPUT_FOLDER": p2_dir,
            },
            context=context,
            feedback=feedback,
            is_child_algorithm=True,
        )
        path_clean = p2["OUTPUT_CLEAN_VEC"]

    append_log("  ✓ Phase 02 completed\n", log_path, feedback)

    append_log("════════════════════════════════════════════════════════════", log_path, feedback)
    append_log("Global Phase 03 | Initial Modeling".center(60), log_path, feedback)
    append_log("════════════════════════════════════════════════════════════", log_path, feedback)
    p3_params = {
        "INPUT_STACK": p1["OUTPUT_FEATURES"],
        "INPUT_POINTS": path_clean,
        "FIELD_DEPTH": field_depth,
        "FIELD_WEIGHT": algorithm.parameterAsString(
            parameters, algorithm.FIELD_WEIGHT, context
        ),
        "SELECTED_ALGOS": parameters[algorithm.SELECTED_ALGOS],
        "OPTIMIZER_METHOD": parameters[algorithm.OPTIMIZER_METHOD],
        "COLLISION_HANDLING": parameters[algorithm.COLLISION_HANDLING],
        "N_ITERATIONS": parameters[algorithm.N_ITERATIONS],
        "MEDIAN_SIZE": parameters[algorithm.MEDIAN_SIZE],
        "FEATURE_CORR_THRESHOLD": p3_thresh,
        "FEATURE_CORR_METHOD": parameters.get(algorithm.FEATURE_CORR_METHOD, 3),
        "OUTPUT_FOLDER": p3_dir,
        "LOG_FILE": log_path,
        "PARAM_RF": parameters[algorithm.PARAM_RF],
        "PARAM_GB": parameters[algorithm.PARAM_GB],
        "PARAM_ET": parameters[algorithm.PARAM_ET],
        "PARAM_SVR": parameters[algorithm.PARAM_SVR],
        "PARAM_MLP": parameters[algorithm.PARAM_MLP],
        "PARAM_RIDGE": parameters.get(algorithm.PARAM_RIDGE, ""),
        "PARAM_LASSO": parameters.get(algorithm.PARAM_LASSO, ""),
        "PARAM_ELASTICNET": parameters.get(algorithm.PARAM_ELASTICNET, ""),
        "PARAM_KNN": parameters.get(algorithm.PARAM_KNN, ""),
        "PARAM_DT": parameters.get(algorithm.PARAM_DT, ""),
        "PARAM_HUBER": parameters.get(algorithm.PARAM_HUBER, ""),
        "PARAM_XGB": parameters.get(algorithm.PARAM_XGB, ""),
        "PARAM_LGBM": parameters.get(algorithm.PARAM_LGBM, ""),
        "PARAM_CATBOOST": parameters.get(algorithm.PARAM_CATBOOST, ""),
        "ENABLE_ENSEMBLE": parameters.get(algorithm.ENABLE_ENSEMBLE, False),
        "ENSEMBLE_METHOD": parameters.get(algorithm.ENSEMBLE_METHOD, 0),
        "ENSEMBLE_SIZE": parameters.get(algorithm.ENSEMBLE_SIZE, 3),
        "SPATIAL_CV": parameters.get(algorithm.SPATIAL_CV_P3, False),
        "TRAIN_TEST_SPLIT": parameters[algorithm.TRAIN_TEST_SPLIT],
        "RANDOM_STATE": parameters[algorithm.RANDOM_STATE],
        "NUM_THREADS": parameters[algorithm.NUM_THREADS],
        "OUTPUT_FORMAT": parameters[algorithm.OUTPUT_FORMAT],
    }
    if p1.get("OUTPUT_MASK"):
        p3_params["INPUT_MASK"] = p1["OUTPUT_MASK"]

    p3 = processing.run(
        "sdb_tools:sdb_03_initial_modeling",
        p3_params,
        context=context,
        feedback=feedback,
        is_child_algorithm=True,
    )

    if "BEST_R2" in p3:
        append_log(f"      → R2: {p3['BEST_R2']:.4f}", log_path, feedback)
        append_log("  ✓ Phase 03 completed\n", log_path, feedback)

    path_refined = None
    
    # [USER REQUIREMENT] Phase 04 is STRICTLY for Control Points. 
    # If no control points are provided, it is bypassed completely.
    ad_layer = algorithm.parameterAsVectorLayer(parameters, algorithm.INPUT_ADAPTIVE_TRAIN, context)
    
    if ad_layer:
        append_log("  [Phase 04] Adaptive Refinement", log_path, feedback)
        append_log("      → Control Points found. Executing...", log_path, feedback)

        temp_adapt = os.path.join(p4_dir, "temp_reprojected_adaptive.gpkg")
        final_ad = reproject_layer_if_needed(
            ad_layer, target_crs, temp_adapt, context, feedback
        )
        field_ad_depth = algorithm.parameterAsString(
            parameters, algorithm.FIELD_ADAPTIVE_DEPTH, context
        )
        final_ad = filter_by_depth(
            final_ad, field_ad_depth, max_depth, context, feedback
        )

        p4_thresh_idx = parameters.get(algorithm.FEATURE_CORR_THRESHOLD_P4, 0)
        p4_method = parameters.get(algorithm.FEATURE_CORR_METHOD_P4, 3)
        
        if isinstance(p4_thresh_idx, (int, float)) and not isinstance(p4_thresh_idx, bool):
            if isinstance(p4_thresh_idx, float):
                p4_thresh = p4_thresh_idx
            else:
                p4_thresh = -1.0 if p4_thresh_idx == 0 else float(corr_thresh_p4_opt[p4_thresh_idx])
        else:
            p4_thresh = -1.0

        if p4_thresh < 0:
            p4_thresh = p3_thresh
            p4_method = parameters.get(algorithm.FEATURE_CORR_METHOD, 3)

        p4_params = {
            "INPUT_GLOBAL_RASTER": p3["OUTPUT_DEPTH_MAP"],
            "INPUT_ORIGINAL_FEAT": p1["OUTPUT_FEATURES"],
            "INPUT_TRAIN": final_ad,
            "FIELD_TRAIN": field_ad_depth,
            "STACK_COMPONENTS": parameters.get(algorithm.STACK_COMPONENTS_P4, [0, 1, 2]),
            "SELECTED_ALGOS": parameters[algorithm.SELECTED_ALGOS],
            "OPTIMIZER_METHOD": parameters[algorithm.OPTIMIZER_METHOD],
            "COLLISION_HANDLING": parameters[algorithm.COLLISION_HANDLING],
            "N_ITERATIONS": parameters[algorithm.N_ITERATIONS],
            "MEDIAN_SIZE": parameters[algorithm.MEDIAN_SIZE],
            "FEATURE_CORR_THRESHOLD": p4_thresh,
            "FEATURE_CORR_METHOD": p4_method,
            "OUTPUT_FOLDER": p4_dir,
            "LOG_FILE": log_path,
            "PARAM_RF": parameters[algorithm.PARAM_RF],
            "PARAM_GB": parameters[algorithm.PARAM_GB],
            "PARAM_ET": parameters[algorithm.PARAM_ET],
            "PARAM_SVR": parameters[algorithm.PARAM_SVR],
            "PARAM_MLP": parameters[algorithm.PARAM_MLP],
            "PARAM_RIDGE": parameters.get(algorithm.PARAM_RIDGE, ""),
            "PARAM_LASSO": parameters.get(algorithm.PARAM_LASSO, ""),
            "PARAM_ELASTICNET": parameters.get(algorithm.PARAM_ELASTICNET, ""),
            "PARAM_KNN": parameters.get(algorithm.PARAM_KNN, ""),
            "PARAM_DT": parameters.get(algorithm.PARAM_DT, ""),
            "PARAM_HUBER": parameters.get(algorithm.PARAM_HUBER, ""),
            "PARAM_XGB": parameters.get(algorithm.PARAM_XGB, ""),
            "PARAM_LGBM": parameters.get(algorithm.PARAM_LGBM, ""),
            "PARAM_CATBOOST": parameters.get(algorithm.PARAM_CATBOOST, ""),
            "ENABLE_ENSEMBLE": parameters.get(algorithm.ENABLE_ENSEMBLE_P4, False),
            "ENSEMBLE_METHOD": parameters.get(algorithm.ENSEMBLE_METHOD_P4, 0),
            "ENSEMBLE_SIZE": parameters.get(algorithm.ENSEMBLE_SIZE_P4, 3),
            "RESIDUAL_INTERP_METHOD": parameters.get(algorithm.RESIDUAL_INTERP_METHOD, 0),
            "KNN_NEIGHBORS": parameters.get(algorithm.KNN_NEIGHBORS, 15),
            "SPATIAL_CV": parameters.get(algorithm.SPATIAL_CV_P4, False),
            "TRAIN_TEST_SPLIT": parameters[algorithm.TRAIN_TEST_SPLIT],
            "RANDOM_STATE": parameters[algorithm.RANDOM_STATE],
            "NUM_THREADS": parameters[algorithm.NUM_THREADS],
            "OUTPUT_FORMAT": parameters[algorithm.OUTPUT_FORMAT],
        }
        if p1.get("OUTPUT_MASK"):
            p4_params["INPUT_MASK"] = p1["OUTPUT_MASK"]
        
        p4 = processing.run(
            "sdb_tools:sdb_phase4_adaptive",
            p4_params,
            context=context,
            feedback=feedback,
            is_child_algorithm=True,
        )
        path_refined = p4["OUTPUT_FINAL"]

        if "BEST_R2" in p4:
            append_log(f"      → R2: {p4['BEST_R2']:.4f}", log_path, feedback)
            append_log("  ✓ Phase 04 completed\n", log_path, feedback)
    else:
        append_log("  [Phase 04] Adaptive Refinement", log_path, feedback)
        append_log("      → No Control Points found. SKIPPED.\n", log_path, feedback)

    feat_stack = p1["OUTPUT_FEATURES"]

    if p3.get("OUTPUT_DEPTH_MAP") and os.path.exists(p3["OUTPUT_DEPTH_MAP"]):
        p3_clamped = os.path.join(p3_dir, "Phase3_Depth_Cleaned.tif")
        clean_depth_map(
            p3["OUTPUT_DEPTH_MAP"], feat_stack, max_depth, p3_clamped, context, feedback
        )

        current_p3 = p3_clamped

        if apply_slope_filter:
            p3_slope = os.path.join(p3_dir, "Phase03_Depth_Final_SlopeFiltered.tif")
            current_p3 = slope_filter_depth(
                current_p3,
                slope_threshold=slope_threshold_val,
                out_path=p3_slope,
                context=context,
                feedback=feedback,
            )

        if remove_positives_flag:
            p3_no_pos = os.path.join(p3_dir, "Phase03_Depth_Final_NoPositives.tif")
            remove_positive_pixels(current_p3, p3_no_pos, feedback)
            current_p3 = p3_no_pos

        if p1.get("OUTPUT_OSW_POLY") and os.path.exists(p1["OUTPUT_OSW_POLY"]):
            append_log("  → Clipping Phase 03 Map with OSW Polygon...", log_path, feedback)
            p3_osw_clipped = os.path.join(p3_dir, "Phase03_Depth_OSW_Clipped.tif")
            processing.run(
                "gdal:cliprasterbymasklayer",
                {
                    "INPUT": current_p3,
                    "MASK": p1["OUTPUT_OSW_POLY"],
                    "SOURCE_CRS": crs_id,
                    "TARGET_CRS": crs_id,
                    "NODATA": -9999.0,
                    "ALPHA_BAND": False,
                    "CROP_TO_CUTLINE": False,
                    "KEEP_RESOLUTION": True,
                    "DATA_TYPE": 0,
                    "OUTPUT": p3_osw_clipped,
                },
                context=context,
                feedback=feedback,
                is_child_algorithm=True,
            )
            if os.path.exists(p3_osw_clipped):
                current_p3 = p3_osw_clipped

        p3["OUTPUT_DEPTH_MAP"] = current_p3
        write_qml_style(current_p3)

        # ---------------------------------------------------------
        # [NEW] Post-process isolated Linear Regression Depth & Uncertainty Maps (Clean & OSW Clip)
        # ---------------------------------------------------------
        lr_dir = os.path.join(p3_dir, "Linear_Regression")
        raw_lr_map = os.path.join(lr_dir, "Linear_Regression_Depth.tif")
        raw_lr_uncert = os.path.join(lr_dir, "Linear_Regression_Uncertainty.tif")

        if os.path.exists(raw_lr_map):
            try:
                lr_clamped = os.path.join(lr_dir, "Linear_Regression_Cleaned.tif")
                clean_depth_map(raw_lr_map, feat_stack, max_depth, lr_clamped, context, feedback)
                
                lr_current = lr_clamped

                if apply_slope_filter:
                    lr_slope = os.path.join(lr_dir, "Linear_Regression_SlopeFiltered.tif")
                    lr_current = slope_filter_depth(
                        lr_current,
                        slope_threshold=slope_threshold_val,
                        out_path=lr_slope,
                        context=context,
                        feedback=feedback,
                    )

                if remove_positives_flag:
                    lr_no_pos = os.path.join(lr_dir, "Linear_Regression_NoPositives.tif")
                    remove_positive_pixels(lr_current, lr_no_pos, feedback)
                    lr_current = lr_no_pos
                    
                if p1.get("OUTPUT_OSW_POLY") and os.path.exists(p1["OUTPUT_OSW_POLY"]):
                    lr_osw_clipped = os.path.join(lr_dir, "Linear_Regression_Depth_OSW_Clipped.tif")
                    processing.run(
                        "gdal:cliprasterbymasklayer",
                        {
                            "INPUT": lr_current,
                            "MASK": p1["OUTPUT_OSW_POLY"],
                            "SOURCE_CRS": crs_id,
                            "TARGET_CRS": crs_id,
                            "NODATA": -9999.0,
                            "ALPHA_BAND": False,
                            "CROP_TO_CUTLINE": False,
                            "KEEP_RESOLUTION": True,
                            "DATA_TYPE": 0,
                            "OUTPUT": lr_osw_clipped,
                        },
                        context=context,
                        feedback=feedback,
                        is_child_algorithm=True,
                    )
                    if os.path.exists(lr_osw_clipped):
                        import shutil
                        shutil.copy2(lr_osw_clipped, raw_lr_map)
                elif os.path.exists(lr_current):
                    import shutil
                    shutil.copy2(lr_current, raw_lr_map)
                append_log("      → Linear Regression depth map cleaned and OSW clipped.", log_path, feedback)
            except Exception as e:
                append_log(f"  ⚠ WARNING: Failed to post-process Linear Regression depth map: {e}", log_path, feedback)

        if os.path.exists(raw_lr_uncert):
            try:
                lr_u_clamped = os.path.join(lr_dir, "Linear_Regression_Uncertainty_Cleaned.tif")
                clean_depth_map(raw_lr_uncert, feat_stack, max_depth, lr_u_clamped, context, feedback)
                lr_u_current = lr_u_clamped

                if p1.get("OUTPUT_OSW_POLY") and os.path.exists(p1["OUTPUT_OSW_POLY"]):
                    lr_u_osw_clipped = os.path.join(lr_dir, "Linear_Regression_Uncertainty_OSW_Clipped.tif")
                    processing.run(
                        "gdal:cliprasterbymasklayer",
                        {
                            "INPUT": lr_u_current,
                            "MASK": p1["OUTPUT_OSW_POLY"],
                            "SOURCE_CRS": crs_id,
                            "TARGET_CRS": crs_id,
                            "NODATA": -9999.0,
                            "ALPHA_BAND": False,
                            "CROP_TO_CUTLINE": False,
                            "KEEP_RESOLUTION": True,
                            "DATA_TYPE": 0,
                            "OUTPUT": lr_u_osw_clipped,
                        },
                        context=context,
                        feedback=feedback,
                        is_child_algorithm=True,
                    )
                    if os.path.exists(lr_u_osw_clipped):
                        import shutil
                        shutil.copy2(lr_u_osw_clipped, raw_lr_uncert)
                elif os.path.exists(lr_u_current):
                    import shutil
                    shutil.copy2(lr_u_current, raw_lr_uncert)
                append_log("   [Analytics] Linear Regression uncertainty map successfully cleaned and OSW clipped.", log_path, feedback)
            except Exception as e:
                append_log(f"   [Warning] Failed to post-process Linear Regression uncertainty map: {e}", log_path, feedback)
        # ---------------------------------------------------------

        p3_uncert = p3.get("OUTPUT_UNCERT_MAP")
        p3_uncert_clamped = None
        if p3_uncert and os.path.exists(p3_uncert):
            p3_uncert_clamped = os.path.join(p3_dir, "Phase3_Uncertainty_Cleaned.tif")
            clean_depth_map(
                p3_uncert, feat_stack, max_depth, p3_uncert_clamped, context, feedback
            )

            if p1.get("OUTPUT_OSW_POLY") and os.path.exists(p1["OUTPUT_OSW_POLY"]):
                p3_uncert_osw = os.path.join(p3_dir, "Phase03_Uncertainty_OSW_Clipped.tif")
                processing.run(
                    "gdal:cliprasterbymasklayer",
                    {
                        "INPUT": p3_uncert_clamped,
                        "MASK": p1["OUTPUT_OSW_POLY"],
                        "SOURCE_CRS": crs_id,
                        "TARGET_CRS": crs_id,
                        "NODATA": -9999.0,
                        "ALPHA_BAND": False,
                        "CROP_TO_CUTLINE": False,
                        "KEEP_RESOLUTION": True,
                        "DATA_TYPE": 0,
                        "OUTPUT": p3_uncert_osw,
                    },
                    context=context,
                    feedback=feedback,
                    is_child_algorithm=True,
                )
                if os.path.exists(p3_uncert_osw):
                    p3_uncert_clamped = p3_uncert_osw

            p3["OUTPUT_UNCERT_MAP"] = p3_uncert_clamped

    p4_uncert_clamped = None
    if path_refined and os.path.exists(path_refined):
        p4_clamped = os.path.join(p4_dir, "Final_Depth_Cleaned.tif")
        clean_depth_map(
            path_refined, feat_stack, max_depth, p4_clamped, context, feedback
        )

        if apply_slope_filter:
            slope_filtered = os.path.join(p4_dir, "Final_Depth_SlopeFiltered.tif")
            path_refined = slope_filter_depth(
                p4_clamped,
                slope_threshold=slope_threshold_val,
                out_path=slope_filtered,
                context=context,
                feedback=feedback,
            )
        else:
            path_refined = p4_clamped

        if remove_positives_flag:
            p4_no_pos = os.path.join(p4_dir, "Phase04_Final_Depth_NoPositives.tif")
            remove_positive_pixels(path_refined, p4_no_pos, feedback)
            path_refined = p4_no_pos

        if p1.get("OUTPUT_OSW_POLY") and os.path.exists(p1["OUTPUT_OSW_POLY"]):
            append_log("  → Clipping Phase 04 Map with OSW Polygon...", log_path, feedback)
            p4_osw_clipped = os.path.join(p4_dir, "Phase04_Final_Depth_OSW_Clipped.tif")
            processing.run(
                "gdal:cliprasterbymasklayer",
                {
                    "INPUT": path_refined,
                    "MASK": p1["OUTPUT_OSW_POLY"],
                    "SOURCE_CRS": crs_id,
                    "TARGET_CRS": crs_id,
                    "NODATA": -9999.0,
                    "ALPHA_BAND": False,
                    "CROP_TO_CUTLINE": False,
                    "KEEP_RESOLUTION": True,
                    "DATA_TYPE": 0,
                    "OUTPUT": p4_osw_clipped,
                },
                context=context,
                feedback=feedback,
                is_child_algorithm=True,
            )
            if os.path.exists(p4_osw_clipped):
                path_refined = p4_osw_clipped

        if path_refined:
            write_qml_style(path_refined)

        p4_uncert = p4.get("OUTPUT_UNCERT")
        if p4_uncert and os.path.exists(p4_uncert):
            p4_uncert_clamped = os.path.join(p4_dir, "Final_Uncertainty_95.tif")
            clean_depth_map(
                p4_uncert, feat_stack, max_depth, p4_uncert_clamped, context, feedback
            )

            if p1.get("OUTPUT_OSW_POLY") and os.path.exists(p1["OUTPUT_OSW_POLY"]):
                p4_uncert_osw = os.path.join(p4_dir, "Final_Uncertainty_95_OSW_Clipped.tif")
                processing.run(
                    "gdal:cliprasterbymasklayer",
                    {
                        "INPUT": p4_uncert_clamped,
                        "MASK": p1["OUTPUT_OSW_POLY"],
                        "SOURCE_CRS": crs_id,
                        "TARGET_CRS": crs_id,
                        "NODATA": -9999.0,
                        "ALPHA_BAND": False,
                        "CROP_TO_CUTLINE": False,
                        "KEEP_RESOLUTION": True,
                        "DATA_TYPE": 0,
                        "OUTPUT": p4_uncert_osw,
                    },
                    context=context,
                    feedback=feedback,
                    is_child_algorithm=True,
                )
                if os.path.exists(p4_uncert_osw):
                    p4_uncert_clamped = p4_uncert_osw

            p4["OUTPUT_UNCERT"] = p4_uncert_clamped

    if enable_val and final_test:
        append_log("  [Phase 05] Validation", log_path, feedback)
        append_log("      → Generating Evaluation Metrics...", log_path, feedback)
        processing.run(
            "sdb_tools:sdb_05_reporting",
            {
                "INPUT_MAP_P3": p3["OUTPUT_DEPTH_MAP"],
                "INPUT_MAP_P4": path_refined if path_refined else p3["OUTPUT_DEPTH_MAP"],
                "INPUT_TRAIN": path_clean,
                "FIELD_TRAIN": field_depth,
                "INPUT_VALIDATION": final_test,
                "FIELD_VAL_DEPTH": algorithm.parameterAsString(
                    parameters, algorithm.FIELD_TEST_DEPTH, context
                ),
                "OUTPUT_FOLDER": out_dir,
            },
            context=context,
            feedback=feedback,
            is_child_algorithm=True,
        )

    if p3.get("OUTPUT_DEPTH_MAP") and os.path.exists(p3["OUTPUT_DEPTH_MAP"]):
        details_init = QgsProcessingContext.LayerDetails(
            "Initial SDB Map [Phase 03]", QgsProject.instance(), "Initial SDB"
        )
        qml_path_p3 = os.path.splitext(p3["OUTPUT_DEPTH_MAP"])[0] + ".qml"
        if os.path.exists(qml_path_p3):
            details_init.setPostProcessor(StylePostProcessor(qml_path_p3))
        context.addLayerToLoadOnCompletion(p3["OUTPUT_DEPTH_MAP"], details_init)



    if path_refined and os.path.exists(path_refined):
        details_ref = QgsProcessingContext.LayerDetails(
            "Refined SDB Map [Phase 04]", QgsProject.instance(), "Refined SDB"
        )
        qml_path_p4 = os.path.splitext(path_refined)[0] + ".qml"
        if os.path.exists(qml_path_p4):
            details_ref.setPostProcessor(StylePostProcessor(qml_path_p4))
        context.addLayerToLoadOnCompletion(path_refined, details_ref)



    spatial_cv_p3 = algorithm.parameterAsBool(parameters, algorithm.SPATIAL_CV_P3, context)
    spatial_cv_p4 = algorithm.parameterAsBool(parameters, algorithm.SPATIAL_CV_P4, context)
    enable_ransac = algorithm.parameterAsBool(parameters, algorithm.ENABLE_RANSAC, context)
    filter_mode = algorithm.parameterAsInt(parameters, algorithm.FILTER_MODE, context)
    field_depth = algorithm.parameterAsString(parameters, algorithm.FIELD_DEPTH, context)
    field_weight = algorithm.parameterAsString(parameters, algorithm.FIELD_WEIGHT, context)
    collision_handling_idx = algorithm.parameterAsInt(parameters, algorithm.COLLISION_HANDLING, context)

    input_train_layer = algorithm.parameterAsVectorLayer(parameters, algorithm.INPUT_TRAIN, context)
    input_test_layer = algorithm.parameterAsVectorLayer(parameters, algorithm.INPUT_TEST, context) if enable_val else None

    # Generate static 3D seabed PNG
    final_depth_for_3d = None
    if path_refined and os.path.exists(path_refined):
        final_depth_for_3d = path_refined
    elif p3.get("OUTPUT_DEPTH_MAP") and os.path.exists(p3["OUTPUT_DEPTH_MAP"]):
        final_depth_for_3d = p3["OUTPUT_DEPTH_MAP"]
        
    if final_depth_for_3d:
        out_3d_png = os.path.join(out_dir, "5_Plot_3D_Seabed.png")
        generate_3d_seabed_png(final_depth_for_3d, out_3d_png, feedback)

    generate_html_dashboard(
        out_dir=out_dir,
        p3_dir=p3_dir,
        p4_dir=p4_dir,
        spatial_cv_p3=spatial_cv_p3,
        spatial_cv_p4=spatial_cv_p4,
        enable_ransac=enable_ransac,
        filter_mode=filter_mode,
        field_depth=field_depth,
        field_weight=field_weight,
        collision_handling_idx=collision_handling_idx,
        log_path=log_path,
        feedback=feedback,
        raster_name=input_raster.name() if input_raster else "Unknown Raster",
        train_name=input_train_layer.name() if input_train_layer else "Unknown Vector",
        test_name=input_test_layer.name() if input_test_layer else "N/A"
    )
    import time
    end_time = time.time()
    elapsed = end_time - start_time
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    
    append_log("════════════════════════════════════════════════════════════", log_path, feedback)
    append_log(f"✓ SDB Single Masterflow Completed in {mins}m {secs}s".center(60), log_path, feedback)
    append_log("════════════════════════════════════════════════════════════\n", log_path, feedback)

    return {}


def generate_pdf_report(out_dir, p3_models, p4_models, has_p4, enable_ransac, pt_count, depth_min, depth_max, has_weight_stats, weight_min, weight_max, actual_pt_count, collision_handling, filter_mode_name, strat_rows, log_path=None, feedback=None, raster_name="Satellite Imagery", train_name="ICESat-2 (ATL24) LiDAR", test_name="In-situ Echosounder Surveys", is_spatiospectral=False, p2_dir=None):
    import datetime
    
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    folder_name = os.path.basename(out_dir)

    # Compile tables
    # P3 Table
    p3_rows_html = ""
    for idx, m in enumerate(p3_models):
        algo = m["Algorithm"]
        r2 = m["R2"]
        rmse = m["RMSE"]
        wmape = m["wMAPE"]
        bg = ' bgcolor="#f8fafc"' if idx % 2 == 1 else ''
        p3_rows_html += f"""
        <tr{bg}>
            <td style="white-space: nowrap;"><strong>{algo}</strong></td>
            <td style="color: #0f766e; font-weight: bold; white-space: nowrap;">{r2:.4f}</td>
            <td style="color: #1d4ed8; font-weight: bold; white-space: nowrap;">{rmse:.2f}m</td>
            <td style="color: #4f46e5; font-weight: bold; white-space: nowrap;">{wmape:.2f}%</td>
        </tr>
        """

    # P4 Table
    p4_rows_html = ""
    if has_p4:
        for idx, m in enumerate(p4_models):
            algo = m["Algorithm"]
            r2 = m["R2"]
            rmse = m["RMSE"]
            wmape = m["wMAPE"]
            bg = ' bgcolor="#f8fafc"' if idx % 2 == 1 else ''
            p4_rows_html += f"""
            <tr{bg}>
                <td style="white-space: nowrap;"><strong>{algo}</strong></td>
                <td style="color: #0f766e; font-weight: bold; white-space: nowrap;">{r2:.4f}</td>
                <td style="color: #1d4ed8; font-weight: bold; white-space: nowrap;">{rmse:.2f}m</td>
                <td style="color: #4f46e5; font-weight: bold; white-space: nowrap;">{wmape:.2f}%</td>
            </tr>
            """
    else:
        p4_rows_html = "<tr><td colspan='4' style='text-align: center; color: #64748b; padding: 6px;'>Phase 04 Refinement was bypassed/disabled.</td></tr>"

    # Stratified Table
    strat_rows_html = ""
    if strat_rows:
        for idx, r in enumerate(strat_rows):
            model = r["Model"]
            depth_bin = r["Depth_Bin"]
            count = r["Count"]
            mean_depth = r["Mean_Depth"]
            rmse = r["RMSE"]
            model_tvu = r["Model_TVU_95"]
            iho_limit = r["IHO_TVU_Limit"]
            iho_order = r["IHO_Order"]
            uses = r["Suggested_Uses"]
            bg = ' bgcolor="#f8fafc"' if idx % 2 == 1 else ''

            if iho_order == "Special Order":
                badge_style = "background-color: #d1fae5; color: #065f46; border: 1px solid #a7f3d0;"
            elif iho_order == "Order 1a/1b":
                badge_style = "background-color: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd;"
            elif iho_order == "Order 2":
                badge_style = "background-color: #fef3c7; color: #92400e; border: 1px solid #fde68a;"
            else:
                badge_style = "background-color: #fee2e2; color: #991b1b; border: 1px solid #fca5a5;"

            strat_rows_html += f"""
            <tr{bg}>
                <td style="white-space: nowrap;">{model}</td>
                <td style="white-space: nowrap;"><strong>{depth_bin}</strong></td>
                <td style="white-space: nowrap;">{count}</td>
                <td style="white-space: nowrap;">{mean_depth:.2f}m</td>
                <td style="white-space: nowrap;">{rmse:.3f}m</td>
                <td style="color: #0284c7; font-weight: bold; white-space: nowrap;">{model_tvu:.3f}m</td>
                <td style="white-space: nowrap;">{iho_limit:.3f}m</td>
                <td style="white-space: nowrap;"><span style="padding: 2px 4px; font-weight: bold; border-radius: 3px; font-size: 8pt; {badge_style}">{iho_order}</span></td>
                <td style="font-size: 8pt; color: #475569;">{uses}</td>
            </tr>
            """
    else:
        strat_rows_html = "<tr><td colspan='9' style='text-align: center; color: #64748b; padding: 6px;'>Phase 05 Validation was not run. No independent validation data available.</td></tr>"

    # Phase 02 Plots
    p2_plots_html = ""
    if not p2_dir:
        p2_dir = os.path.join(out_dir, "Phase_02_Filtering")
        
    p2_plot1_path = os.path.abspath(os.path.join(p2_dir, "2_Plot_1_Trend.png")).replace("\\", "/")
    p2_plot2_path = os.path.abspath(os.path.join(p2_dir, "2_Plot_2_Variance.png")).replace("\\", "/")
    p2_plot3_path = os.path.abspath(os.path.join(p2_dir, "2_Plot_3_Envelope.png")).replace("\\", "/")

    # Check which plots exist and match the filter mode
    p2_plots_to_show = []
    if os.path.exists(p2_plot1_path):
        p2_plots_to_show.append((p2_plot1_path, "Figure 1: Regression Trend & Outlier Rejection"))
        
    if filter_mode_name and "LS Variance" in filter_mode_name:
        if os.path.exists(p2_plot2_path):
            p2_plots_to_show.append((p2_plot2_path, "Figure 2: Depth vs Variance Analysis"))
    else:
        if os.path.exists(p2_plot3_path):
            p2_plots_to_show.append((p2_plot3_path, "Figure 2: Residuals & Uncertainty Envelope"))

    if p2_plots_to_show:
        p2_plots_html = """
        <h2>📈 Phase 02: Training Dataset Filtering & Uncertainty Plots</h2>
        <p style="color: #64748b; font-size: 9pt; margin-bottom: 12pt;">
            The following figures display the pre-filtering regression, trend fitting, and outlier rejection results.
        </p>
        """
        if len(p2_plots_to_show) == 1:
            img_path, caption = p2_plots_to_show[0]
            p2_plots_html += f"""
            <div style="display: block; width: 100%; margin: 0 auto 12pt auto; text-align: center;">
                <img src="{img_path}" width="480" style="border: 1px solid #cbd5e1; border-radius: 4px; display: block; margin: 0 auto;" />
                <div style="font-size: 8pt; color: #64748b; margin-top: 3pt; font-weight: bold; text-align: center;">{caption}</div>
            </div>
            """
        else:
            p2_plots_html += """
            <table align="center" border="0" cellspacing="0" cellpadding="0" style="width: 600px; border: none; margin-top: 10pt; margin-bottom: 10pt;">
                <tr bgcolor="transparent">
            """
            for img_path, caption in p2_plots_to_show:
                p2_plots_html += f"""
                    <td style="width: 300px; text-align: center; border: none; padding: 4pt; background-color: transparent;">
                        <img src="{img_path}" width="280" style="border: 1px solid #cbd5e1; border-radius: 4px;" />
                        <div style="font-size: 8pt; color: #64748b; margin-top: 3pt; font-weight: bold; text-align: center;">{caption}</div>
                    </td>
                """
            p2_plots_html += """
                </tr>
            </table>
            """

    # Image paths
    scatter_path = os.path.abspath(os.path.join(out_dir, "5_Plot_Scatter_Comparison.png")).replace("\\", "/")
    residuals_path = os.path.abspath(os.path.join(out_dir, "5_Plot_Residuals.png")).replace("\\", "/")
    histogram_path = os.path.abspath(os.path.join(out_dir, "5_Plot_Error_Histogram.png")).replace("\\", "/")
    seabed_3d_path = os.path.abspath(os.path.join(out_dir, "5_Plot_3D_Seabed.png")).replace("\\", "/")

    plots_section_html = ""
    if os.path.exists(scatter_path) or os.path.exists(residuals_path) or os.path.exists(histogram_path) or os.path.exists(seabed_3d_path):
        plots_section_html = f"""
        <div style="page-break-before: always;"></div>
        <h2>📊 Scientific Validation & 3D Topography Plots</h2>
        <p style="color: #64748b; font-size: 9pt; margin-bottom: 12pt;">
            The following plots display the model's accuracy comparison and the generated 3D seabed topography.
        </p>
        """
        
        if os.path.exists(seabed_3d_path) and (os.path.exists(scatter_path) or os.path.exists(residuals_path) or os.path.exists(histogram_path)):
            val_col_html = ""
            if os.path.exists(scatter_path):
                val_col_html += f"""
                <div style="margin-bottom: 8pt; text-align: center;">
                    <img src="{scatter_path}" width="380" style="border: 1px solid #cbd5e1; border-radius: 4px;" />
                    <div style="font-size: 8pt; color: #64748b; margin-top: 2pt; font-weight: bold;">Figure 2: 1:1 Scatter Comparison Plot</div>
                </div>
                """
            
            if os.path.exists(residuals_path) or os.path.exists(histogram_path):
                val_col_html += f"""
                <table align="center" border="0" cellspacing="0" cellpadding="0" style="width: 380px; border: none; margin: 0 auto;">
                    <tr bgcolor="transparent">
                """
                if os.path.exists(residuals_path):
                    val_col_html += f"""
                        <td style="width: 50%; text-align: center; border: none; padding: 2pt; background-color: transparent;">
                            <img src="{residuals_path}" width="180" style="border: 1px solid #cbd5e1; border-radius: 4px;" />
                            <div style="font-size: 7.5pt; color: #64748b; margin-top: 2pt; font-weight: bold;">Figure 3: Residual Error</div>
                        </td>
                    """
                else:
                    val_col_html += """<td style="width: 50%; border: none;"></td>"""
                    
                if os.path.exists(histogram_path):
                    val_col_html += f"""
                        <td style="width: 50%; text-align: center; border: none; padding: 2pt; background-color: transparent;">
                            <img src="{histogram_path}" width="180" style="border: 1px solid #cbd5e1; border-radius: 4px;" />
                            <div style="font-size: 7.5pt; color: #64748b; margin-top: 2pt; font-weight: bold;">Figure 4: Error Histogram</div>
                        </td>
                    """
                else:
                    val_col_html += """<td style="width: 50%; border: none;"></td>"""
                    
                val_col_html += """
                    </tr>
                </table>
                """
                
            plots_section_html += f"""
            <table border="0" cellspacing="0" cellpadding="0" style="width: 100%; border: none; margin-top: 10pt; margin-bottom: 10pt;">
                <tr bgcolor="transparent">
                    <td style="width: 50%; vertical-align: middle; text-align: center; border: none; padding: 4pt; background-color: transparent;">
                        <img src="{seabed_3d_path}" width="460" style="border: 1px solid #cbd5e1; border-radius: 4px;" />
                        <div style="font-size: 8pt; color: #64748b; margin-top: 3pt; font-weight: bold; text-align: center;">Figure 1: 3D Seabed Elevation Model Plot (Spectral Colormap)</div>
                    </td>
                    <td style="width: 50%; vertical-align: top; text-align: center; border: none; padding: 4pt; background-color: transparent;">
                        {val_col_html}
                    </td>
                </tr>
            </table>
            """
        else:
            if os.path.exists(seabed_3d_path):
                plots_section_html += f"""
                <div style="display: block; width: 100%; margin: 0 auto 12pt auto; text-align: center;">
                    <img src="{seabed_3d_path}" width="480" style="border: 1px solid #cbd5e1; border-radius: 4px; display: block; margin: 0 auto;" />
                    <div style="font-size: 8pt; color: #64748b; margin-top: 3pt; font-weight: bold; text-align: center;">Figure 1: 3D Seabed Elevation Model Plot (Spectral Colormap)</div>
                </div>
                """
            if os.path.exists(scatter_path):
                plots_section_html += f"""
                <div style="display: block; width: 100%; margin: 0 auto 12pt auto; text-align: center;">
                    <img src="{scatter_path}" width="480" style="border: 1px solid #cbd5e1; border-radius: 4px; display: block; margin: 0 auto;" />
                    <div style="font-size: 8pt; color: #64748b; margin-top: 3pt; font-weight: bold; text-align: center;">Figure 2: 1:1 Scatter Comparison Plot (Predicted vs. Observed Depths)</div>
                </div>
                """
            if os.path.exists(residuals_path) or os.path.exists(histogram_path):
                plots_section_html += f"""
                <table align="center" border="0" cellspacing="0" cellpadding="0" style="width: 500px; border: none; margin-top: 10pt; margin-bottom: 10pt;">
                    <tr bgcolor="transparent">
                """
                if os.path.exists(residuals_path):
                    plots_section_html += f"""
                        <td style="width: 250px; text-align: center; border: none; padding: 4pt; background-color: transparent;">
                            <img src="{residuals_path}" width="235" style="border: 1px solid #cbd5e1; border-radius: 4px;" />
                            <div style="font-size: 8pt; color: #64748b; margin-top: 3pt; font-weight: bold; text-align: center;">Figure 3: Residual Error Plot</div>
                        </td>
                    """
                else:
                    plots_section_html += """<td style="width: 250px; border: none;"></td>"""
                if os.path.exists(histogram_path):
                    plots_section_html += f"""
                        <td style="width: 250px; text-align: center; border: none; padding: 4pt; background-color: transparent;">
                            <img src="{histogram_path}" width="235" style="border: 1px solid #cbd5e1; border-radius: 4px;" />
                            <div style="font-size: 8pt; color: #64748b; margin-top: 3pt; font-weight: bold; text-align: center;">Figure 4: Density Histogram of Residual Errors</div>
                        </td>
                    """
                else:
                    plots_section_html += """<td style="width: 250px; border: none;"></td>"""
                plots_section_html += """
                    </tr>
                </table>
                """


    html_content = f"""<!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: Arial, Helvetica, sans-serif;
            color: #333333;
            margin: 10pt;
            line-height: 1.4;
            font-size: 10pt;
        }}
        .header {{
            background-color: #1a365d;
            color: white;
            padding: 12pt;
            border-radius: 4pt;
            margin-bottom: 15pt;
        }}
        .header h1 {{
            margin: 0;
            font-size: 16pt;
            font-weight: bold;
        }}
        .header p {{
            margin: 4pt 0 0 0;
            font-size: 9.5pt;
            color: #cbd5e1;
        }}
        h2 {{
            font-size: 12pt;
            color: #1a365d;
            margin-top: 15pt;
            border-bottom: 1.5pt solid #e2e8f0;
            padding-bottom: 3pt;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 8pt;
            margin-bottom: 12pt;
        }}
        th {{
            background-color: #f1f5f9;
            color: #1e293b;
            font-weight: bold;
            text-align: left;
            border: 1px solid #cbd5e1;
            padding: 4pt 6pt;
            font-size: 8.5pt;
            white-space: nowrap;
        }}
        td {{
            border: 1px solid #cbd5e1;
            padding: 4pt 6pt;
            font-size: 8pt;
            white-space: nowrap;
        }}
        tr:nth-child(even) {{
            background-color: #f8fafc;
        }}
        .meta-box {{
            background-color: #f8fafc;
            border: 1px solid #cbd5e1;
            border-radius: 4pt;
            padding: 8pt;
            margin-bottom: 12pt;
        }}
        .meta-table {{
            width: 100%;
            margin: 0;
        }}
        .meta-table td {{
            border: none;
            padding: 2pt 4pt;
            font-size: 8.5pt;
            white-space: nowrap;
        }}
        .meta-table tr:nth-child(even) {{
            background-color: transparent;
        }}
        .footer {{
            margin-top: 15pt;
            border-top: 1.5pt solid #cbd5e1;
            padding-top: 5pt;
            text-align: center;
            color: #64748b;
            font-size: 7.5pt;
        }}
    </style>
    </head>
    <body>
        <div class="header">
            <h1>BATHYMETRIX-AI V6.3: TECHNICAL VALIDATION REPORT</h1>
            <p>SDB MasterFlow | High-Precision Satellite-Derived Bathymetry Calibration & Validation</p>
        </div>

        <div class="meta-box">
            <table class="meta-table">
                <tr>
                    <td style="width: 20%; font-weight: bold; color: #475569; white-space: nowrap;">Project Folder:</td>
                    <td style="width: 30%; font-weight: bold; color: #1e293b; white-space: nowrap;">{folder_name}</td>
                    <td style="width: 20%; font-weight: bold; color: #475569; white-space: nowrap;">Date:</td>
                    <td style="width: 30%; color: #1e293b; white-space: nowrap;">{date_str}</td>
                </tr>
                <tr>
                    <td style="font-weight: bold; color: #475569; white-space: nowrap;">Satellite Data:</td>
                    <td style="color: #1e293b; white-space: nowrap;">{raster_name}</td>
                    <td style="font-weight: bold; color: #475569; white-space: nowrap;">Training Reference:</td>
                    <td style="color: #1e293b; white-space: nowrap;">{train_name}</td>
                </tr>
                <tr>
                    <td style="font-weight: bold; color: #475569; white-space: nowrap;">Validation Ground Truth:</td>
                    <td style="color: #1e293b; white-space: nowrap;" colspan="3">{test_name}</td>
                </tr>
            </table>
        </div>

        <h2>🧹 Phase 02: Training Dataset Filtering & Uncertainty Summary</h2>
        <table border="1" cellspacing="0" cellpadding="6" bordercolor="#cbd5e1" style="width: 100%; border-collapse: collapse; margin-top: 8pt; margin-bottom: 12pt;">
            <thead>
                <tr bgcolor="#f1f5f9">
                    <th style="white-space: nowrap;">Filtering Algorithm</th>
                    <th style="white-space: nowrap;">Collision Resolution</th>
                    <th style="white-space: nowrap;">Cleaned Training Samples</th>
                    <th style="white-space: nowrap;">Final Model Input Points</th>
                    <th style="white-space: nowrap;">Calibration Depth Range</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="white-space: nowrap;"><strong>{filter_mode_name}</strong></td>
                    <td style="white-space: nowrap;">{collision_handling}</td>
                    <td style="white-space: nowrap;">{pt_count}</td>
                    <td style="white-space: nowrap;"><strong>{actual_pt_count}</strong></td>
                    <td style="white-space: nowrap;">{depth_min}m to {depth_max}m</td>
                </tr>
            </tbody>
        </table>

        {p2_plots_html}

        {
        f'''<h2>🏆 Phase 03: AutoML Leaderboard (Global Model)</h2>
        <p style="color: #64748b; font-size: 8pt; margin-bottom: 5px;">
            The algorithms are optimized and evaluated against independent cross-validation blocks.
        </p>
        <table border="1" cellspacing="0" cellpadding="6" bordercolor="#cbd5e1" style="width: 100%; border-collapse: collapse; margin-top: 8pt; margin-bottom: 12pt;">
            <thead>
                <tr bgcolor="#f1f5f9">
                    <th style="white-space: nowrap;">Algorithm</th>
                    <th style="white-space: nowrap;">R² Accuracy</th>
                    <th style="white-space: nowrap;">RMSE (Vertical Error)</th>
                    <th style="white-space: nowrap;">wMAPE (%)</th>
                </tr>
            </thead>
            <tbody>
                {p3_rows_html}
            </tbody>
        </table>''' if not is_spatiospectral else ''
        }

        {f"<h2>🔄 Phase 04: Depth-Dependent Residual Calibration Leaderboard</h2><table border='1' cellspacing='0' cellpadding='6' bordercolor='#cbd5e1' style='width: 100%; border-collapse: collapse; margin-top: 8pt; margin-bottom: 12pt;'><thead><tr bgcolor='#f1f5f9'><th style='white-space: nowrap;'>Algorithm</th><th style='white-space: nowrap;'>R² Accuracy</th><th style='white-space: nowrap;'>RMSE (Vertical Error)</th><th style='white-space: nowrap;'>wMAPE (%)</th></tr></thead><tbody>{p4_rows_html}</tbody></table>" if has_p4 and not is_spatiospectral else ""}

        <div class="footer">
            Report generated automatically by Bathymetrix-AI V6.4. All rights reserved. &copy; Mohamed Aly Nasef (2026).
        </div>
        <div style="page-break-before: always;"></div>

        <h2>📏 IHO S-44 Standards Compliance & Stratified Error Analysis</h2>
        <p style="color: #64748b; font-size: 8pt; margin-bottom: 5px;">
            Compliance evaluation at a 95% confidence level (Model TVU = 1.96 * RMSE) calculated across depth layers.
        </p>
        <table border="1" cellspacing="0" cellpadding="5" bordercolor="#cbd5e1" style="width: 100%; border-collapse: collapse; margin-top: 8pt; margin-bottom: 12pt;">
            <thead>
                <tr bgcolor="#f1f5f9">
                    <th style="white-space: nowrap;">Model Phase</th>
                    <th style="white-space: nowrap;">Depth Bin</th>
                    <th style="white-space: nowrap;">Points</th>
                    <th style="white-space: nowrap;">Mean Depth</th>
                    <th style="white-space: nowrap;">RMSE</th>
                    <th style="white-space: nowrap;">Model TVU (95%)</th>
                    <th style="white-space: nowrap;">IHO TVU Limit</th>
                    <th style="white-space: nowrap;">IHO Order Achieved</th>
                    <th>Suggested Industrial Uses</th>
                </tr>
            </thead>
            <tbody>
                {strat_rows_html}
            </tbody>
        </table>

        {f'<div class="footer">Report generated automatically by Bathymetrix-AI V6.4. All rights reserved. &copy; Mohamed Aly Nasef (2026).</div>' if plots_section_html else ""}
        {plots_section_html}

        <div class="footer">
            Report generated automatically by Bathymetrix-AI V6.4. All rights reserved. &copy; Mohamed Aly Nasef (2026).
        </div>
    </body>
    </html>
    """

    try:
        html_report_path = os.path.join(out_dir, "SDB Technical Report.html")
        with open(html_report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        success_msg = f"HTML Technical Report generated successfully: {html_report_path}"
        if feedback is not None:
            feedback.pushInfo(success_msg)
        elif log_path:
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(success_msg + "\n")
            except Exception:
                pass
        return html_report_path
    except Exception as e:
        error_msg = f"Failed to generate SDB Technical Report HTML: {str(e)}"
        if feedback is not None:
            feedback.pushWarning(error_msg)
        elif log_path:
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"[Warning] {error_msg}\n")
            except Exception:
                pass
        return None
