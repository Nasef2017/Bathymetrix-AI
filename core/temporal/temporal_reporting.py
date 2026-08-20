import os
import json
import numpy as np
from typing import Dict, List, Any
from qgis.core import (
    QgsProject,
    QgsProcessingContext,
    QgsProcessingLayerPostProcessorInterface,
    QgsLayerTreeGroup,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsProcessingFeedback,
    QgsCategorizedSymbolRenderer,
    QgsRendererCategory,
    QgsSymbol,
    QgsPalettedRasterRenderer,
    QgsSingleBandPseudoColorRenderer,
    QgsRasterShader,
    QgsColorRampShader,
    QgsRasterLayerTemporalProperties,
    QgsDateTimeRange,
    QgsRasterBandStats
)
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtCore import QDateTime, QDate, QTime


class StylePostProcessor(QgsProcessingLayerPostProcessorInterface):
    """Thread-safe style applicator executed on the main GUI thread after algorithm completion."""
    def __init__(self, qml_path: str):
        super().__init__()
        self.qml_path = qml_path

    def postProcessLayer(self, layer, context, feedback):
        if layer and layer.isValid() and os.path.exists(self.qml_path):
            layer.loadNamedStyle(self.qml_path)
            layer.triggerRepaint()


class TemporalReportGenerator:
    """
    Populates an organized QGIS Layer Group with all temporal products
    and exports summary multi-year analytics CSV/JSON reports.
    """

    def __init__(self):
        self._post_processors = []

    def generate_layer_group_and_reports(
        self,
        yearly_sdb_results: Dict[int, Dict[str, str]],
        change_polygons: List[str],
        analytics_results: List[Dict[str, Any]],
        benthic_results: Dict[int, Dict[str, Any]],
        output_dir: str,
        feedback: QgsProcessingFeedback,
        uncertainty_mode: str = "classical",
        qr_confidence: float = 0.95,
        context: QgsProcessingContext = None
    ):
        """
        Populates QGIS Layer Group and exports summary analytics report.
        """
        feedback.pushInfo("📁 [LAYER GROUP & REPORTING] Organizing QGIS layers and generating summary report...")

        import shutil
        years = sorted(list(yearly_sdb_results.keys()))
        first_yr = years[0] if years else None
        last_yr = years[-1] if years else None

        for yr, res in sorted(yearly_sdb_results.items()):
            if res.get("sdb_depth_map") and os.path.exists(res["sdb_depth_map"]):
                old_path = res["sdb_depth_map"]
                new_name = f"SDB_{yr}.tif"
                new_path = os.path.join(os.path.dirname(old_path), new_name)
                if old_path != new_path:
                    old_qml = os.path.splitext(old_path)[0] + ".qml"
                    new_qml = os.path.splitext(new_path)[0] + ".qml"
                    try:
                        shutil.copy2(old_path, new_path)
                        if os.path.exists(old_qml):
                            shutil.copy2(old_qml, new_qml)
                        res["sdb_depth_map"] = new_path
                    except Exception as e:
                        feedback.pushInfo(f"⚠️ Could not copy {old_path} to {new_name}: {e}")

        project = context.project() if (context and context.project()) else QgsProject.instance()

        # Add Benthic Layers
        for yr, res in sorted(benthic_results.items()):
            if res and res.get("benthic_map_path") and os.path.exists(res["benthic_map_path"]):
                layer_name = f"Benthic Habitat {yr}"
                rlayer = QgsRasterLayer(res["benthic_map_path"], layer_name)
                if rlayer.isValid():
                    classes = [
                        QgsPalettedRasterRenderer.Class(1, QColor(34, 139, 34), "Vegetation")
                    ]
                    renderer = QgsPalettedRasterRenderer(rlayer.dataProvider(), 1, classes)
                    rlayer.setRenderer(renderer)
                    style_path = os.path.splitext(res["benthic_map_path"])[0] + ".qml"
                    rlayer.saveNamedStyle(style_path)
                    if context:
                        details = QgsProcessingContext.LayerDetails(layer_name, project, "")
                        pp = StylePostProcessor(style_path)
                        self._post_processors.append(pp)
                        details.setPostProcessor(pp)
                        context.addLayerToLoadOnCompletion(res["benthic_map_path"], details)

        # Add Shoreline Change Polygons for ALL periods (Sequential and Overall)
        if change_polygons:
            for poly_shp in change_polygons:
                if os.path.exists(poly_shp):
                    yr_str = os.path.basename(poly_shp).replace("Shoreline_Change_Polygons_", "").replace(".shp", "")
                    yr_display = yr_str.replace("_", "-")
                    layer_name = f"Shoreline Change Polygons ({yr_display})"
                    vlayer = QgsVectorLayer(poly_shp, layer_name, "ogr")
                    if vlayer.isValid():
                        cat_list = []
                        sym_erosion = QgsSymbol.defaultSymbol(vlayer.geometryType())
                        sym_erosion.setColor(QColor(255, 0, 0, 150))
                        cat_list.append(QgsRendererCategory('Erosion', sym_erosion, 'Erosion'))

                        sym_accretion = QgsSymbol.defaultSymbol(vlayer.geometryType())
                        sym_accretion.setColor(QColor(0, 255, 0, 150))
                        cat_list.append(QgsRendererCategory('Accretion', sym_accretion, 'Accretion'))

                        renderer = QgsCategorizedSymbolRenderer('ChangeType', cat_list)
                        vlayer.setRenderer(renderer)
                        vlayer.setOpacity(0.7)

                        style_path = os.path.splitext(poly_shp)[0] + ".qml"
                        vlayer.saveNamedStyle(style_path)

                        if context:
                            details = QgsProcessingContext.LayerDetails(layer_name, project, "")
                            pp = StylePostProcessor(style_path)
                            self._post_processors.append(pp)
                            details.setPostProcessor(pp)
                            context.addLayerToLoadOnCompletion(poly_shp, details)

        # Add Volumetric Erosion Accretion and MSI (For all periods: Overall and Sequential)
        for res in analytics_results:
            period = res.get("period", "")
            
            # Volumetric Erosion Accretion
            if res.get("statcd_raster_path") and os.path.exists(res["statcd_raster_path"]):
                layer_name = f"Volumetric Erosion Accretion Trend ({period})"
                rlayer = QgsRasterLayer(res["statcd_raster_path"], layer_name)
                if rlayer.isValid():
                    provider = rlayer.dataProvider()
                    # Use sample size of 250000 to prevent QGIS UI from freezing on massive rasters
                    stats = provider.bandStatistics(1, QgsRasterBandStats.All, rlayer.extent(), 250000)
                    min_val = stats.minimumValue
                    max_val = stats.maximumValue
                    mean_val = stats.mean
                    std_val = stats.stdDev
                    
                    # Robust mapping: use mean +/- 2.5 std to filter out extreme outliers that wash out the legend
                    robust_min = mean_val - 2.5 * std_val
                    robust_max = mean_val + 2.5 * std_val
                    
                    # Create symmetric bounds around 0 for diverging color ramp
                    abs_max = max(abs(robust_min) if not np.isnan(robust_min) else 0.1, abs(robust_max) if not np.isnan(robust_max) else 0.1)
                    if abs_max < 0.1:
                        abs_max = 0.5
                        
                    fnc = QgsColorRampShader()
                    fnc.setColorRampType(QgsColorRampShader.Interpolated)
                    # Diverging Color Ramp for Continuous Change dynamically scaled
                    lst = [
                        QgsColorRampShader.ColorRampItem(-abs_max, QColor(215, 25, 28), f"High Erosion ({-abs_max:.2f}m)"),
                        QgsColorRampShader.ColorRampItem(-abs_max/2, QColor(253, 174, 97), f"Erosion ({-abs_max/2:.2f}m)"),
                        QgsColorRampShader.ColorRampItem(0.0, QColor(255, 255, 255, 0), "Stable / Noise (0.00m)"), 
                        QgsColorRampShader.ColorRampItem(abs_max/2, QColor(166, 217, 106), f"Accretion (+{abs_max/2:.2f}m)"),
                        QgsColorRampShader.ColorRampItem(abs_max, QColor(26, 150, 65), f"High Accretion (+{abs_max:.2f}m)")
                    ]
                    fnc.setColorRampItemList(lst)
                    shader = QgsRasterShader()
                    shader.setRasterShaderFunction(fnc)
                    renderer = QgsSingleBandPseudoColorRenderer(rlayer.dataProvider(), 1, shader)
                    renderer.setClassificationMin(-abs_max)
                    renderer.setClassificationMax(abs_max)
                    rlayer.setRenderer(renderer)
                    
                    # Save style to qml
                    style_path = os.path.splitext(res["statcd_raster_path"])[0] + ".qml"
                    rlayer.saveNamedStyle(style_path)
                    
                    if context:
                        details = QgsProcessingContext.LayerDetails(layer_name, project, "")
                        pp = StylePostProcessor(style_path)
                        self._post_processors.append(pp)
                        details.setPostProcessor(pp)
                        context.addLayerToLoadOnCompletion(res["statcd_raster_path"], details)

            # MSI (Singleband pseudocolor, inverted Red)
            if res.get("msi_raster_path") and os.path.exists(res["msi_raster_path"]):
                layer_name = f"Morphological Stability Trend ({period})"
                rlayer = QgsRasterLayer(res["msi_raster_path"], layer_name)
                if rlayer.isValid():
                    fnc = QgsColorRampShader()
                    fnc.setColorRampType(QgsColorRampShader.Interpolated)
                    # Spectral Red-Yellow-Blue Ramp for MSI Stability:
                    # 0.00 -> Red (Highly Unstable)
                    # 0.25 -> Orange (Unstable)
                    # 0.50 -> Yellow (Moderate)
                    # 0.75 -> Light Blue (Semi-Stable)
                    # 1.00 -> Solid Blue (Highly Stable)
                    lst = [
                        QgsColorRampShader.ColorRampItem(0.0, QColor(215, 25, 28), "Highly Unstable"),
                        QgsColorRampShader.ColorRampItem(0.25, QColor(253, 174, 97), "Unstable"),
                        QgsColorRampShader.ColorRampItem(0.5, QColor(255, 255, 191), "Moderate"),
                        QgsColorRampShader.ColorRampItem(0.75, QColor(220, 220, 220), "Semi-Stable"),
                        QgsColorRampShader.ColorRampItem(1.0, QColor(44, 123, 182), "Highly Stable")
                    ]
                    fnc.setColorRampItemList(lst)
                    shader = QgsRasterShader()
                    shader.setRasterShaderFunction(fnc)
                    renderer = QgsSingleBandPseudoColorRenderer(rlayer.dataProvider(), 1, shader)
                    renderer.setClassificationMin(0.0)
                    renderer.setClassificationMax(1.0)
                    rlayer.setRenderer(renderer)
                    
                    # Save style to qml
                    style_path = os.path.splitext(res["msi_raster_path"])[0] + ".qml"
                    rlayer.saveNamedStyle(style_path)
                    
                    if context:
                        details = QgsProcessingContext.LayerDetails(layer_name, project, "")
                        pp = StylePostProcessor(style_path)
                        self._post_processors.append(pp)
                        details.setPostProcessor(pp)
                        context.addLayerToLoadOnCompletion(res["msi_raster_path"], details)

        # 2. Export Summary CSV Reports
        import csv
        sediment_csv_path = os.path.join(output_dir, "Sediment_Mass_Balance.csv")
        html_report_path = os.path.join(output_dir, "Temporal_Analytics_Report.html")
        
        if analytics_results:
            has_target = any(res.get("target_accretion_volume_m3") is not None for res in analytics_results)
            try:
                with open(sediment_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    header = ["Period", "Type", "Accretion_m3", "Erosion_m3", "Net_Balance_m3"]
                    if has_target:
                        header.extend(["Target_Accretion_m3", "Target_Erosion_m3", "Target_Net_Balance_m3"])
                    writer.writerow(header)

                    for res in analytics_results:
                        row = [
                            res.get("period", "N/A"),
                            res.get("analysis_type", "Overall Trend" if res.get("is_overall") else "Sequential"),
                            f"{res.get('accretion_volume_m3', 0):.2f}",
                            f"{res.get('erosion_volume_m3', 0):.2f}",
                            f"{res.get('net_sediment_balance_m3', 0):.2f}"
                        ]
                        if has_target:
                            row.extend([
                                f"{res.get('target_accretion_volume_m3', 0):.2f}",
                                f"{res.get('target_erosion_volume_m3', 0):.2f}",
                                f"{res.get('target_net_sediment_balance_m3', 0):.2f}"
                            ])
                        writer.writerow(row)
                feedback.pushInfo(f"✅ Sediment Mass Balance exported: {sediment_csv_path}")
            except PermissionError:
                feedback.pushWarning(f"⚠️ Permission Denied: Could not write to {sediment_csv_path}. Please close the file if it is open in Excel.")

            # Generate Beautiful HTML Report
            try:
                html_content = f"""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <title>Bathymetrix-AI Coastal Dynamics Report</title>
                    <style>
                        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; color: #333; background-color: #f9fbfd; }}
                        .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }}
                        h1 {{ color: #2E86C1; border-bottom: 2px solid #2E86C1; padding-bottom: 10px; }}
                        h2 {{ color: #117A65; margin-top: 30px; }}
                        h3 {{ color: #D35400; }}
                        p {{ line-height: 1.6; font-size: 15px; }}
                        .highlight {{ background-color: #E8F8F5; padding: 15px; border-left: 4px solid #117A65; border-radius: 4px; margin: 20px 0; }}
                        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                        th {{ background-color: #2E86C1; color: white; }}
                        tr:nth-child(even) {{ background-color: #f2f2f2; }}
                        .footer {{ margin-top: 50px; text-align: center; font-size: 12px; color: #777; border-top: 1px solid #eee; padding-top: 20px; }}
                        .image-placeholder {{ width: 100%; height: 300px; background-color: #e0e0e0; display: flex; align-items: center; justify-content: center; color: #777; border: 2px dashed #bbb; margin-top: 15px; font-style: italic; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>🌊 Bathymetrix-AI: Coastal Dynamics Report</h1>
                        <p>This report summarizes the multi-year coastal analysis, detailing volumetric changes and morphological stability.</p>
                        
                        <div class="highlight">
                            <h3>📊 1. Volumetric Erosion & Accretion</h3>
                            <p><b>What it measures:</b> The <i>Magnitude and Direction</i> of physical sand volume change (in cubic meters).</p>
                            <p><b>Methodology:</b> Uses a robust Linear Regression (Time-Series Trend) across all provided years. This mathematically filters out deep-water random noise and calculates the true rate of sand movement, identifying dredging (Erosion) and shoaling (Accretion).</p>
                            <p><b>Uncertainty Modeling:</b> {"Quantile Regression (Spatially Adaptive &sigma;)" if uncertainty_mode == "quantile_regression" else "Classical Z-score (Homoscedastic Gaussian)"} at {(qr_confidence * 100):.0f}% Confidence Level.</p>
                            <p><b>Importance:</b> Essential for estimating dredging budgets, understanding the sediment budget, and verifying how much sand was added or removed.</p>
                        </div>

                        <table>
                            <tr>
                                <th>Period</th>
                                <th>Analysis Type</th>
                                <th>Accretion (m³)</th>
                                <th>Erosion / Dredging (m³)</th>
                                <th>Net Balance (m³)</th>
                """
                if has_target:
                    html_content += """
                                <th>Target ROI Accretion (m³)</th>
                                <th>Target ROI Erosion (m³)</th>
                                <th>Target ROI Net Balance (m³)</th>
                    """
                html_content += """
                            </tr>
                """
                
                for res in analytics_results:
                    html_content += f"""
                            <tr>
                                <td>{res.get('period', 'N/A')}</td>
                                <td>{res.get('analysis_type', 'Sequential')}</td>
                                <td style="color: #27AE60; font-weight: bold;">+ {res.get('accretion_volume_m3', 0):,.2f}</td>
                                <td style="color: #C0392B; font-weight: bold;">- {res.get('erosion_volume_m3', 0):,.2f}</td>
                                <td><b>{res.get('net_sediment_balance_m3', 0):,.2f}</b></td>
                    """
                    if has_target:
                        html_content += f"""
                                <td style="color: #27AE60; font-weight: bold;">+ {res.get('target_accretion_volume_m3', 0):,.2f}</td>
                                <td style="color: #C0392B; font-weight: bold;">- {res.get('target_erosion_volume_m3', 0):,.2f}</td>
                                <td><b>{res.get('target_net_sediment_balance_m3', 0):,.2f}</b></td>
                        """
                    html_content += """
                            </tr>
                    """


                html_content += """
                        </table>

                        <div class="highlight" style="background-color: #FEF9E7; border-left-color: #F1C40F;">
                            <h3>📈 2. Morphological Stability Index (MSI)</h3>
                            <p><b>What it measures:</b> The <i>Volatility and Consistency</i> of the seabed, regardless of whether it is eroding or accreting.</p>
                            <p><b>Methodology:</b> A normalized index from 0 to 1. It divides the Standard Deviation of depth by the Mean depth across the time series.</p>
                            <p><b>Importance:</b> <br>
                            • Values near <b>0 (Highly Unstable)</b> indicate extreme fluctuations (e.g., shifting sandbars, collapsing navigation channels).<br>
                            • Values near <b>1 (Highly Stable)</b> indicate a quiet, unchanging seabed (e.g., flat bedrock or deep water).<br>
                            This is crucial for deciding where to route marine cables or build permanent structures.</p>
                            
                            <!-- MSI Image Placeholder -->
                            <div class="image-placeholder">
                                [ MSI Map Visualization ]<br>
                                Note: You can export the MSI map from QGIS and replace this placeholder.
                            </div>
                        </div>

                        <div class="highlight" style="background-color: #F5EEF8; border-left-color: #9B59B6;">
                            <h3>🗺️ 3. Shoreline Change Polygons</h3>
                            <p><b>What it measures:</b> The horizontal migration of the coastline (Water-Land boundary) over time.</p>
                            <p><b>Importance:</b> Accretion polygons (Green) show beach advancement, while Erosion polygons (Red) highlight coastal retreat. These vectors allow precise area calculations of lost or gained beach real estate.</p>
                            
                            <!-- Shoreline Image Placeholder -->
                            <div class="image-placeholder" style="border-color: #9B59B6; background-color: #FDFEFE;">
                                [ Shoreline Change Polygons Visualization ]<br>
                                Note: You can export the Shoreline map from QGIS and replace this placeholder.
                            </div>
                        </div>

                        <div class="footer">
                            Generated autonomously by Bathymetrix-AI Coastal Intelligence Engine.<br>
                            QGIS Processing Framework
                        </div>
                    </div>
                </body>
                </html>
                """
                
                with open(html_report_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                feedback.pushInfo(f"📄 Comprehensive HTML Report exported: {html_report_path}")
            except Exception as e:
                feedback.pushWarning(f"⚠️ Could not generate HTML report: {str(e)}")

        benthic_csv_path = os.path.join(output_dir, "Benthic_Classification_Stats.csv")
        if benthic_results:
            try:
                with open(benthic_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Year", "Veg_km2"])
                    for yr, res in sorted(benthic_results.items()):
                        if res:
                            writer.writerow([
                                yr,
                                f"{res.get('veg_area_km2', 0):.3f}"
                            ])
                feedback.pushInfo(f"✅ Benthic Stats exported: {benthic_csv_path}")
            except PermissionError:
                feedback.pushWarning(f"⚠️ Permission Denied: Could not write to {benthic_csv_path}. Please close the file if it is open in Excel.")

        feedback.pushInfo(f"🎉 QGIS Layers successfully loaded to map canvas!")
