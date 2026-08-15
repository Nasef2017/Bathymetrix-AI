import os
import warnings
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterFile,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterDefinition,
    QgsProcessingOutputFolder,
    QgsProcessingException,
    QgsProcessing,
    QgsProcessingParameterFolderDestination
)

warnings.filterwarnings("ignore")


class SDBCoastalDynamics(QgsProcessingAlgorithm):
    r"""
    QGIS Processing Algorithm: 4. Coastal Dynamics Analysis (Temporal Intelligence).
    
    This tool serves as the UI and Orchestrator for the Temporal Analytics Engine.
    It takes multi-year Satellite-Derived Bathymetry (SDB) maps and computes
    their morphological evolution.

    ### Algorithm Outputs:
    1. **Volumetric Sediment Change**: Computes mass balance (Erosion/Accretion) in $m^3$.
       Supports robust multi-year Linear Regression: $Z_{diff} = Z - \bar{Z}$.
    2. **Morphological Stability Index (MSI)**: $MSI = 1 - \frac{\sigma}{\mu}$
    3. **Shoreline Migration**: Extracts the 0m contour to track coastal retreat/advance.
    4. **HTML Analytics Report**: Generates an automated, user-friendly data dashboard.
    
    ### Uncertainty Masking (StatCD):
    To eliminate optical noise, changes are masked against the combined uncertainty of the rasters:
    $|\Delta Z| > U_{comb}$ (Uses Classical, MAD, or Quantile adaptive modeling).
    """


    INPUT_MASTER_FOLDER = "INPUT_MASTER_FOLDER"
    USER_OUTPUT_FOLDER = "USER_OUTPUT_FOLDER"
    SHORELINE_ROI = "SHORELINE_ROI"
    COMPARISON_MODE = "COMPARISON_MODE"
    OVERALL_TREND_METHOD = "OVERALL_TREND_METHOD"

    # Advanced parameters
    SHORELINE_DEPTH = "SHORELINE_DEPTH"
    GRID_RESOLUTION = "GRID_RESOLUTION"
    THRESHOLDING_MODE = "THRESHOLDING_MODE"
    UNCERTAINTY_THRESHOLD = "UNCERTAINTY_THRESHOLD"
    DEPTH_NOISE_FACTOR = "DEPTH_NOISE_FACTOR"
    MAX_ANALYZED_DEPTH = "MAX_ANALYZED_DEPTH"
    UNCERTAINTY_MODE = "UNCERTAINTY_MODE"
    QUANTILE_LEVEL = "QUANTILE_LEVEL"
    MSI_INSTABILITY_THRESHOLD = "MSI_INSTABILITY_THRESHOLD"

    TREND_METHODS_NAMES = ["Long-term Trend", "Net Difference"]
    COMPARISON_MODE_NAMES = ["Sequential (Year-to-Year)", "Baseline Reference (First Year Fixed)"]

    def name(self):
        return "sdb_06_coastal_dynamics"

    def displayName(self):
        return "5. Coastal Dynamics Analysis"

    def group(self):
        return ""

    def groupId(self):
        return ""

    def createInstance(self):
        return SDBCoastalDynamics()

    def shortHelpString(self):
        return """
        <div style="font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.5; color: #2C3E50;">
            <h2 style="margin-bottom: 5px; color: #2E86C1;">🌊 Bathymetrix-AI: Coastal Dynamics Analysis</h2>
            <p style="margin-top: 0; margin-bottom: 15px; font-size: 13px;">
                An advanced AI-driven coastal engineering toolkit for autonomous <b>multi-year morphological analysis</b> using time-series SDB maps. 
                Quantifies volumetric sediment transport (m³), identifies erosion/accretion zones, tracks shoreline migration, and computes seabed stability.
            </p>
            
            <h3 style="color: #117A65; margin-bottom: 5px; border-bottom: 2px solid #117A65; padding-bottom: 3px;">🔬 Physics & Analytical Engine</h3>
            
            <p style="margin-bottom: 2px; font-size: 12px;"><b>1. Morphological Stability Index (MSI):</b></p>
            <p style="font-style: italic; background-color: #F8F9F9; padding: 8px 12px; border-left: 4px solid #117A65; font-size: 11px; margin-top: 2px;">
                A relative, depth-normalized metric following the Beer-Lambert law. Dividing temporal standard deviation by depth eliminates optical noise in deeper waters:<br>
                <b>Equation:</b> <code>MSI = 1.0 - (Std_Dev_Z / (|Mean_Z| + 0.5))</code>
            </p>

            <p style="margin-bottom: 2px; font-size: 12px;"><b>2. Volumetric Sediment Tracking & Dynamic Noise Floor (StatCD):</b></p>
            <p style="font-style: italic; background-color: #F8F9F9; padding: 8px 12px; border-left: 4px solid #117A65; font-size: 11px; margin-top: 2px;">
                Computes net volumetric sediment change (m³), erosion volume, accretion volume, and annual rates using Linear Regression across multi-temporal rasters. Employs a <b>Depth-Dependent Noise Floor</b> to prevent false changes in deep water:<br>
                <b>Equation:</b> <code>Dynamic_Threshold = Otsu_Base_Threshold + (Depth_Noise_Factor × |Depth|)</code>
            </p>

            <h3 style="color: #D35400; margin-top: 15px; margin-bottom: 5px; border-bottom: 2px solid #D35400; padding-bottom: 3px;">🧠 AI Thresholding Modes</h3>
            <ul style="font-size: 12px; margin-top: 5px; padding-left: 20px;">
                <li><b>Auto (Otsu's Method):</b> Automatically determines optimal statistical thresholds for both MSI and Volumetric Base Noise Floor based on inter-class histogram variance.</li>
                <li><b>Manual:</b> Custom user-defined thresholds for specialized coastal engineering specifications.</li>
            </ul>

            <h3 style="color: #8E44AD; margin-top: 15px; margin-bottom: 5px; border-bottom: 2px solid #8E44AD; padding-bottom: 3px;">📌 Inputs</h3>
            <p style="font-size: 12px; margin-top: 5px;">
                Requires the Master Output Workspace generated by <b>SDB Spatiotemporal Masterflow</b> (containing <code>Year_*</code> subfolders with SDB raster maps).
            </p>
            <br><b>Developer:</b> Mohamed Aly Nasef
        </div>
        """

    def helpString(self):
        return self.shortHelpString()

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_MASTER_FOLDER,
                "📂 [0] Master Output Folder (From Spatiotemporal Masterflow)",
                behavior=QgsProcessingParameterFile.Folder,
            )
        )
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.USER_OUTPUT_FOLDER,
                "📁 Output Results Folder",
                optional=False
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.SHORELINE_ROI,
                "🎯 [1] Study Area / Target ROI (Boundary & Clip Polygon)",
                types=[QgsProcessing.TypeVectorPolygon],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.COMPARISON_MODE,
                "📅 [2] Temporal Comparison Mode",
                options=self.COMPARISON_MODE_NAMES,
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.OVERALL_TREND_METHOD,
                "📈 [3] Overall Mass Balance Trend Method",
                options=self.TREND_METHODS_NAMES,
                defaultValue=0,
            )
        )

        # Advanced Parameters
        param_depth = QgsProcessingParameterNumber(
            self.SHORELINE_DEPTH,
            "⚙️ [4] Shoreline Depth Threshold (m) [Default: 0.0]",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.0,
        )
        param_depth.setFlags(param_depth.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(param_depth)

        param_grid = QgsProcessingParameterNumber(
            self.GRID_RESOLUTION,
            "⚙️ [4] Grid Resolution for Mass Balance (m) [Default: 5.0]",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=5.0,
            minValue=0.1,
        )
        param_grid.setFlags(param_grid.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(param_grid)

        param_thresh_mode = QgsProcessingParameterEnum(
            self.THRESHOLDING_MODE,
            "🧠 [AI] Detection Thresholding Mode (Otsu Auto / Manual)",
            options=["Auto (Otsu's Method)", "Manual"],
            defaultValue=0
        )
        param_thresh_mode.setFlags(param_thresh_mode.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(param_thresh_mode)

        param_unc = QgsProcessingParameterNumber(
            self.UNCERTAINTY_THRESHOLD,
            "⚙️ [Manual Override] Volumetric Base Threshold (m) [Default: 0.15]",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.15,
            minValue=0.0,
        )
        param_unc.setFlags(param_unc.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(param_unc)

        param_depth_noise = QgsProcessingParameterNumber(
            self.DEPTH_NOISE_FACTOR,
            "🛡️ [Physics] Depth-Dependent Noise Factor (e.g. 0.10 = 10%)",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.10,
            minValue=0.0,
            maxValue=1.0,
        )
        param_depth_noise.setFlags(param_depth_noise.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(param_depth_noise)

        param_doc = QgsProcessingParameterNumber(
            self.MAX_ANALYZED_DEPTH,
            "🌊 [Physics] Depth of Closure (Max Analyzed Depth) in meters [Default: 25.0]",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=25.0,
            minValue=0.1,
        )
        param_doc.setFlags(param_doc.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(param_doc)

        param_umode = QgsProcessingParameterEnum(
            self.UNCERTAINTY_MODE,
            "🎲 [Advanced] Temporal Uncertainty Mode",
            options=["Classical Z-score (1.96σ)", "Non-parametric (MAD)", "Quantile Regression"],
            defaultValue=2,
        )
        param_umode.setFlags(param_umode.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(param_umode)

        param_quantile = QgsProcessingParameterNumber(
            self.QUANTILE_LEVEL,
            "🎲 [Advanced] Quantile Regression Confidence Level (0.50-0.99)",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.95,
            minValue=0.5,
            maxValue=0.99
        )
        param_quantile.setFlags(param_quantile.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(param_quantile)
        
        param_msi = QgsProcessingParameterNumber(
            self.MSI_INSTABILITY_THRESHOLD,
            "⚙️ [Manual Override] MSI Instability Threshold [0.5-1.0]",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.75,
            minValue=0.5,
            maxValue=1.0,
        )
        param_msi.setFlags(param_msi.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(param_msi)
        
        self.addOutput(
            QgsProcessingOutputFolder(
                "OUTPUT_FOLDER",
                "Coastal Dynamics Results Folder"
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        try:
            from ...core.temporal.shoreline_tracker import ShorelineDynamicsTracker
            from ...core.temporal.temporal_analytics import TemporalAnalyticsEngine
            from ...core.temporal.temporal_reporting import TemporalReportGenerator
            from ...infrastructure.logging import append_log
        except ImportError:
            from Bathymetrix_AI.core.temporal.shoreline_tracker import ShorelineDynamicsTracker
            from Bathymetrix_AI.core.temporal.temporal_analytics import TemporalAnalyticsEngine
            from Bathymetrix_AI.core.temporal.temporal_reporting import TemporalReportGenerator
            from Bathymetrix_AI.infrastructure.logging import append_log

        master_folder = self.parameterAsString(parameters, self.INPUT_MASTER_FOLDER, context)
        user_out_folder = self.parameterAsString(parameters, self.USER_OUTPUT_FOLDER, context)
        target_roi_shp = self.parameterAsVectorLayer(parameters, self.SHORELINE_ROI, context)
        comp_mode_idx = self.parameterAsEnum(parameters, self.COMPARISON_MODE, context)
        trend_method_idx = self.parameterAsEnum(parameters, self.OVERALL_TREND_METHOD, context)
        
        comp_mode_str = self.COMPARISON_MODE_NAMES[comp_mode_idx]
        trend_method_str = self.TREND_METHODS_NAMES[trend_method_idx]
        
        # Advanced Parameters
        shoreline_depth = self.parameterAsDouble(parameters, self.SHORELINE_DEPTH, context)
        grid_res = self.parameterAsDouble(parameters, self.GRID_RESOLUTION, context)
        
        thresh_mode_idx = self.parameterAsEnum(parameters, self.THRESHOLDING_MODE, context)
        thresh_mode_str = "Auto" if thresh_mode_idx == 0 else "Manual"
        
        uncert_thresh = self.parameterAsDouble(parameters, self.UNCERTAINTY_THRESHOLD, context)
        depth_noise_factor = self.parameterAsDouble(parameters, self.DEPTH_NOISE_FACTOR, context)
        max_analyzed_depth = self.parameterAsDouble(parameters, self.MAX_ANALYZED_DEPTH, context)
        uncert_mode_idx = self.parameterAsEnum(parameters, self.UNCERTAINTY_MODE, context)
        uncert_mode_str = ["classical", "mad", "quantile"][uncert_mode_idx]
        qr_conf = self.parameterAsDouble(parameters, self.QUANTILE_LEVEL, context)
        
        msi_thresh = self.parameterAsDouble(parameters, self.MSI_INSTABILITY_THRESHOLD, context)

        if not master_folder or not os.path.exists(master_folder):
            raise QgsProcessingException("Invalid Master Folder provided.")

        if not user_out_folder or not os.path.exists(user_out_folder):
            os.makedirs(user_out_folder, exist_ok=True)
            
        out_folder = user_out_folder
        os.makedirs(out_folder, exist_ok=True)
        
        # Initialize full process log
        log_file_path = os.path.join(out_folder, "Coastal_Dynamics_Log.txt")
        
        with open(log_file_path, "w", encoding="utf-8") as f:
            pass
            
        import time
        start_time = time.time()
        start_str = time.strftime('%H:%M:%S', time.localtime(start_time))

        append_log("════════════════════════════════════════════════════════════", log_file_path, feedback)
        append_log("Coastal Dynamics Analysis".center(60), log_file_path, feedback)
        append_log("════════════════════════════════════════════════════════════", log_file_path, feedback)
        append_log(f"Started: {start_str}", log_file_path, feedback)
        append_log("════════════════════════════════════════════════════════════\n", log_file_path, feedback)
        
        append_log("  [Initialization] Configuration Settings", log_file_path, feedback)
        append_log(f"      → Master Folder: {master_folder}", log_file_path, feedback)
        append_log(f"      → Comparison Mode: {comp_mode_str}", log_file_path, feedback)
        append_log(f"      → Overall Trend Method: {trend_method_str}", log_file_path, feedback)
        append_log(f"      → Shoreline Depth: {shoreline_depth}m", log_file_path, feedback)
        append_log(f"      → Grid Resolution: {grid_res}m", log_file_path, feedback)
        append_log(f"      → Volumetric Threshold: {uncert_thresh}m", log_file_path, feedback)
        append_log("  ✓ Settings loaded successfully\n", log_file_path, feedback)

        append_log("  [Scanning] Input Folders", log_file_path, feedback)

        year_folders = [f for f in os.listdir(master_folder) if f.startswith("Year_") and os.path.isdir(os.path.join(master_folder, f))]
        if not year_folders:
            err = "✗ ERROR: No Year_XXXX folders found in the Master Folder."
            append_log(err, log_file_path, feedback)
            raise QgsProcessingException(err)

        yearly_sdb_results = {}
        for yf in year_folders:
            year_str = yf.split("_")[1]
            try:
                year = int(year_str)
            except:
                continue
            
            y_dir = os.path.join(master_folder, yf)
            
            sdb_depth_map = None
            sdb_linear_map = None
            uncertainty_map = None
            
            for root, _, files in os.walk(y_dir):
                for f in files:
                    lower_f = f.lower()
                    if lower_f == f"sdb {year}.tif" or lower_f == f"sdb_{year}.tif":
                        sdb_depth_map = os.path.join(root, f)
                    if "linear_regression_depth" in lower_f:
                        sdb_linear_map = os.path.join(root, f)
                    if "linear_regression_uncertainty" in lower_f:
                        uncertainty_map = os.path.join(root, f)
                    elif "uncertainty" in lower_f and not uncertainty_map:
                        uncertainty_map = os.path.join(root, f)
            
            if not sdb_depth_map and sdb_linear_map:
                sdb_depth_map = sdb_linear_map
                
            if sdb_depth_map:
                yearly_sdb_results[year] = {
                    "year": year,
                    "year_out_dir": y_dir,
                    "sdb_depth_map": sdb_depth_map,
                    "sdb_linear_map": sdb_linear_map if sdb_linear_map else sdb_depth_map,
                    "uncertainty_map": uncertainty_map,
                    "linear_uncertainty_map": uncertainty_map
                }
                append_log(f"      → Found SDB for Year {year}: {os.path.basename(sdb_depth_map)}", log_file_path, feedback)
            else:
                append_log(f"  ⚠ WARNING: No SDB map found for Year {year} in folder {yf}", log_file_path, feedback)
            
        if not yearly_sdb_results:
            err = "✗ ERROR: Could not extract any valid SDB maps from the provided folder."
            append_log(err, log_file_path, feedback)
            raise QgsProcessingException(err)
            
        append_log(f"  ✓ Found {len(yearly_sdb_results)} valid years\n", log_file_path, feedback)
            
        target_roi_path = None
        if target_roi_shp and target_roi_shp.isValid():
            from qgis.core import QgsVectorFileWriter
            target_roi_path = os.path.join(out_folder, "Target_ROI_Analytics.shp")
            options = QgsVectorFileWriter.SaveVectorOptions()
            options.driverName = "ESRI Shapefile"
            options.fileEncoding = "UTF-8"
            QgsVectorFileWriter.writeAsVectorFormatV2(target_roi_shp, target_roi_path, context.transformContext(), options)
            append_log(f"      → Target ROI provided and saved to {target_roi_path}", log_file_path, feedback)
        
        # Step 1: Analytics Engine
        append_log("  [Step 1] Temporal Analytics Engine", log_file_path, feedback)
        append_log("      → Running Volume & Trend calculations...", log_file_path, feedback)
        bathymetric_out_folder = os.path.join(out_folder, "Bathymetric_Change_Analysis")
        os.makedirs(bathymetric_out_folder, exist_ok=True)
        
        sdb_maps = {y: res["sdb_depth_map"] for y, res in yearly_sdb_results.items()}
        uncert_maps = {y: res["uncertainty_map"] for y, res in yearly_sdb_results.items()}
        
        analytics_engine = TemporalAnalyticsEngine()
        analytics_results = analytics_engine.process_temporal_change(
            sdb_maps=sdb_maps,
            uncertainty_maps=uncert_maps,
            output_dir=bathymetric_out_folder,
            feedback=feedback,
            osw_shp=target_roi_path,
            overall_trend_method=trend_method_str,
            comparison_mode=comp_mode_str,
            target_roi_path=target_roi_path,
            uncertainty_mode=uncert_mode_str,
            qr_confidence=qr_conf,
            uncert_thresh_mode=thresh_mode_str,
            uncert_thresh=uncert_thresh,
            depth_noise_factor=depth_noise_factor,
            max_analyzed_depth=max_analyzed_depth,
            msi_mode=thresh_mode_str,
            msi_threshold=msi_thresh
        )
        append_log("  ✓ Analytics Engine completed\n", log_file_path, feedback)
        
        # Step 2: Shoreline Dynamics
        append_log("  [Step 2] Shoreline Dynamics", log_file_path, feedback)
        append_log("      → Running Shoreline Tracker...", log_file_path, feedback)
        shoreline_out_folder = os.path.join(out_folder, "Shoreline_Dynamics")
        os.makedirs(shoreline_out_folder, exist_ok=True)
        
        sdb_maps = {y: res["sdb_depth_map"] for y, res in yearly_sdb_results.items()}
        
        tracker = ShorelineDynamicsTracker()
        shoreline_results = tracker.process_shorelines(
            sdb_maps=sdb_maps,
            output_dir=shoreline_out_folder,
            feedback=feedback,
            osw_shp=target_roi_path,
            shoreline_depth=shoreline_depth,
            comparison_mode=comp_mode_str
        )
        append_log("  ✓ Shoreline Dynamics Tracker completed\n", log_file_path, feedback)
        
        # Step 3: Reporting
        append_log("  [Step 3] Coastal Dynamics Dashboard", log_file_path, feedback)
        append_log("      → Generating Report...", log_file_path, feedback)
        reporter = TemporalReportGenerator()
        reporter.generate_layer_group_and_reports(
            yearly_sdb_results=yearly_sdb_results,
            change_polygons=shoreline_results,
            analytics_results=analytics_results,
            benthic_results={},
            output_dir=out_folder,
            feedback=feedback,
            uncertainty_mode=uncert_mode_str,
            qr_confidence=qr_conf
        )
        
        dashboard_path = os.path.join(out_folder, "Temporal_Analytics_Report.html")
        append_log(f"      → Dashboard saved at: {dashboard_path}", log_file_path, feedback)
        append_log("  ✓ Report generation completed\n", log_file_path, feedback)
        
        end_time = time.time()
        elapsed = end_time - start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)

        append_log("════════════════════════════════════════════════════════════", log_file_path, feedback)
        append_log(f"✓ Coastal Dynamics Completed in {mins}m {secs}s".center(60), log_file_path, feedback)
        append_log("════════════════════════════════════════════════════════════\n", log_file_path, feedback)
        
        return {"OUTPUT_FOLDER": out_folder}
