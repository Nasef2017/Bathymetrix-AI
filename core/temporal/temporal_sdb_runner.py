import os
import shutil
from typing import Dict, Any, Optional
from qgis.core import (
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsProcessingException,
    QgsVectorLayer,
    QgsFeature,
    QgsField,
    QgsVectorFileWriter,
    QgsWkbTypes,
    QgsProject,
)
from qgis.PyQt.QtCore import QVariant
import processing


class TemporalSDBRunner:
    """
    Executes the SDB MasterFlow pipeline for each year dynamically.
    Flexible ICESat-2 dataset handling:
    - Uses per-year shapefile inside subfolders (/2020/icesat.shp) if present
    - OR uses global ICESat-2 layer selected in UI (filtered by year attribute if present, or all points if no year field)
    """

    def __init__(self, master_output_folder: str):
        self.master_output_folder = master_output_folder

    def _resolve_depth_field(self, layer_path: str, default_field: str, feedback: QgsProcessingFeedback) -> str:
        """Attempts to find the provided field in the layer; if missing, falls back to common depth names."""
        from qgis.core import QgsVectorLayer
        vlayer = QgsVectorLayer(layer_path, "temp", "ogr")
        if not vlayer.isValid():
            return default_field
            
        field_names = [f.name() for f in vlayer.fields()]
        if default_field in field_names:
            return default_field
            
        # Common depth field names to try if default_field is not found
        common_names = ["depth", "z", "elevation", "z_value", "value", "ortho_h", "dz"]
        lower_fields = {f.lower(): f for f in field_names}
        
        for cn in common_names:
            if cn in lower_fields:
                detected = lower_fields[cn]
                feedback.pushInfo(f"⚠️ Field '{default_field}' not found in {os.path.basename(layer_path)}. Auto-detected '{detected}' instead.")
                return detected
                
        # Fallback to the last numeric field
        for f in reversed(vlayer.fields().toList()):
            if f.isNumeric():
                feedback.pushInfo(f"⚠️ Field '{default_field}' not found. Auto-fallback to numeric field '{f.name()}'.")
                return f.name()
                
        return default_field

    def run_year(
        self,
        year_info: Dict[str, Any],
        masterflow_params: Dict[str, Any],
        feedback: QgsProcessingFeedback,
        context: QgsProcessingContext,
    ) -> Dict[str, str]:
        """
        Executes SDB MasterFlow for a single year.
        """
        year = year_info["year"]
        feedback.pushInfo("==================================================")
        feedback.pushInfo(f"🌊 [TEMPORAL SDB] Starting SDB MasterFlow for Year: {year}")
        feedback.pushInfo("==================================================")

        # 1. Create Year Output Folder
        year_out_dir = os.path.join(self.master_output_folder, f"SDB_{year}")
        os.makedirs(year_out_dir, exist_ok=True)

        # 2. Determine ICESat-2 Layer for this year
        if year_info.get("icesat_file_path") and os.path.exists(year_info["icesat_file_path"]):
            feedback.pushInfo(f"📌 Found per-year ICESat-2 file in subfolder: {os.path.basename(year_info['icesat_file_path'])}")
            icesat_year_layer_path = year_info["icesat_file_path"]
        elif year_info.get("icesat_layer") and year_info["icesat_layer"].isValid():
            icesat_year_layer_path = self._prepare_year_vector_layer(
                raw_layer=year_info["icesat_layer"],
                year_field=year_info.get("icesat_year_field", ""),
                target_year=year,
                output_dir=year_out_dir,
                prefix="ICESat2_Points",
                feedback=feedback,
            )
        else:
            raise QgsProcessingException(f"No ICESat-2 vector points found for year {year}!")

        # 3. Extract / Filter Unseen Test Points for this specific year (if provided)
        unseen_year_layer_path = None
        if year_info.get("unseen_file_path") and os.path.exists(year_info["unseen_file_path"]):
            unseen_year_layer_path = year_info["unseen_file_path"]
        elif year_info.get("unseen_layer") and year_info["unseen_layer"].isValid():
            unseen_year_layer_path = self._prepare_year_vector_layer(
                raw_layer=year_info["unseen_layer"],
                year_field=year_info.get("unseen_year_field", ""),
                target_year=year,
                output_dir=year_out_dir,
                prefix="Unseen_Points",
                feedback=feedback,
            )

        # 4. Assemble parameters for MasterFlow algorithm
        run_params = masterflow_params.copy()
        run_params["INPUT_RASTER"] = year_info["image_path"]
        run_params["INPUT_TRAIN"] = icesat_year_layer_path
        run_params["OUTPUT_FOLDER"] = year_out_dir
        
        main_depth_field = run_params.get("FIELD_DEPTH", "")
        
        # Smartly resolve field for main training points
        resolved_main_depth = self._resolve_depth_field(
            icesat_year_layer_path, main_depth_field, feedback
        )
        run_params["FIELD_DEPTH"] = resolved_main_depth
        
        ui_adaptive_depth = run_params.get("FIELD_ADAPTIVE_DEPTH", main_depth_field)
        ui_test_depth = run_params.get("FIELD_TEST_DEPTH", main_depth_field)

        if year_info.get("control_path"):
            run_params["INPUT_ADAPTIVE_TRAIN"] = year_info["control_path"]
            run_params["ENABLE_ADAPTIVE"] = True
            
            # Smartly resolve field for Control Points
            run_params["FIELD_ADAPTIVE_DEPTH"] = self._resolve_depth_field(
                year_info["control_path"], ui_adaptive_depth, feedback
            )
            feedback.pushInfo(f"🎯 Control Points found for {year}: Phase 04 Depth-Dependent Residual Calibration ENABLED.")
        else:
            run_params["ENABLE_ADAPTIVE"] = False
            feedback.pushInfo(f"⏭️ No Control Points found for {year}: Phase 04 Depth-Dependent Residual Calibration SKIPPED.")

        if unseen_year_layer_path:
            run_params["ENABLE_VALIDATION"] = True
            run_params["INPUT_TEST"] = unseen_year_layer_path
            
            # Smartly resolve field for Validation Points
            run_params["FIELD_TEST_DEPTH"] = self._resolve_depth_field(
                unseen_year_layer_path, ui_test_depth, feedback
            )

        # 5. Run QGIS SDB MasterFlow algorithm
        feedback.pushInfo(f"🚀 Running MasterFlow on image: {os.path.basename(year_info['image_path'])}")
        
        try:
            child_context = QgsProcessingContext()
            child_context.setFeedback(feedback)
            if hasattr(context, "project"):
                child_context.setProject(context.project())

            res = processing.run(
                "sdb_tools:sdb_master_orchestrator",
                run_params,
                context=child_context,
                feedback=feedback,
                is_child_algorithm=True
            )
        except Exception as e:
            feedback.reportError(f"Error running MasterFlow for year {year}: {str(e)}")
            raise QgsProcessingException(f"MasterFlow failed for year {year}: {str(e)}")

        # 6. Locate generated SDB depth map & uncertainty map in year output folder
        sdb_depth_map = self._find_raster_output(year_out_dir, ["Phase04_Final_Depth", "Phase03_Depth", "SDB_Depth"])
        uncertainty_map = self._find_raster_output(year_out_dir, ["Uncertainty", "Error", "Residual"])

        if not sdb_depth_map:
            # Fallback scan for any tif generated in the folder
            for root, _, files in os.walk(year_out_dir):
                for f in files:
                    lower_f = f.lower()
                    if lower_f.endswith(".tif") and "mask" not in lower_f and "glint" not in lower_f and "feature_stack" not in lower_f and "features" not in lower_f:
                        sdb_depth_map = os.path.join(root, f)
                        break
                if sdb_depth_map:
                    break

        if not sdb_depth_map:
            raise QgsProcessingException(f"Failed to find generated SDB depth map raster in '{year_out_dir}'.")

        feedback.pushInfo(f"✅ Year {year} SDB Completed: {os.path.basename(sdb_depth_map)}")

        # ---------------------------------------------------------
        # [NEW] Locate strictly Linear Regression SDB for MSI & Volumetric Change
        # ---------------------------------------------------------
        feedback.pushInfo(f"📈 [TEMPORAL SDB] Locating isolated Linear Regression SDB for Analytics...")
        sdb_linear_map = sdb_depth_map # Fallback in case of failure
        linear_uncertainty_map = uncertainty_map
        
        candidate_lr_map = os.path.join(year_out_dir, "Phase_03_Initial_Modeling", "Linear_Regression", "Linear_Regression_Depth.tif")
        candidate_lr_uncert = os.path.join(year_out_dir, "Phase_03_Initial_Modeling", "Linear_Regression", "Linear_Regression_Uncertainty.tif")

        if not os.path.exists(candidate_lr_map):
            candidate_lr_map = os.path.join(year_out_dir, "Linear_Regression", "Linear_Regression_Depth.tif")
        if not os.path.exists(candidate_lr_uncert):
            candidate_lr_uncert = os.path.join(year_out_dir, "Linear_Regression", "Linear_Regression_Uncertainty.tif")
            
        if os.path.exists(candidate_lr_map):
            sdb_linear_map = candidate_lr_map
            feedback.pushInfo(f"✅ Found isolated Linear Regression SDB for {year}.")
        else:
            feedback.pushWarning(f"⚠️ Linear Regression map not found at {candidate_lr_map}. Falling back to primary map.")

        if os.path.exists(candidate_lr_uncert):
            linear_uncertainty_map = candidate_lr_uncert

        return {
            "year": year,
            "year_out_dir": year_out_dir,
            "sdb_depth_map": sdb_depth_map,
            "sdb_linear_map": sdb_linear_map,
            "uncertainty_map": uncertainty_map,
            "linear_uncertainty_map": linear_uncertainty_map,
            "image_path": year_info["image_path"],
            "icesat_path": icesat_year_layer_path,
        }

    def _prepare_year_vector_layer(
        self,
        raw_layer: QgsVectorLayer,
        year_field: str,
        target_year: int,
        output_dir: str,
        prefix: str,
        feedback: QgsProcessingFeedback,
    ) -> str:
        """
        Filters vector layer by year attribute if specified and present.
        If no year attribute exists, copies/saves all features for the target year.
        """
        out_shp = os.path.join(output_dir, f"{prefix}_{target_year}.shp")
        if os.path.exists(out_shp):
            try:
                os.remove(out_shp)
            except Exception:
                pass

        fields = raw_layer.fields()
        writer = QgsVectorFileWriter(
            out_shp,
            "UTF-8",
            fields,
            raw_layer.wkbType(),
            raw_layer.sourceCrs(),
            "ESRI Shapefile",
        )

        has_filter = bool(year_field and year_field in [f.name() for f in fields])
        count = 0

        for feat in raw_layer.getFeatures():
            if has_filter:
                val = feat[year_field]
                if val is not None and str(val).strip() != "":
                    try:
                        feat_year = int(float(val))
                        if feat_year != target_year:
                            continue
                    except (ValueError, TypeError):
                        pass

            writer.addFeature(feat)
            count += 1

        del writer

        if count == 0:
            if has_filter:
                feedback.pushWarning(
                    f"⚠️ No points matched year {target_year} in field '{year_field}'. Using all points in layer as fallback."
                )
            # Rewrite all features as fallback
            writer = QgsVectorFileWriter(
                out_shp, "UTF-8", fields, raw_layer.wkbType(), raw_layer.sourceCrs(), "ESRI Shapefile"
            )
            for feat in raw_layer.getFeatures():
                writer.addFeature(feat)
            del writer

        return out_shp

    def _resolve_depth_field(self, layer_path: str, requested_field: str, feedback: QgsProcessingFeedback) -> str:
        """
        Attempts to resolve the depth field for a shapefile if the requested field does not exist.
        """
        layer = QgsVectorLayer(layer_path, "temp", "ogr")
        if not layer.isValid():
            return requested_field
            
        fields = [f.name() for f in layer.fields()]
        if requested_field in fields:
            return requested_field
            
        # Fallback to common depth fields
        common_depth_fields = ["depth", "z", "elevation", "ortho_h", "z_depth", "z_value", "z_m"]
        for cf in common_depth_fields:
            for f in fields:
                if f.lower() == cf:
                    feedback.pushWarning(f"⚠️ Requested depth field '{requested_field}' not found. Falling back to '{f}'.")
                    return f
                    
        # Fallback to the last numeric field
        for i in range(len(layer.fields()) - 1, -1, -1):
            if layer.fields().at(i).type() in [QVariant.Double, QVariant.Int]:
                f_name = layer.fields().at(i).name()
                feedback.pushWarning(f"⚠️ No common depth field found. Falling back to last numeric field: '{f_name}'.")
                return f_name
                
        return requested_field

    def _find_raster_output(self, folder: str, keywords: list) -> Optional[str]:
        for kw in keywords:
            matches = []
            for root, dirs, files in os.walk(folder):
                for f in files:
                    if f.lower().endswith(".tif") and kw.lower() in f.lower():
                        matches.append(os.path.join(root, f))
            
            if matches:
                # Prioritize OSW_Clipped versions if they exist for this keyword
                for m in matches:
                    if "osw_clipped" in m.lower():
                        return m
                return matches[0]
                
        return None
