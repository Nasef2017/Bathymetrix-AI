from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProject,
    QgsCoordinateTransform,
    QgsPointXY,
)
import os
import time
import numpy as np
import pandas as pd

try:
    import rasterio
    import matplotlib

    # Use Agg backend to avoid QGIS crashing when trying to show window
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    libs_installed = True
except ImportError:
    libs_installed = False


class EvaluateModelAlgorithm(QgsProcessingAlgorithm):
    INPUT_PREDICTED_RASTER = "INPUT_PREDICTED_RASTER"
    INPUT_UNSEEN_VEC = "INPUT_UNSEEN_VEC"
    DEPTH_FIELD_VEC = "DEPTH_FIELD_VEC"
    OUTPUT_FILE = "OUTPUT_FILE"  # Reverted to FILE to match Master Workflow expectation

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_PREDICTED_RASTER, "Input Predicted Depth Raster"
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_UNSEEN_VEC, "Unseen Validation Points (Vector)"
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.DEPTH_FIELD_VEC,
                "True Depth Field",
                parentLayerParameterName=self.INPUT_UNSEEN_VEC,
                type=QgsProcessingParameterField.Numeric,
            )
        )

        # We accept a FILE path (usually for the report), but we will save other files in the same directory
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_FILE,
                "Output Evaluation Report",
                fileFilter="Text files (*.txt)",
            )
        )

    def name(self):
        return "sdb_evaluate_model"

    def displayName(self):
        return "SDB: Evaluate Model (Unseen Data)"

    def group(self):
        return ""

    def groupId(self):
        return ""

    def shortHelpString(self):
        return "Evaluates predicted depth against unseen data. Generates Report, Scatter Plot, and CSV in the output file directory."

    def createInstance(self):
        return EvaluateModelAlgorithm()

    def _transform_point_if_needed(self, x, y, sample_crs, raster_crs, feedback):
        if sample_crs == raster_crs:
            return x, y
        try:
            tr = QgsCoordinateTransform(sample_crs, raster_crs, QgsProject.instance())
            p = tr.transform(QgsPointXY(x, y))
            return float(p.x()), float(p.y())
        except Exception as e:
            feedback.pushWarning(
                f"CRS transform failed: {e}. Assuming coordinates match raster CRS."
            )
            return x, y

    def processAlgorithm(self, parameters, context, feedback):
        if not libs_installed:
            raise RuntimeError(
                "Required libraries (scikit-learn, rasterio, matplotlib) are not installed."
            )

        predicted_raster = self.parameterAsRasterLayer(
            parameters, self.INPUT_PREDICTED_RASTER, context
        )
        validation_layer = self.parameterAsVectorLayer(
            parameters, self.INPUT_UNSEEN_VEC, context
        )
        depth_field = self.parameterAsString(parameters, self.DEPTH_FIELD_VEC, context)

        # Get the report file path provided by user or Master Workflow
        output_txt_path = self.parameterAsFileOutput(
            parameters, self.OUTPUT_FILE, context
        )

        # --- INTELLIGENT PATH HANDLING ---
        # Determine the directory from the file path
        output_folder = os.path.dirname(output_txt_path)

        # Define other file names relative to that directory
        # If MasterWorkflow sends "report.txt", we save "scatter_plot.png" next to it.
        base_name = os.path.splitext(os.path.basename(output_txt_path))[0]
        output_plot_path = os.path.join(output_folder, f"{base_name}_scatter_plot.png")
        output_csv_path = os.path.join(output_folder, f"{base_name}_data.csv")

        # Ensure directory exists
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
            except OSError as e:
                feedback.pushWarning(f"Could not create folder {output_folder}: {e}")
        # ---------------------------------

        feedback.pushInfo(f"Step 1: Preparing data in {output_folder}...")
        if not predicted_raster:
            raise RuntimeError("Invalid input predicted raster.")
        if not validation_layer:
            raise RuntimeError("Invalid input validation points.")

        validation_crs = validation_layer.crs()
        raster_crs = predicted_raster.crs()

        field_names = [field.name() for field in validation_layer.fields()]

        with rasterio.open(predicted_raster.source()) as src:
            transform = src.transform
            raster_band = src.read(1).astype("float32")
            nodata_val = src.nodata

        feedback.pushInfo(
            f"Step 2: Extracting values for {validation_layer.featureCount()} points..."
        )

        data_list = []

        for feat in validation_layer.getFeatures():
            geom = feat.geometry()
            if geom is None:
                continue

            pt = geom.centroid().asPoint() if geom.isMultipart() else geom.asPoint()
            orig_x, orig_y = float(pt.x()), float(pt.y())

            try:
                val_true = float(feat[depth_field])
            except (KeyError, TypeError, ValueError):
                continue

            x_trans, y_trans = self._transform_point_if_needed(
                orig_x, orig_y, validation_crs, raster_crs, feedback
            )

            col, row_idx = ~transform * (x_trans, y_trans)
            r_i, c_i = int(row_idx), int(col)

            val_pred = np.nan
            if 0 <= r_i < raster_band.shape[0] and 0 <= c_i < raster_band.shape[1]:
                val_pred = raster_band[r_i, c_i]
                if nodata_val is not None and val_pred == nodata_val:
                    val_pred = np.nan

            if not np.isnan(val_pred):
                row_data = {f: feat[f] for f in field_names}
                row_data["X_Coordinate"] = orig_x
                row_data["Y_Coordinate"] = orig_y
                row_data["True_Depth_Unseen"] = val_true
                row_data["Predicted_Depth"] = float(val_pred)
                row_data["Error_Diff"] = float(val_pred) - val_true

                data_list.append(row_data)

        if not data_list:
            raise RuntimeError(
                "No valid overlapping points found between Vector and Raster."
            )

        df = pd.DataFrame(data_list)

        y_true = df["True_Depth_Unseen"].values
        y_pred = df["Predicted_Depth"].values

        feedback.pushInfo(f"Evaluated {len(df)} points successfully.")

        # --- Calculate Metrics ---
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        bias = np.mean(y_pred - y_true)

        # --- 1. Export CSV ---
        feedback.pushInfo(f"Saving Data CSV to {output_csv_path}...")
        df.to_csv(output_csv_path, index=False)

        # --- 2. Generate Plot ---
        feedback.pushInfo(f"Generating Scatter Plot to {output_plot_path}...")
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolors="b", label="Test Points")

        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            linewidth=2,
            label="1:1 Ideal",
        )

        plt.title(
            f"SDB Evaluation: Unseen Data (Test Set)\nR²={r2:.3f}, RMSE={rmse:.3f}"
        )
        plt.xlabel("True Depth (m)")
        plt.ylabel("Predicted Depth (m)")
        plt.legend()
        plt.grid(True)

        try:
            plt.savefig(output_plot_path, dpi=150)
        except Exception as e:
            feedback.pushWarning(f"Could not save plot: {e}")
        finally:
            plt.close()

        # --- 3. Generate Text Report ---
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write("====================================================\n")
            f.write(" SDB Model Evaluation Report - UNSEEN DATA (TEST SET)\n")
            f.write("====================================================\n\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Predicted Raster: {predicted_raster.name()}\n")
            f.write(f"Unseen Validation Vector: {validation_layer.name()}\n")
            f.write(f"Count of points: {len(df)}\n")
            f.write("\n--- Accuracy Metrics ---\n")
            f.write(f"R-squared (R2): {r2:.4f}\n")
            f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
            f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
            f.write(f"Bias (Mean Error): {bias:.4f}\n")
            f.write("\n--- Files Generated ---\n")
            f.write(f"CSV Data: {output_csv_path}\n")
            f.write(f"Scatter Plot: {output_plot_path}\n")

        feedback.pushInfo("Process complete.")

        return {self.OUTPUT_FILE: output_txt_path}
