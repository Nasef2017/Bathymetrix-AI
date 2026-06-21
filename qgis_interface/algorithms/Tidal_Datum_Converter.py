# -*- coding: utf-8 -*-

"""
***************************************************************************
*   Tidal Datum Converter                                                 *
*   -------------------------------------------------------------------   *
*   Simple tool to convert vertical levels between tidal datums           *
*   using FES2014 data.                                                   *
***************************************************************************
"""

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFile,
    QgsProcessingParameterField,
    QgsProcessingParameterEnum,
    QgsField,
    QgsFeature,
    QgsFeatureSink,
    QgsCoordinateTransform,
    QgsProject,
    QgsCoordinateReferenceSystem,
)
import os
import numpy as np
import netCDF4


class TidalDatumConverter(QgsProcessingAlgorithm):
    # Constants
    INPUT = "INPUT"
    HEIGHT_FIELD = "HEIGHT_FIELD"
    FES_FOLDER = "FES_FOLDER"
    SOURCE_DATUM = "SOURCE_DATUM"
    TARGET_DATUM = "TARGET_DATUM"
    OUTPUT = "OUTPUT"

    # Datum List
    DATUMS = [
        "HAT  (Highest Astronomical Tide)",  # 0
        "MHWS (Mean High Water Springs)",  # 1
        "MHWN (Mean High Water Neaps)",  # 2
        "MSL  (Mean Sea Level)",  # 3 (Default)
        "MLWN (Mean Low Water Neaps)",  # 4
        "MLWS (Mean Low Water Springs)",  # 5
        "MLLW (Mean Lower Low Water)",  # 6
        "LAT  (Lowest Astronomical Tide)",  # 7
    ]

    def tr(self, string):
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        return TidalDatumConverter()

    def name(self):
        return "tidal_datum_converter"

    def displayName(self):
        return self.tr("Tidal Datum Converter (FES2014)")

    def group(self):
        return ""

    def groupId(self):
        return ""

    def shortHelpString(self):
        return """
        <div style="font-family: Arial, sans-serif; line-height: 1.2;">
            <h2 style="margin-bottom: 5px;">🌊 <span style="color: #2E86C1;">Tidal-Mapper</span>: Vertical Datum Converter</h2>
            <p style="margin-top: 0; margin-bottom: 10px;">A precision tool to transform vertical levels (Height/Depth) between tidal datums using the FES2014 global hydrodynamic model.</p>

            <b style="display: block; margin-bottom: 2px;">⚙️ Workflow & Logic</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li><b>Extracts Constituents:</b> Reads M2, S2, K1, O1, etc., from NetCDF files.</li>
                <li><b>Calculates Offsets:</b> Derives datum separation relative to Mean Sea Level (MSL).</li>
                <li><b>Transformation Formula:</b><br>
                <i>Final = Input + (Offset<sub>Target</sub> - Offset<sub>Source</sub>)</i></li>
            </ul>

            <b style="display: block; margin-bottom: 2px;">📏 Supported Datums</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li><b>Astronomical:</b> HAT, LAT <i>(Max/Min tidal range)</i>.</li>
                <li><b>Mean Levels:</b> MSL, MHWS, MLWS, MHWN, MLWN.</li>
                <li><b>Hydrographic:</b> MLLW <i>(Common chart datum)</i>.</li>
            </ul>

            <b style="display: block; margin-bottom: 2px;">🧠 Smart Features</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li><b>Smart Coastal Fix:</b> Automatically estimates values for near-shore points using a search radius if exact pixel is null.</li>
                <li><b>Auto-Unit Conversion:</b> Handles FES2014 raw units (cm) to output meters.</li>
            </ul>

            <p style="margin-top: 10px; border-top: 1px solid #ccc; padding-top: 5px;">
                <b style="color: #E74C3C;">⚠️ Requirement:</b> FES2014 'ocean_tide' folder containing .nc files.
            </p>

            <p style="margin-top: 5px;">
                <b>Developer:</b> Mohamed Aly Nasef
            </p>
        </div>
        """

    def helpString(self):
        return self.shortHelpString()

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                self.tr("Input Point Layer"),
                [QgsProcessing.TypeVectorPoint],
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.HEIGHT_FIELD,
                self.tr("Height/Depth Column"),
                None,
                self.INPUT,
                QgsProcessingParameterField.Numeric,
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.SOURCE_DATUM,
                self.tr("Source Datum (Input)"),
                self.DATUMS,
                defaultValue=3,
            )
        )  # MSL
        self.addParameter(
            QgsProcessingParameterEnum(
                self.TARGET_DATUM,
                self.tr("Target Datum (Output)"),
                self.DATUMS,
                defaultValue=7,
            )
        )  # LAT

        self.addParameter(
            QgsProcessingParameterFile(
                self.FES_FOLDER,
                self.tr("Select FES2014 'ocean_tide' Folder"),
                behavior=QgsProcessingParameterFile.Folder,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(self.OUTPUT, self.tr("Transformed Layer"))
        )

    def processAlgorithm(self, parameters, context, feedback):
        # 1. Inputs
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if not source:
            raise QgsProcessingException("Invalid Input")

        height_field = self.parameterAsString(parameters, self.HEIGHT_FIELD, context)
        src_idx = self.parameterAsInt(parameters, self.SOURCE_DATUM, context)
        tgt_idx = self.parameterAsInt(parameters, self.TARGET_DATUM, context)
        fes_folder = self.parameterAsString(parameters, self.FES_FOLDER, context)

        src_name = self.DATUMS[src_idx].split(" ")[0]
        tgt_name = self.DATUMS[tgt_idx].split(" ")[0]

        # Log the General Formula
        feedback.pushInfo("\n" + "=" * 40)
        feedback.pushInfo(f"TRANSFORMATION: {src_name} -> {tgt_name}")
        feedback.pushInfo(
            "FORMULA: Final_Level = Input_Level + (Target_Offset - Source_Offset)"
        )
        feedback.pushInfo("Offsets are calculated relative to MSL (0)")
        feedback.pushInfo("=" * 40 + "\n")

        # 2. Output Setup
        fields = source.fields()
        fields.append(QgsField("Shift_Applied_m", QVariant.Double))
        fields.append(QgsField("Final_Level", QVariant.Double))

        sink, dest_id = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            fields,
            source.wkbType(),
            source.sourceCrs(),
        )
        if not sink:
            raise QgsProcessingException("Invalid Output Sink")

        # 3. Load FES2014
        feedback.pushInfo(f"Loading FES2014 data from: {fes_folder}")

        required_files = [
            "m2.nc",
            "s2.nc",
            "k1.nc",
            "o1.nc",
            "n2.nc",
            "p1.nc",
            "k2.nc",
            "q1.nc",
        ]
        grids = {}

        lat_grid = None
        lon_grid = None
        files_loaded = 0

        for fname in required_files:
            path = os.path.join(fes_folder, fname)
            if not os.path.exists(path):
                feedback.reportError(f"Warning: {fname} missing.", fatal=False)
                continue

            try:
                ds = netCDF4.Dataset(path, "r")
                if lat_grid is None:
                    lat_grid = (
                        ds.variables["lat"][:]
                        if "lat" in ds.variables
                        else ds.variables["latitude"][:]
                    )
                    lon_grid = (
                        ds.variables["lon"][:]
                        if "lon" in ds.variables
                        else ds.variables["longitude"][:]
                    )

                var_name = None
                if "amplitude" in ds.variables:
                    var_name = "amplitude"
                elif "amp" in ds.variables:
                    var_name = "amp"

                if var_name:
                    key = fname.split(".")[0]
                    grids[key] = ds.variables[var_name][:].filled(0)
                    files_loaded += 1

                ds.close()
            except Exception as e:
                feedback.reportError(f"Error reading {fname}: {e}", fatal=False)

        if files_loaded == 0:
            raise QgsProcessingException("No FES files loaded!")

        # 4. Coordinate Transform
        transform = QgsCoordinateTransform(
            source.sourceCrs(),
            QgsCoordinateReferenceSystem("EPSG:4326"),
            QgsProject.instance(),
        )

        # 5. Helper Functions
        max_r, max_c = len(lat_grid) - 1, len(lon_grid) - 1

        def get_amp_meters(comp_name, r, c):
            if comp_name not in grids:
                return 0.0
            val_cm = grids[comp_name][r, c]

            # Smart Coastal Fix
            if val_cm == 0:
                found = False
                for rad in range(1, 4):
                    rmin, rmax = max(0, r - rad), min(max_r, r + rad)
                    cmin, cmax = max(0, c - rad), min(max_c, c + rad)
                    win = grids[comp_name][rmin:rmax + 1, cmin:cmax + 1]
                    wet = win[win > 0]
                    if len(wet) > 0:
                        val_cm = np.mean(wet)
                        found = True
                        break
                if not found:
                    val_cm = 0.0

            # Divide by 100 for Meters
            return val_cm / 100.0

        # 6. Main Loop
        features = source.getFeatures()
        total = source.featureCount() if source.featureCount() > 0 else 1
        count = 0
        height_idx = source.fields().indexFromName(height_field)

        for feat in features:
            if feedback.isCanceled():
                break

            try:
                pt = transform.transform(feat.geometry().asPoint())
            except Exception:
                pt = feat.geometry().asPoint()

            x, y = pt.x(), pt.y()
            if x < 0:
                x += 360

            r = np.clip(np.searchsorted(lat_grid, y), 0, max_r)
            c = np.clip(np.searchsorted(lon_grid, x), 0, max_c)

            # Get Amplitudes
            m2 = get_amp_meters("m2", r, c)
            s2 = get_amp_meters("s2", r, c)
            k1 = get_amp_meters("k1", r, c)
            o1 = get_amp_meters("o1", r, c)

            sum_all = 0.0
            for k in grids.keys():
                sum_all += get_amp_meters(k, r, c)

            # Calculate Offsets
            offsets = {}
            offsets[3] = 0.0  # MSL
            offsets[0] = -sum_all  # HAT
            offsets[7] = +sum_all  # LAT
            offsets[1] = -(m2 + s2)  # MHWS
            offsets[5] = +(m2 + s2)  # MLWS
            offsets[2] = -abs(m2 - s2)  # MHWN
            offsets[4] = +abs(m2 - s2)  # MLWN
            offsets[6] = +(m2 + s2 + k1 + o1)  # MLLW

            src_off = offsets.get(src_idx, 0.0)
            tgt_off = offsets.get(tgt_idx, 0.0)

            # The Equation
            shift_val = tgt_off - src_off

            try:
                h_in = float(feat[height_idx] or 0.0)
            except Exception:
                h_in = 0.0

            final_z = h_in + shift_val

            # --- LOGGING EXAMPLE (First Feature Only) ---
            if count == 0:
                feedback.pushInfo(
                    f"--- Example Calculation (Feature ID: {feat.id()}) ---"
                )
                feedback.pushInfo(f"Input Level ({src_name}): {h_in:.3f} m")
                feedback.pushInfo(
                    f"Source Offset ({src_name}): {src_off:.4f} m (relative to MSL)"
                )
                feedback.pushInfo(
                    f"Target Offset ({tgt_name}): {tgt_off:.4f} m (relative to MSL)"
                )
                feedback.pushInfo(
                    f"Shift Applied: {tgt_off:.4f} - {src_off:.4f} = {shift_val:.4f} m"
                )
                feedback.pushInfo(
                    f"Result: {h_in:.3f} + ({shift_val:.4f}) = {final_z:.3f} m"
                )
                feedback.pushInfo("-" * 45)

            # Save
            new_feat = QgsFeature(fields)
            new_feat.setGeometry(feat.geometry())
            attrs = feat.attributes()
            attrs.append(float(shift_val))
            attrs.append(float(final_z))
            new_feat.setAttributes(attrs)

            sink.addFeature(new_feat, QgsFeatureSink.FastInsert)

            count += 1
            if count % 1000 == 0:
                feedback.setProgress(int(count / total * 100))

        return {self.OUTPUT: dest_id}
