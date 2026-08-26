# -*- coding: utf-8 -*-

"""
***************************************************************************
*   Tidal Datum Converter                                                 *
*   -------------------------------------------------------------------   *
*   High-precision hydrodynamic transformation tool to convert and        *
*   vertically align depth and elevation observations between tidal       *
*   datums using global tide models (GOT4.10c & FES2014).                *
***************************************************************************
"""

import os
import tarfile
import urllib.request
import numpy as np
import netCDF4

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
    QgsProcessingParameterBoolean,
    QgsField,
    QgsFeature,
    QgsFeatureSink,
    QgsCoordinateTransform,
    QgsProject,
    QgsCoordinateReferenceSystem,
    QgsApplication,
)


GOT410C_REMOTE_URL = "https://earth.gsfc.nasa.gov/sites/default/files/2023-12/got4.10c.tar.gz"
GOT410C_FALLBACK_URL = "https://earth.gsfc.nasa.gov/sites/default/files/2023-12/got4.10c.tar.gz"

GOT410C_CONSTITUENTS = [
    "m2.nc",
    "s2.nc",
    "k1.nc",
    "o1.nc",
    "n2.nc",
    "p1.nc",
    "k2.nc",
    "q1.nc",
    "s1.nc",
    "m4.nc",
]

FES_CONSTITUENTS = [
    "m2.nc",
    "s2.nc",
    "k1.nc",
    "o1.nc",
    "n2.nc",
    "p1.nc",
    "k2.nc",
    "q1.nc",
]


def get_default_got410c_dir():
    local_app_data = os.environ.get(
        "LOCALAPPDATA",
        os.path.join(os.path.expanduser("~"), "AppData", "Local"),
    )
    target_dir = os.path.join(local_app_data, "Bathymetrix_AI", "tide_models", "GOT4.10c")
    return os.path.normpath(target_dir)


class TidalDatumConverter(QgsProcessingAlgorithm):
    # Constants
    INPUT = "INPUT"
    HEIGHT_FIELD = "HEIGHT_FIELD"
    MODEL_ENGINE = "MODEL_ENGINE"
    SOURCE_DATUM = "SOURCE_DATUM"
    TARGET_DATUM = "TARGET_DATUM"
    FES_FOLDER = "FES_FOLDER"
    CUSTOM_MODEL_DIR = "CUSTOM_MODEL_DIR"
    AUTO_DOWNLOAD = "AUTO_DOWNLOAD"
    OUTPUT = "OUTPUT"

    # Model Engine Options
    MODEL_ENGINES = [
        "GOT4.10c (NASA GSFC - Automated / Offline)",  # 0
        "FES2014 (Local ocean_tide Folder)",          # 1
    ]

    # Datum List
    DATUMS = [
        "HAT  (Highest Astronomical Tide)",  # 0
        "MHWS (Mean High Water Springs)",    # 1
        "MHWN (Mean High Water Neaps)",      # 2
        "MSL  (Mean Sea Level)",             # 3 (Default)
        "MLWN (Mean Low Water Neaps)",       # 4
        "MLWS (Mean Low Water Springs)",     # 5
        "MLLW (Mean Lower Low Water)",       # 6
        "LAT  (Lowest Astronomical Tide)",   # 7
    ]

    def tr(self, string):
        return QCoreApplication.translate("Processing", string)

    def createInstance(self):
        return TidalDatumConverter()

    def name(self):
        return "tidal_datum_converter"

    def displayName(self):
        return self.tr("Tidal Datum Converter")

    def group(self):
        return ""

    def groupId(self):
        return ""

    def shortHelpString(self):
        return """
        <div style="font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.5; color: #2C3E50;">
            <h2 style="margin-bottom: 5px; color: #2E86C1;">🌊 Bathymetrix-AI: Tidal Datum Converter</h2>
            <p style="margin-top: 0; margin-bottom: 15px; font-size: 13px;">
                A precision hydrodynamic transformation tool to convert and vertically align depth and elevation observations between tidal datums using global hydrodynamic and empirical ocean tide models (<b>GOT4.10c</b> & <b>FES2014</b>).
            </p>

            <h3 style="color: #117A65; margin-bottom: 5px; border-bottom: 2px solid #117A65; padding-bottom: 3px;">⚙️ Supported Tide Model Engines</h3>
            <ul style="font-size: 12px; margin-top: 5px; padding-left: 20px;">
                <li><b>GOT4.10c (Default):</b> NASA GSFC Global Ocean Tide model. Lightweight (~15 MB) with automatic download and caching.</li>
                <li><b>FES2014:</b> High-resolution Finite Element Solution hydrodynamic tidal atlas for coastal areas (requires local <code>ocean_tide</code> directory).</li>
            </ul>

            <h3 style="color: #D35400; margin-top: 15px; margin-bottom: 5px; border-bottom: 2px solid #D35400; padding-bottom: 3px;">📏 Supported Tidal Datums</h3>
            <ul style="font-size: 12px; margin-top: 5px; padding-left: 20px;">
                <li><b>Astronomical Extremes:</b> Highest Astronomical Tide (HAT), Lowest Astronomical Tide (LAT).</li>
                <li><b>Mean Tidal Levels:</b> Mean Sea Level (MSL), Mean High Water Springs (MHWS), Mean Low Water Springs (MLWS), Mean High Water Neaps (MHWN), Mean Low Water Neaps (MLWN).</li>
                <li><b>Hydrographic Chart Datums:</b> Mean Lower Low Water (MLLW).</li>
            </ul>

            <h3 style="color: #8E44AD; margin-top: 15px; margin-bottom: 5px; border-bottom: 2px solid #8E44AD; padding-bottom: 3px;">🧠 Smart Features</h3>
            <ul style="font-size: 12px; margin-top: 5px; padding-left: 20px;">
                <li><b>Transformation Formula:</b> <code>Z_Target = Z_Source + (Offset_Target - Offset_Source)</code></li>
                <li><b>Smart Coastal Fix:</b> Automatically interpolates and extrapolates nearshore points using a spatial search radius if an exact coastal grid cell is null or masked.</li>
                <li><b>Automatic Unit Normalization:</b> Automatically standardizes constituent amplitude units (cm to meters).</li>
            </ul>

            <br><b>Developer:</b> Mohamed Aly Nasef
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
                self.MODEL_ENGINE,
                self.tr("Tidal Model Engine"),
                self.MODEL_ENGINES,
                defaultValue=0,
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

        # Optional folder inputs depending on model choice
        self.addParameter(
            QgsProcessingParameterFile(
                self.FES_FOLDER,
                self.tr("Select FES2014 'ocean_tide' Folder (Required for FES2014)"),
                behavior=QgsProcessingParameterFile.Folder,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFile(
                self.CUSTOM_MODEL_DIR,
                self.tr("Custom GOT4.10c Directory (Optional)"),
                behavior=QgsProcessingParameterFile.Folder,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.AUTO_DOWNLOAD,
                self.tr("Auto-download GOT4.10c if missing?"),
                defaultValue=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterFeatureSink(self.OUTPUT, self.tr("Transformed Layer"))
        )

    def _ensure_got410c_model(self, target_dir, auto_download, feedback):
        """
        Verifies that GOT4.10c constituent files exist in target_dir.
        If missing and auto_download is True, downloads and extracts them.
        """
        os.makedirs(target_dir, exist_ok=True)
        
        # Check existing files
        present_files = [
            f for f in GOT410C_CONSTITUENTS
            if os.path.exists(os.path.join(target_dir, f)) and os.path.getsize(os.path.join(target_dir, f)) > 0
        ]
        
        if len(present_files) >= len(GOT410C_CONSTITUENTS) - 2:
            feedback.pushInfo(f"✔ Found valid GOT4.10c model files in: {target_dir}")
            return target_dir

        if not auto_download:
            raise QgsProcessingException(
                f"GOT4.10c model files are missing in: {target_dir}. "
                f"Please enable 'Auto-download GOT4.10c' or specify a valid directory."
            )

        feedback.pushInfo(f"Downloading GOT4.10c archive from NASA GSFC (~44 MB) to: {target_dir}...")
        archive_path = os.path.join(target_dir, "got410c_netcdf.tar.gz")

        download_success = False
        for url in [GOT410C_REMOTE_URL, GOT410C_FALLBACK_URL]:
            try:
                feedback.pushInfo(f"Connecting to: {url}")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
                with urllib.request.urlopen(req, timeout=90) as response, open(archive_path, "wb") as out_file:
                    total_length = response.getheader("content-length")
                    total_bytes = int(total_length) if total_length else 44325003
                    downloaded = 0
                    block_size = 131072
                    last_pct = -1
                    while True:
                        if feedback.isCanceled():
                            return target_dir
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        downloaded += len(buffer)
                        out_file.write(buffer)
                        pct = int((downloaded / total_bytes) * 50)
                        if pct != last_pct:
                            feedback.setProgress(pct)
                            last_pct = pct
                download_success = True
                break
            except Exception as dl_err:
                feedback.pushWarning(f"Download attempt from {url} failed: {dl_err}")

        if not download_success or not os.path.exists(archive_path) or os.path.getsize(archive_path) < 10000:
            # Check if any constituent files were placed manually
            present_after = [
                f for f in GOT410C_CONSTITUENTS
                if os.path.exists(os.path.join(target_dir, f)) and os.path.getsize(os.path.join(target_dir, f)) > 0
            ]
            if len(present_after) >= 4:
                return target_dir
            raise QgsProcessingException(
                f"Failed to automatically download GOT4.10c model files. "
                f"Please check your internet connection, or manually place the GOT4.10c .nc files into: {target_dir}"
            )

        feedback.pushInfo("Extracting GOT4.10c constituent archive...")
        try:
            with tarfile.open(archive_path, "r:*") as tar:
                for member in tar.getmembers():
                    # Flatten filename
                    member_name = os.path.basename(member.name).lower()
                    if member_name.endswith(".nc"):
                        source_f = tar.extractfile(member)
                        if source_f:
                            with open(os.path.join(target_dir, member_name), "wb") as target_f:
                                target_f.write(source_f.read())
            feedback.pushInfo("✔ GOT4.10c extraction completed successfully.")
        except Exception as ex_err:
            feedback.pushWarning(f"Archive extraction warning: {ex_err}")

        try:
            if os.path.exists(archive_path):
                os.remove(archive_path)
        except Exception:
            pass

        return target_dir

    def processAlgorithm(self, parameters, context, feedback):
        # 1. Inputs
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if not source:
            raise QgsProcessingException("Invalid Input Point Layer.")

        height_field = self.parameterAsString(parameters, self.HEIGHT_FIELD, context)
        model_engine_idx = self.parameterAsInt(parameters, self.MODEL_ENGINE, context)
        src_idx = self.parameterAsInt(parameters, self.SOURCE_DATUM, context)
        tgt_idx = self.parameterAsInt(parameters, self.TARGET_DATUM, context)
        fes_folder = self.parameterAsString(parameters, self.FES_FOLDER, context)
        custom_model_dir = self.parameterAsString(parameters, self.CUSTOM_MODEL_DIR, context)
        auto_download = self.parameterAsBoolean(parameters, self.AUTO_DOWNLOAD, context)

        src_name = self.DATUMS[src_idx].split(" ")[0]
        tgt_name = self.DATUMS[tgt_idx].split(" ")[0]
        model_name = "GOT4.10c" if model_engine_idx == 0 else "FES2014"

        # Log Header
        feedback.pushInfo("\n" + "=" * 55)
        feedback.pushInfo(f"🌊 TIDAL DATUM CONVERTER: {src_name} -> {tgt_name}")
        feedback.pushInfo(f"MODEL ENGINE: {model_name}")
        feedback.pushInfo("FORMULA: Final_Level = Input_Level + (Target_Offset - Source_Offset)")
        feedback.pushInfo("DATUM OFFSETS: Evaluated relative to Mean Sea Level (MSL = 0.0m)")
        feedback.pushInfo("=" * 55 + "\n")

        # 2. Determine Model Directory
        if model_engine_idx == 0:
            # GOT4.10c Mode
            model_dir = custom_model_dir.strip() if custom_model_dir and custom_model_dir.strip() else get_default_got410c_dir()
            model_dir = self._ensure_got410c_model(model_dir, auto_download, feedback)
            target_constituents = GOT410C_CONSTITUENTS
        else:
            # FES2014 Mode
            if not fes_folder or not os.path.exists(fes_folder):
                raise QgsProcessingException(
                    "Please select a valid FES2014 'ocean_tide' folder containing NetCDF constituent files (.nc)."
                )
            model_dir = fes_folder
            target_constituents = FES_CONSTITUENTS

        # 3. Load Constituent Grids via NetCDF4
        feedback.pushInfo(f"Loading {model_name} constituent grids from: {model_dir}")
        grids = {}
        lat_grid = None
        lon_grid = None
        files_loaded = 0
        grid_is_cm = False

        for fname in target_constituents:
            # Check direct and case-insensitive filename matching
            path = os.path.join(model_dir, fname)
            if not os.path.exists(path):
                # Search case-insensitively
                found_match = None
                for candidate in os.listdir(model_dir):
                    if candidate.lower() == fname.lower():
                        found_match = os.path.join(model_dir, candidate)
                        break
                if found_match:
                    path = found_match
                else:
                    feedback.pushWarning(f"Constituent file '{fname}' not found in model directory.")
                    continue

            try:
                ds = netCDF4.Dataset(path, "r")
                if lat_grid is None:
                    if "lat" in ds.variables:
                        lat_grid = ds.variables["lat"][:]
                    elif "latitude" in ds.variables:
                        lat_grid = ds.variables["latitude"][:]

                    if "lon" in ds.variables:
                        lon_grid = ds.variables["lon"][:]
                    elif "longitude" in ds.variables:
                        lon_grid = ds.variables["longitude"][:]

                var_name = None
                for v_candidate in ["amplitude", "amp", "z", "z0", "h"]:
                    if v_candidate in ds.variables:
                        var_name = v_candidate
                        break

                if var_name:
                    key = fname.split(".")[0].lower()
                    raw_arr = ds.variables[var_name][:]
                    if np.ma.isMaskedArray(raw_arr):
                        raw_arr = raw_arr.filled(0.0)
                    else:
                        raw_arr = np.nan_to_num(raw_arr, nan=0.0)

                    # Determine if unit is cm (FES2014 is in cm, GOT4.10c can be cm)
                    var_obj = ds.variables[var_name]
                    units_attr = getattr(var_obj, "units", "").lower()
                    if "cm" in units_attr or np.nanmax(raw_arr) > 15.0:
                        grid_is_cm = True

                    grids[key] = np.asarray(raw_arr, dtype=float)
                    files_loaded += 1

                ds.close()
            except Exception as e:
                feedback.reportError(f"Error reading {fname}: {e}", fatal=False)

        if files_loaded == 0 or lat_grid is None or lon_grid is None:
            raise QgsProcessingException(
                f"No valid constituent grids could be loaded from: {model_dir}. "
                f"Please verify that the directory contains valid NetCDF tidal constituent files."
            )

        feedback.pushInfo(f"✔ Successfully loaded {files_loaded} tidal constituent grids ({', '.join(grids.keys())}).")
        unit_info = "cm (converted to meters)" if grid_is_cm else "meters"
        feedback.pushInfo(f"✔ Constituent grid unit: {unit_info}")

        # 4. Output Setup
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
            raise QgsProcessingException("Invalid Output Feature Sink.")

        # 5. Coordinate Transformation Setup
        transform = QgsCoordinateTransform(
            source.sourceCrs(),
            QgsCoordinateReferenceSystem("EPSG:4326"),
            QgsProject.instance(),
        )

        max_r = len(lat_grid) - 1
        max_c = len(lon_grid) - 1

        def get_amp_meters(comp_name, r_idx, c_idx):
            if comp_name not in grids:
                return 0.0
            val = grids[comp_name][r_idx, c_idx]

            # Smart Coastal Fix (spatial search radius for coastal wet cells)
            if val <= 0.0 or not np.isfinite(val):
                found = False
                for rad in range(1, 5):
                    rmin, rmax = max(0, r_idx - rad), min(max_r, r_idx + rad)
                    cmin, cmax = max(0, c_idx - rad), min(max_c, c_idx + rad)
                    win = grids[comp_name][rmin:rmax + 1, cmin:cmax + 1]
                    wet = win[(win > 0.0) & np.isfinite(win)]
                    if len(wet) > 0:
                        val = float(np.mean(wet))
                        found = True
                        break
                if not found:
                    val = 0.0

            if grid_is_cm:
                return float(val) / 100.0
            return float(val)

        # 6. Process Features
        features = source.getFeatures()
        total = source.featureCount() if source.featureCount() > 0 else 1
        count = 0
        height_idx = source.fields().indexFromName(height_field)

        for feat in features:
            if feedback.isCanceled():
                break

            try:
                geom = feat.geometry()
                if geom is None or geom.isEmpty():
                    continue
                pt = transform.transform(geom.asPoint())
            except Exception:
                pt = feat.geometry().asPoint()

            x, y = pt.x(), pt.y()

            # Handle 0..360 vs -180..180 longitude coordinate systems
            if lon_grid[0] >= 0 and lon_grid[-1] > 180 and x < 0:
                x_query = x + 360.0
            elif lon_grid[0] < 0 and x > 180:
                x_query = x - 360.0
            else:
                x_query = x

            if lat_grid[-1] < lat_grid[0]:
                r = np.clip(np.searchsorted(-lat_grid, -y), 0, max_r)
            else:
                r = np.clip(np.searchsorted(lat_grid, y), 0, max_r)

            if lon_grid[-1] < lon_grid[0]:
                c = np.clip(np.searchsorted(-lon_grid, -x_query), 0, max_c)
            else:
                c = np.clip(np.searchsorted(lon_grid, x_query), 0, max_c)

            # Extract amplitudes in meters
            m2 = get_amp_meters("m2", r, c)
            s2 = get_amp_meters("s2", r, c)
            k1 = get_amp_meters("k1", r, c)
            o1 = get_amp_meters("o1", r, c)

            sum_all = 0.0
            for k in grids.keys():
                sum_all += get_amp_meters(k, r, c)

            # Calculate Offsets relative to Mean Sea Level (MSL = 0)
            offsets = {}
            offsets[3] = 0.0                            # MSL
            offsets[0] = -sum_all                       # HAT (Highest Astronomical Tide)
            offsets[7] = +sum_all                       # LAT (Lowest Astronomical Tide)
            offsets[1] = -(m2 + s2)                     # MHWS (Mean High Water Springs)
            offsets[5] = +(m2 + s2)                     # MLWS (Mean Low Water Springs)
            offsets[2] = -abs(m2 - s2)                  # MHWN (Mean High Water Neaps)
            offsets[4] = +abs(m2 - s2)                  # MLWN (Mean Low Water Neaps)
            offsets[6] = +(m2 + s2 + k1 + o1)           # MLLW (Mean Lower Low Water)

            src_off = offsets.get(src_idx, 0.0)
            tgt_off = offsets.get(tgt_idx, 0.0)

            # Apply vertical transformation
            shift_val = float(tgt_off - src_off)

            try:
                h_in = float(feat[height_idx] if feat[height_idx] is not None else 0.0)
            except Exception:
                h_in = 0.0

            final_z = float(h_in + shift_val)

            # Detailed log for the first feature
            if count == 0:
                feedback.pushInfo(f"--- Example Transformation (Feature ID: {feat.id()}) ---")
                feedback.pushInfo(f"Coordinates: Lon={pt.x():.5f}, Lat={pt.y():.5f}")
                feedback.pushInfo(f"Extracted Harmonics: M2={m2:.3f}m, S2={s2:.3f}m, K1={k1:.3f}m, O1={o1:.3f}m")
                feedback.pushInfo(f"Input Level ({src_name}): {h_in:.3f} m")
                feedback.pushInfo(f"Source Offset ({src_name}): {src_off:.4f} m (rel. to MSL)")
                feedback.pushInfo(f"Target Offset ({tgt_name}): {tgt_off:.4f} m (rel. to MSL)")
                feedback.pushInfo(f"Shift Applied: {tgt_off:.4f} - ({src_off:.4f}) = {shift_val:+.4f} m")
                feedback.pushInfo(f"Final Level ({tgt_name}): {h_in:.3f} + ({shift_val:+.4f}) = {final_z:.3f} m")
                feedback.pushInfo("-" * 55)

            new_feat = QgsFeature(fields)
            new_feat.setGeometry(feat.geometry())
            attrs = feat.attributes()
            attrs.append(shift_val)
            attrs.append(final_z)
            new_feat.setAttributes(attrs)

            sink.addFeature(new_feat, QgsFeatureSink.FastInsert)
            count += 1

            if count % 2000 == 0:
                feedback.setProgress(int(count / total * 100))

        feedback.pushInfo(f"\n✔ Successfully transformed {count} features to datum: {tgt_name}")
        return {self.OUTPUT: dest_id}
