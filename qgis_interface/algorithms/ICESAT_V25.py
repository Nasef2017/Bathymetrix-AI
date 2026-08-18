# -*- coding: utf-8 -*-

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (QgsProcessing,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterVectorDestination,
                       QgsProcessingParameterEnum,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterString,
                       QgsProcessingParameterBoolean,
                       QgsCoordinateReferenceSystem,
                       QgsCoordinateTransform,
                       QgsFeature,
                       QgsGeometry,
                       QgsPointXY,
                       QgsFields,
                       QgsField,
                       QgsWkbTypes,
                       QgsProcessingException)
import sys
import pandas as pd
import numpy as np
import os
import tempfile

# Attempt to import external libraries
try:
    from sliderule import sliderule as sr
except ImportError:
    pass


class SlideRuleFinalTool(QgsProcessingAlgorithm):
    # Constants
    INPUT = 'INPUT'
    OUTPUT = 'OUTPUT'
    SCENARIO = 'SCENARIO'
    RGT = 'RGT'
    CYCLE = 'CYCLE'
    BEAM = 'BEAM'
    TIME_START = 'TIME_START'
    TIME_END = 'TIME_END'
    USE_CONF = 'USE_CONF'
    CONF_THRESHOLD = 'CONF_THRESHOLD'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return SlideRuleFinalTool()

    def name(self):
        return 'sliderule_icesat2_downloader'

    def displayName(self):
        return self.tr('1. ICESat-2 Data Downloader')

    def group(self):
        return self.tr('')

    def groupId(self):
        return ''

    def shortHelpString(self):
        return """
        <div style="font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.5; color: #2C3E50;">
            <h2 style="margin-bottom: 5px; color: #2E86C1;">🛰️ ICESat-2 Data Downloader</h2>
            <p style="margin-top: 0; margin-bottom: 15px; font-size: 13px;">
                A specialized interface to NASA's <b>SlideRule servers</b> for high-performance, on-demand querying, processing, and downloading of ICESat-2 photon altimetry and elevation products for integration into the Bathymetrix-AI pipelines.
            </p>

            <h3 style="color: #D35400; margin-bottom: 5px; border-bottom: 2px solid #D35400; padding-bottom: 3px;">📦 Supported Data Scenarios</h3>
            <ul style="font-size: 12px; margin-top: 5px; padding-left: 20px;">
                <li><b>ATL24 (Bathymetry):</b> Specialized geolocated bathymetry photon product with refraction-corrected sea floor depths.</li>
                <li><b>ATL03 (Raw Photons):</b> Full-rate geolocated photon events (with optional <b>YAPC</b> noise cleaning).</li>
                <li><b>ATL06 (Land Ice Elevation):</b> Surface height product for terrestrial/ice elevation baselines.</li>
            </ul>

            <h3 style="color: #117A65; margin-top: 15px; margin-bottom: 5px; border-bottom: 2px solid #117A65; padding-bottom: 3px;">🔍 Spatiotemporal & Track Filters</h3>
            <ul style="font-size: 12px; margin-top: 5px; padding-left: 20px;">
                <li><b>AOI Extent:</b> Spatially bounded by an input vector polygon (Shapefile / GeoPackage).</li>
                <li><b>Temporal Range:</b> Start and End date filters for targeted satellite passes.</li>
                <li><b>Orbital Filtering:</b> Target specific Reference Ground Tracks (RGT), Cycles, and Beams (gt1l, gt1r, gt2l, gt2r, gt3l, gt3r).</li>
            </ul>

            <p style="margin-top: 15px; background: #FDEDEC; border-left: 4px solid #E74C3C; padding: 8px 12px; font-size: 12px; color: #78281F;">
                <b>⚠️ Requirements:</b> Requires active internet connection and the <code>sliderule</code> and <code>geopandas</code> Python libraries.
            </p>
            <br><b>Developer:</b> Mohamed Aly Nasef
        </div>
        """

    def helpString(self):
        return self.shortHelpString()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(self.INPUT, self.tr('Input AOI (Shapefile/Polygon)'), [QgsProcessing.TypeVectorPolygon]))

        scenarios = [
            "1. Quick Access (All Data)",
            "2. Single Track (Requires RGT)",
            "3. Detailed Track (Classified)",
            "4. ATL03 Photons (Raw Data)",
            "5. ATL03 + YAPC (Cleaned Raw Data)",
            "6. ATL06 Surface Height",
            "7. ATL24 Bathymetry (Filtered)"
        ]
        self.addParameter(QgsProcessingParameterEnum(self.SCENARIO, self.tr('Scenario / Data Product'), options=scenarios, defaultValue=6))

        # Filters
        self.addParameter(QgsProcessingParameterBoolean(self.USE_CONF, self.tr('Apply Confidence Filter?'), defaultValue=False))
        self.addParameter(QgsProcessingParameterNumber(self.CONF_THRESHOLD, self.tr('Confidence Threshold'), type=QgsProcessingParameterNumber.Double, defaultValue=0.9, optional=True))

        # Time & Track
        self.addParameter(QgsProcessingParameterString(self.TIME_START, self.tr('Start Time (YYYY-MM-DD)'), defaultValue='2019-10-01', optional=False))
        self.addParameter(QgsProcessingParameterString(self.TIME_END, self.tr('End Time (YYYY-MM-DD)'), defaultValue='2019-11-01', optional=False))
        self.addParameter(QgsProcessingParameterNumber(self.RGT, self.tr('RGT'), type=QgsProcessingParameterNumber.Integer, optional=True))
        self.addParameter(QgsProcessingParameterNumber(self.CYCLE, self.tr('Cycle'), type=QgsProcessingParameterNumber.Integer, optional=True))
        self.addParameter(QgsProcessingParameterString(self.BEAM, self.tr('Beam'), optional=True))

        self.addParameter(QgsProcessingParameterVectorDestination(self.OUTPUT, self.tr('Output Layer')))

    def processAlgorithm(self, parameters, context, feedback):
        if 'sliderule' not in sys.modules:
            raise QgsProcessingException("The 'sliderule' library is not installed.")

        # Capture all feedback log messages in a list for writing to a text file later
        log_messages = []
        original_pushInfo = feedback.pushInfo
        
        # If feedback.pushInfo is a Mock, use side_effect to keep the Mock tracking intact
        from unittest.mock import Mock
        if isinstance(original_pushInfo, Mock):
            def mock_push_side_effect(msg):
                log_messages.append(msg)
            original_pushInfo.side_effect = mock_push_side_effect
        else:
            def custom_pushInfo(msg):
                log_messages.append(msg)
                original_pushInfo(msg)
            feedback.pushInfo = custom_pushInfo

        dest_id = None

        # Derive a fixed, flat temp directory using environment variables,
        # NOT tempfile.gettempdir(), which returns the previously-set tempfile.tempdir
        # and causes infinite recursive nesting on subsequent runs.
        tempfile.tempdir = None  # Reset any previously set global override first
        _base_temp = (
            os.environ.get('USERPROFILE') or
            os.path.expanduser('~') or
            'C:\\Temp'
        )
        safe_temp_dir = os.path.join(_base_temp, "AppData", "Local", "Temp", "sliderule_qgis_temp")
        os.makedirs(safe_temp_dir, exist_ok=True)

        def save_log():
            try:
                log_filepath = None
                output_param = parameters.get(self.OUTPUT)
                
                # Check if user provided an explicit file path string
                if isinstance(output_param, str) and output_param != 'TEMPORARY_OUTPUT' and output_param.strip():
                    clean_str = output_param.split('|')[0]
                    if os.path.dirname(clean_str):
                        base, _ = os.path.splitext(clean_str)
                        log_filepath = base + "_log.txt"
                
                # Fallback to dest_id from parameterAsSink
                if not log_filepath:
                    nonlocal dest_id
                    if isinstance(dest_id, str) and dest_id.strip():
                        clean_path = dest_id.split('|')[0]
                        if os.path.dirname(clean_path):
                            base, _ = os.path.splitext(clean_path)
                            log_filepath = base + "_log.txt"
                
                # Final fallback to safe temp dir
                if not log_filepath:
                    log_filepath = os.path.join(safe_temp_dir, "icesat2_downloader_log.txt")

                with open(log_filepath, "w", encoding="utf-8") as log_f:
                    log_f.write("\n".join(log_messages))
                original_pushInfo(f"Full process log saved to: {log_filepath}")
            except Exception as log_err:
                original_pushInfo(f"Warning: Could not save process log file: {log_err}")

        # Force GDAL/Arrow temp writes to this fixed path
        os.environ['CPL_TMPDIR'] = safe_temp_dir
        # ===================================================================

        # Initialize SlideRule Session with rethrow=True to catch real errors
        submodule = sys.modules.get('sliderule.sliderule')
        package = sys.modules.get('sliderule')

        def set_session(session_obj):
            if submodule:
                submodule.slideruleSession = session_obj
            if package:
                package.slideruleSession = session_obj

        try:
            session_obj = sr.Session(
                domain="slideruleearth.io",
                cluster="sliderule",
                verbose=False,
                rethrow=True,
                ssl_verify=True
            )
            set_session(session_obj)
            # Try to verify connection
            sr.check_version()
        except Exception as e:
            err_msg = str(e)
            if "SSL" in err_msg or "certificate" in err_msg or "verify" in err_msg:
                feedback.pushInfo("SSL certificate verification failed. Retrying SlideRule connection without SSL verification...")
                try:
                    session_obj = sr.Session(
                        domain="slideruleearth.io",
                        cluster="sliderule",
                        verbose=False,
                        rethrow=True,
                        ssl_verify=False
                    )
                    set_session(session_obj)
                except Exception as retry_err:
                    save_log()
                    raise QgsProcessingException(f"SlideRule initialization failed: {retry_err}")
            else:
                feedback.pushInfo(f"SlideRule initialization warning: {e}")

        # --- Inputs ---
        source = self.parameterAsSource(parameters, self.INPUT, context)
        dest_crs = QgsCoordinateReferenceSystem("EPSG:4326")
        transform = QgsCoordinateTransform(source.sourceCrs(), dest_crs, context.project())
        extent_wgs84 = transform.transformBoundingBox(source.sourceExtent())

        aoi = [
            {"lon": extent_wgs84.xMinimum(), "lat": extent_wgs84.yMinimum()},
            {"lon": extent_wgs84.xMaximum(), "lat": extent_wgs84.yMinimum()},
            {"lon": extent_wgs84.xMaximum(), "lat": extent_wgs84.yMaximum()},
            {"lon": extent_wgs84.xMinimum(), "lat": extent_wgs84.yMaximum()},
            {"lon": extent_wgs84.xMinimum(), "lat": extent_wgs84.yMinimum()}
        ]

        # Params
        parms = {"poly": aoi}
        t0 = self.parameterAsString(parameters, self.TIME_START, context)
        t1 = self.parameterAsString(parameters, self.TIME_END, context)

        if not t0 or not str(t0).strip():
            save_log()
            raise QgsProcessingException("Start Time (YYYY-MM-DD) is required. Please specify a valid start date.")

        if not t1 or not str(t1).strip():
            save_log()
            raise QgsProcessingException("End Time (YYYY-MM-DD) is required. Please specify a valid end date.")

        now = pd.Timestamp.now(tz='UTC')

        try:
            parsed_t0 = pd.to_datetime(str(t0).strip(), utc=True)
        except Exception as e:
            save_log()
            raise QgsProcessingException(f"Invalid Start Time '{t0}'. Expected format: YYYY-MM-DD. Error: {e}")

        try:
            parsed_t1 = pd.to_datetime(str(t1).strip(), utc=True)
        except Exception as e:
            save_log()
            raise QgsProcessingException(f"Invalid End Time '{t1}'. Expected format: YYYY-MM-DD. Error: {e}")

        # If user entered a date without specific time (e.g. YYYY-MM-DD at 00:00:00), adjust parsed_t1 to end of day to make the range inclusive
        if parsed_t1.hour == 0 and parsed_t1.minute == 0 and parsed_t1.second == 0 and parsed_t1.microsecond == 0:
            parsed_t1 = parsed_t1.normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

        if parsed_t0 > parsed_t1:
            save_log()
            raise QgsProcessingException(f"Start Time ({parsed_t0.strftime('%Y-%m-%d')}) cannot be after End Time ({parsed_t1.strftime('%Y-%m-%d')}).")

        if parsed_t1 > now:
            feedback.pushInfo(f"End date '{t1}' is in the future. Capping end date to current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
            parsed_t1 = now

        rgt = self.parameterAsInt(parameters, self.RGT, context)
        if rgt > 0:
            parms["rgt"] = rgt

        cycle = self.parameterAsInt(parameters, self.CYCLE, context)
        if cycle > 0:
            parms["cycle"] = cycle

        beam = self.parameterAsString(parameters, self.BEAM, context)
        if beam:
            parms["beams"] = beam

        scenario = self.parameterAsEnum(parameters, self.SCENARIO, context)
        use_conf = self.parameterAsBool(parameters, self.USE_CONF, context)

        # Endpoint & Parameter Mapping Logic based on Scenario
        api_endpoint = "atl24x"  # Default is ATL24 Bathymetry

        if scenario == 0:  # 1. Quick Access (All Data)
            api_endpoint = "atl24x"

        elif scenario == 1:  # 2. Single Track (Requires RGT)
            api_endpoint = "atl24x"

        elif scenario == 2:  # 3. Detailed Track (Classified)
            api_endpoint = "atl24x"
            parms.setdefault("atl24", {})["class_ph"] = ["bathymetry", "sea_surface"]

        elif scenario == 3:  # 4. ATL03 Photons (Raw Data)
            api_endpoint = "atl03x"
            # Signal confidence: -1 = all photons (no filter), 4 = high confidence
            parms["cnf"] = -1    # include all photons
            parms["srt"] = 0     # surface reference type = land
            # No atl24 keys — they are bathymetry-specific and invalid for atl03x

        elif scenario == 4:  # 5. ATL03 + YAPC (Cleaned Raw Data)
            api_endpoint = "atl03x"
            parms["cnf"] = -1
            parms["srt"] = 0
            # Enable YAPC photon classification
            parms["yapc"] = {
                "score": 0,   # include all YAPC scores (filtering done post-download)
                "knn": 0      # use default knn
            }
            # No atl24 keys — they are bathymetry-specific and invalid for atl03x

        elif scenario == 5:  # 6. ATL06 Surface Height
            api_endpoint = "atl06x"
            parms["srt"] = 3     # 3 = land surface (icesat2.SRT_LAND = 3)
            parms["len"] = 40    # segment length in metres (standard ATL06 = 40 m)
            parms["res"] = 20    # step size in metres
            # No atl24 or atl03 keys — ATL06 has its own parameter schema

        elif scenario == 6:  # 7. ATL24 Bathymetry (Filtered)
            api_endpoint = "atl24x"

        if use_conf:
            if "atl24" in api_endpoint:
                parms.setdefault("atl24", {})["confidence_threshold"] = self.parameterAsDouble(parameters, self.CONF_THRESHOLD, context)
                if "class_ph" not in parms["atl24"]:
                    parms["atl24"]["class_ph"] = ["bathymetry", "sea_surface", "unclassified"]

        # --- Generate Year-by-Year Date Chunks ---
        chunks = []
        current_start = parsed_t0
        while current_start <= parsed_t1:
            year = current_start.year
            current_end = pd.to_datetime(f"{year}-12-31T23:59:59Z", utc=True)
            if current_end > parsed_t1:
                current_end = parsed_t1
            chunks.append((current_start, current_end))
            current_start = pd.to_datetime(f"{year + 1}-01-01T00:00:00Z", utc=True)

        # --- Request Data Chunk-by-Chunk ---
        gdfs_list = []
        feedback.pushInfo(f"Divided query into {len(chunks)} annual processing chunks.")

        try:
            from sliderule import earthdata
        except ImportError:
            earthdata = None

        short_name = "ATL24" if "atl24" in api_endpoint else ("ATL06" if "atl06" in api_endpoint else "ATL03")

        for idx, (start_chunk, end_chunk) in enumerate(chunks):
            if feedback.isCanceled():
                break

            chunk_t0_str = start_chunk.strftime("%Y-%m-%dT%H:%M:%SZ")
            chunk_t1_str = end_chunk.strftime("%Y-%m-%dT%H:%M:%SZ")

            # 1. Quick check using NASA CMR metadata search
            if earthdata:
                feedback.pushInfo(f"[{idx+1}/{len(chunks)}] Checking NASA CMR for data in range: {start_chunk.strftime('%Y-%m-%d')} to {end_chunk.strftime('%Y-%m-%d')}...")
                try:
                    granules = earthdata.cmr(short_name=short_name, polygon=aoi, time_start=chunk_t0_str, time_end=chunk_t1_str)
                    if not granules:
                        feedback.pushInfo(f" -> No data found in this year chunk. Skipping.")
                        continue
                    feedback.pushInfo(f" -> Found {len(granules)} granules. Initiating SlideRule cloud processing...")
                except Exception as cmr_err:
                    feedback.pushInfo(f" -> NASA CMR check failed: {cmr_err}. Querying SlideRule directly.")

            # 2. Query SlideRule for this chunk
            chunk_parms = parms.copy()
            chunk_parms["t0"] = chunk_t0_str
            chunk_parms["t1"] = chunk_t1_str

            # Generate a unique temporary file path for this chunk to prevent SlideRule file table collisions
            chunk_output_path = os.path.join(safe_temp_dir, f"chunk_{idx}_{start_chunk.strftime('%Y%m%d')}.parquet")
            if os.path.exists(chunk_output_path):
                try:
                    os.remove(chunk_output_path)
                except Exception:  # nosec B110
                    pass
            chunk_parms["output"] = {
                "path": chunk_output_path,
                "format": "geoparquet",
                "open_on_complete": False
            }

            feedback.pushInfo(f"[{idx+1}/{len(chunks)}] Requesting {api_endpoint} from SlideRule ({start_chunk.strftime('%Y-%m-%d')} to {end_chunk.strftime('%Y-%m-%d')})...")
            try:
                gdf_chunk_path = sr.run(api_endpoint, chunk_parms)
                if gdf_chunk_path and os.path.exists(gdf_chunk_path):
                    import pyarrow.parquet as pq
                    import shapely
                    table = pq.read_table(gdf_chunk_path)
                    if table.num_rows == 0:
                        feedback.pushInfo(f" -> SlideRule returned 0 points.")
                    else:
                        df_chunk = table.to_pandas()
                        if 'geometry' in df_chunk.columns:
                            df_chunk['geometry'] = shapely.from_wkb(df_chunk['geometry'])
                        feedback.pushInfo(f" -> Successfully downloaded {len(df_chunk)} points.")
                        gdfs_list.append(df_chunk)
                else:
                    feedback.pushInfo(f" -> SlideRule returned 0 points.")
            except Exception as e:
                feedback.pushInfo(f" -> SlideRule query failed for this chunk: {e}")
            finally:
                if os.path.exists(chunk_output_path):
                    try:
                        os.remove(chunk_output_path)
                    except Exception:  # nosec B110
                        pass

        # --- Merge Yearly Results ---
        if gdfs_list:
            gdf = pd.concat(gdfs_list)
        else:
            gdf = pd.DataFrame()

        # --- Local Temporal Filtering (Initial Range) ---
        if not gdf.empty:
            original_len = len(gdf)
            if not isinstance(gdf.index, pd.DatetimeIndex):
                try:
                    gdf.index = pd.to_datetime(gdf.index, utc=True)
                except Exception:  # nosec B110
                    pass

            if isinstance(gdf.index, pd.DatetimeIndex):
                # Standardize index to UTC timezone-aware
                if gdf.index.tz is None:
                    gdf.index = gdf.index.tz_localize('UTC')
                else:
                    gdf.index = gdf.index.tz_convert('UTC')

                # Highlight the latest data date in the log
                latest_retrieved_dt = gdf.index.max()
                feedback.pushInfo(f"==========================================================================")
                feedback.pushInfo(f"⭐ LATEST DATA DATE RETRIEVED: {latest_retrieved_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                feedback.pushInfo(f"==========================================================================")

                t0_utc = parsed_t0 if parsed_t0.tzinfo is not None else parsed_t0.tz_localize('UTC')
                t1_utc = parsed_t1 if parsed_t1.tzinfo is not None else parsed_t1.tz_localize('UTC')

                gdf = gdf[(gdf.index >= t0_utc) & (gdf.index <= t1_utc)]

                filtered_len = len(gdf)
                if filtered_len < original_len:
                    feedback.pushInfo(f"Strict local temporal filter: kept {filtered_len} of {original_len} points matching your exact date range.")

        # --- Handle Empty Results (Inform user of available data range without auto-downloading unwanted dates) ---
        if gdf.empty:
            available_range_msg = ""
            if earthdata:
                try:
                    feedback.pushInfo(f"No data found in requested range. Checking NASA CMR for all available {short_name} data dates in this AOI...")
                    granules = earthdata.cmr(short_name=short_name, polygon=aoi)
                    if granules:
                        dates = []
                        for g in granules:
                            parts = g.split('_')
                            if len(parts) > 1 and len(parts[1]) >= 8:
                                date_str = parts[1][:8]
                                try:
                                    dt = pd.to_datetime(date_str, format="%Y%m%d", utc=True)
                                    dates.append(dt)
                                except ValueError:
                                    pass
                        if dates:
                            min_date = min(dates)
                            max_date = max(dates)
                            available_range_msg = f" Note: Available {short_name} data in this Area of Interest (AOI) ranges from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}."
                        else:
                            available_range_msg = " No granules were found in the NASA CMR database for this AOI."
                    else:
                        available_range_msg = " No granules were found in the NASA CMR database for this AOI."
                except Exception as cmr_err:
                    feedback.pushInfo(f"Could not query NASA CMR for available data dates: {cmr_err}")

            save_log()
            raise QgsProcessingException(
                f"No ICESat-2 ({short_name}) data found in the specified date range ({parsed_t0.strftime('%Y-%m-%d')} to {parsed_t1.strftime('%Y-%m-%d')})."
                f"{available_range_msg} Please adjust your Start Time, End Time, or AOI polygon."
            )

        # Add explicit 'year' and 'acq_date' columns for temporal indexing
        if 'year' not in gdf.columns:
            if isinstance(gdf.index, pd.DatetimeIndex):
                gdf['year'] = gdf.index.year.astype(int)
                gdf['acq_date'] = gdf.index.strftime('%Y-%m-%d')
            elif 'time' in gdf.columns:
                try:
                    dt_col = pd.to_datetime(gdf['time'], utc=True)
                    gdf['year'] = dt_col.dt.year.astype(int)
                    gdf['acq_date'] = dt_col.dt.strftime('%Y-%m-%d')
                except Exception:
                    pass

        # --- Save to QGIS ---
        fields = QgsFields()
        valid_cols = []

        for col in gdf.columns:
            if col == 'geometry':
                continue
            if len(gdf) > 0 and isinstance(gdf[col].iloc[0], (list, dict, np.ndarray)):
                continue

            valid_cols.append(col)
            if pd.api.types.is_float_dtype(gdf[col]):
                fields.append(QgsField(col, QVariant.Double))
            elif pd.api.types.is_integer_dtype(gdf[col]):
                fields.append(QgsField(col, QVariant.Int))
            else:
                fields.append(QgsField(col, QVariant.String))

        dest_crs = QgsCoordinateReferenceSystem("EPSG:4326")
        (sink, dest_id) = self.parameterAsSink(parameters, self.OUTPUT, context, fields, QgsWkbTypes.Point, dest_crs)
        if not sink:
            raise QgsProcessingException("Failed to create output sink for ICESat-2 layer.")
        count = 0
        total = len(gdf) if len(gdf) > 0 else 1
        features = []

        for index, row in gdf.iterrows():
            if feedback.isCanceled():
                break
            fet = QgsFeature()
            fet.setFields(fields)
            
            # Extract coordinates safely
            geom_obj = row['geometry']
            if geom_obj is None or pd.isna(geom_obj):
                continue
                
            fet.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(geom_obj.x, geom_obj.y)))
            
            # Clean and sanitize attributes
            attrs = []
            for c in valid_cols:
                val = row[c]
                if pd.isna(val) or val is None or val is pd.NA:
                    attrs.append(None)
                elif isinstance(val, pd.Timestamp):
                    attrs.append(val.isoformat())
                elif hasattr(val, 'item'):
                    attrs.append(val.item())
                else:
                    attrs.append(val)
                    
            fet.setAttributes(attrs)
            features.append(fet)
            
            # Insert in batches of 10,000 to prevent memory bloat and PyQGIS crashes
            if len(features) >= 10000:
                sink.addFeatures(features)
                count += len(features)
                feedback.setProgress(int(count / total * 100))
                features = []

        # Write remaining features
        if features:
            sink.addFeatures(features)
            count += len(features)
            feedback.setProgress(100)

        save_log()
        return {self.OUTPUT: dest_id}