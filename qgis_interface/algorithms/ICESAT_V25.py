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
        return self.tr('ICESat-2 Data Downloader')

    def group(self):
        return self.tr('')

    def groupId(self):
        return ''

    def shortHelpString(self):
        return """
        <div style="font-family: Arial, sans-serif; line-height: 1.2;">
            <h2 style="margin-bottom: 5px;">🛰️ <span style="color: #2E86C1;">ICESat-2 Downloader</span>: NASA SlideRule Client</h2>
            <p style="margin-top: 0; margin-bottom: 10px;">A powerful interface to NASA's SlideRule servers for on-demand processing and downloading of ICESat-2 photon & elevation data.</p>

            <b style="display: block; margin-bottom: 2px;">📦 Data Scenarios</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li><b>ATL03 (Raw):</b> Geolocated photon data (with optional YAPC noise cleaning).</li>
                <li><b>ATL06 (Land Ice):</b> Surface height standard product.</li>
                <li><b>ATL24 (Bathymetry):</b> Specialized experimental bathymetry product.</li>
            </ul>

            <b style="display: block; margin-bottom: 2px;">🔍 Spatiotemporal Filters</b>
            <ul style="margin-top: 0; margin-bottom: 8px; padding-left: 20px;">
                <li><b>AOI:</b> Defined by input polygon extent.</li>
                <li><b>Time:</b> Filter by specific date range (Start/End).</li>
                <li><b>Track ID:</b> Specific RGT, Cycle, or Beam selection.</li>
            </ul>

            <p style="margin-top: 10px; border-top: 1px solid #ccc; padding-top: 5px;">
                <b style="color: #E74C3C;">⚠️ Requirements:</b> Requires <i>sliderule</i> and <i>geopandas</i> Python libraries.
            </p>

            <p style="margin-top: 5px;">
                <b>Developer:</b> Mohamed Aly Nasef
            </p>
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
        self.addParameter(QgsProcessingParameterString(self.TIME_START, self.tr('Start Time (YYYY-MM-DD)'), defaultValue='2019-10-01', optional=True))
        self.addParameter(QgsProcessingParameterString(self.TIME_END, self.tr('End Time (YYYY-MM-DD)'), defaultValue='2019-11-01', optional=True))
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
                if output_param and output_param != 'TEMPORARY_OUTPUT':
                    base, _ = os.path.splitext(output_param)
                    log_filepath = base + "_log.txt"
                else:
                    nonlocal dest_id
                    if dest_id:
                        clean_path = dest_id.split('|')[0]
                        if os.path.dirname(clean_path):
                            base, _ = os.path.splitext(clean_path)
                            log_filepath = base + "_log.txt"
                
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

        import datetime
        now = pd.Timestamp.now(tz='UTC')
        parsed_t0 = None
        parsed_t1 = None

        if t0:
            try:
                parsed_t0 = pd.to_datetime(t0, utc=True)
            except Exception as e:
                feedback.pushInfo(f"Warning: Could not parse start time '{t0}'. Using mission start. Error: {e}")

        if not parsed_t0:
            parsed_t0 = pd.to_datetime("2018-10-14T00:00:00Z", utc=True)

        if t1:
            try:
                parsed_t1 = pd.to_datetime(t1, utc=True)
                # If end date is in the future, cap it to the current time
                if parsed_t1 > now:
                    feedback.pushInfo(f"End date '{t1}' is in the future. Capping end date to current time: {now.strftime('%Y-%m-%d')}")
                    parsed_t1 = now
            except Exception as e:
                feedback.pushInfo(f"Warning: Could not parse end time '{t1}'. Using current time. Error: {e}")

        if not parsed_t1:
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
                except Exception:
                    pass
            chunk_parms["output"] = {
                "path": chunk_output_path,
                "format": "geoparquet",
                "open_on_complete": True
            }

            feedback.pushInfo(f"[{idx+1}/{len(chunks)}] Requesting {api_endpoint} from SlideRule ({start_chunk.strftime('%Y-%m-%d')} to {end_chunk.strftime('%Y-%m-%d')})...")
            try:
                gdf_chunk = sr.run(api_endpoint, chunk_parms)
                if gdf_chunk is None or gdf_chunk.empty:
                    feedback.pushInfo(f" -> SlideRule returned 0 points.")
                else:
                    feedback.pushInfo(f" -> Successfully downloaded {len(gdf_chunk)} points.")
                    gdfs_list.append(gdf_chunk)
            except Exception as e:
                feedback.pushInfo(f" -> SlideRule query failed for this chunk: {e}")
            finally:
                if os.path.exists(chunk_output_path):
                    try:
                        os.remove(chunk_output_path)
                    except Exception:
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
                except Exception:
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
                # Adjust to end of the day to make it inclusive
                t1_utc = t1_utc.normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

                gdf = gdf[(gdf.index >= t0_utc) & (gdf.index <= t1_utc)]

                filtered_len = len(gdf)
                if filtered_len < original_len:
                    feedback.pushInfo(f"Strict local temporal filter: kept {filtered_len} of {original_len} points matching your exact date range.")

        # --- Empty DataFrame Fallback (Auto-download latest available year of data) ---
        if gdf.empty:
            available_range_msg = ""
            latest_start = None
            latest_end = None
            if earthdata:
                try:
                    feedback.pushInfo(f"No data found. Querying NASA CMR to find available {short_name} data for this area...")
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
                            
                            # Capping end to latest date, and start to beginning of that calendar year
                            latest_end = max_date
                            latest_start = pd.to_datetime(f"{latest_end.year}-01-01", utc=True)
                            feedback.pushInfo(f"Automatically attempting to download the latest available data chunk (from {latest_start.strftime('%Y-%m-%d')} to {latest_end.strftime('%Y-%m-%d')})...")
                        else:
                            available_range_msg = " No granules were found in the CMR database for this AOI."
                    else:
                        available_range_msg = " No granules were found in the CMR database for this AOI."
                except Exception as cmr_err:
                    feedback.pushInfo(f"Could not query NASA CMR for metadata fallback: {cmr_err}")

            # Run automatic query for fallback range (latest year of available data)
            if latest_start and latest_end:
                fallback_chunks = []
                current_start = latest_start
                while current_start <= latest_end:
                    year = current_start.year
                    current_end = pd.to_datetime(f"{year}-12-31T23:59:59Z", utc=True)
                    if current_end > latest_end:
                        current_end = latest_end
                    fallback_chunks.append((current_start, current_end))
                    current_start = pd.to_datetime(f"{year + 1}-01-01T00:00:00Z", utc=True)

                fallback_gdfs = []
                for idx, (fs, fe) in enumerate(fallback_chunks):
                    if feedback.isCanceled():
                        break
                    fs_str = fs.strftime("%Y-%m-%dT%H:%M:%SZ")
                    fe_str = fe.strftime("%Y-%m-%dT%H:%M:%SZ")
                    chunk_parms = parms.copy()
                    chunk_parms["t0"] = fs_str
                    chunk_parms["t1"] = fe_str
                    
                    # Generate a unique temporary file path for this fallback chunk
                    chunk_output_path = os.path.join(safe_temp_dir, f"fallback_{idx}_{fs.strftime('%Y%m%d')}.parquet")
                    if os.path.exists(chunk_output_path):
                        try:
                            os.remove(chunk_output_path)
                        except Exception:
                            pass
                    chunk_parms["output"] = {
                        "path": chunk_output_path,
                        "format": "geoparquet",
                        "open_on_complete": True
                    }
                    
                    feedback.pushInfo(f"[{idx+1}/{len(fallback_chunks)}] Fallback Requesting {api_endpoint} from SlideRule ({fs.strftime('%Y-%m-%d')} to {fe.strftime('%Y-%m-%d')})...")
                    try:
                        gdf_chunk = sr.run(api_endpoint, chunk_parms)
                        if gdf_chunk is not None and not gdf_chunk.empty:
                            feedback.pushInfo(f" -> Successfully downloaded {len(gdf_chunk)} points.")
                            fallback_gdfs.append(gdf_chunk)
                        else:
                            feedback.pushInfo(f" -> SlideRule returned 0 points.")
                    except Exception as fallback_err:
                        feedback.pushInfo(f" -> Fallback query failed for this chunk: {fallback_err}")
                    finally:
                        if os.path.exists(chunk_output_path):
                            try:
                                os.remove(chunk_output_path)
                            except Exception:
                                pass

                if fallback_gdfs:
                    gdf = pd.concat(fallback_gdfs)
                    # Filter locally to fallback range
                    if not gdf.empty:
                        if not isinstance(gdf.index, pd.DatetimeIndex):
                            try:
                                gdf.index = pd.to_datetime(gdf.index, utc=True)
                            except Exception:
                                pass
                        if isinstance(gdf.index, pd.DatetimeIndex):
                            if gdf.index.tz is None:
                                gdf.index = gdf.index.tz_localize('UTC')
                            else:
                                gdf.index = gdf.index.tz_convert('UTC')

                            # Highlight the latest date in the fallback downloaded data
                            max_date = gdf.index.max()
                            feedback.pushInfo(f"==========================================================================")
                            feedback.pushInfo(f"⭐ LATEST DATA DATE RETRIEVED (FALLBACK): {max_date.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                            feedback.pushInfo(f"==========================================================================")

                            t0_utc = latest_start
                            t1_utc = latest_end.normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                            gdf = gdf[(gdf.index >= t0_utc) & (gdf.index <= t1_utc)]

            if gdf.empty:
                save_log()
                raise QgsProcessingException(f"No Data Found.{available_range_msg} Please adjust your temporal or spatial filters.")

        feedback.pushInfo(f"Downloaded {len(gdf)} points.")

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

        (sink, dest_id) = self.parameterAsSink(parameters, self.OUTPUT, context, fields, QgsWkbTypes.Point, dest_crs)

        total = len(gdf)
        count = 0
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