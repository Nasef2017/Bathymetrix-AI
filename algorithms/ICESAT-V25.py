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
                       QgsProject,
                       QgsFeature,
                       QgsGeometry,
                       QgsPointXY,
                       QgsFields,
                       QgsField,
                       QgsWkbTypes,
                       QgsProcessingException)
import sys

# Attempt to import external libraries
try:
    import sliderule
    from sliderule import sliderule as sr
    import geopandas as gpd
    import pandas as pd
    import numpy as np
except ImportError:
    pass

class SlideRuleFinalTool(QgsProcessingAlgorithm):
    # Constants for parameter names
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
        return 'sliderule_atl24_v6_highconf'

    def displayName(self):
        return self.tr('SlideRule ATL24 Downloader (High Confidence Default)')

    # --- Added Group Methods Here ---
    def group(self):
        return self.tr('IceSat-02 Downloader')

    def groupId(self):
        return 'sliderule_tools'
    # --------------------------------

    def initAlgorithm(self, config=None):
        # 1. Basic Inputs
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                self.tr('Input AOI (Shapefile/Polygon)'),
                [QgsProcessing.TypeVectorPolygon]
            )
        )

        # Define available scenarios matching the Python tutorial
        scenarios = [
            "1. Quick Access (All Data)",              
            "2. Single Track (Requires RGT)",                 
            "3. Detailed Track (Classified)",    
            "4. ATL03 Photons (Raw Data - Heavy!)",              
            "5. ATL03 + YAPC (Cleaned Raw Data)",             
            "6. ATL06 Surface Height (Lightweight)",             
            "7. ATL24 Bathymetry (Filtered)"                
        ]
        self.addParameter(
            QgsProcessingParameterEnum(
                self.SCENARIO,
                self.tr('Scenario / Data Product'),
                options=scenarios,
                defaultValue=0
            )
        )

        # --- Filtering Section ---
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.USE_CONF,
                self.tr('Apply Confidence Filter?'),
                defaultValue=False
            )
        )
        
        # Default set to 0.9 for high quality data
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CONF_THRESHOLD,
                self.tr('Confidence Threshold (0.0 to 1.0)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.9,
                minValue=0.0,
                maxValue=1.0,
                optional=True
            )
        )

        # --- Time Filters ---
        self.addParameter(
            QgsProcessingParameterString(
                self.TIME_START,
                self.tr('Start Time (YYYY-MM-DD)'),
                defaultValue='2019-10-01',
                optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.TIME_END,
                self.tr('End Time (YYYY-MM-DD)'),
                defaultValue='2019-11-01',
                optional=True
            )
        )

        # --- Track Filters ---
        self.addParameter(
            QgsProcessingParameterNumber(
                self.RGT,
                self.tr('RGT (Optional)'),
                type=QgsProcessingParameterNumber.Integer,
                optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CYCLE,
                self.tr('Cycle (Optional)'),
                type=QgsProcessingParameterNumber.Integer,
                optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.BEAM,
                self.tr('Beam (e.g., gt3r)'),
                optional=True
            )
        )

        # Output Layer
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT,
                self.tr('Output Layer')
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        # Check if libraries are installed
        if 'sliderule' not in sys.modules:
            raise QgsProcessingException("The 'sliderule' library is not installed in the QGIS Python environment.")

        # Initialize SlideRule
        sr.init("slideruleearth.io", verbose=False)

        # --- Prepare Area of Interest (AOI) ---
        source = self.parameterAsSource(parameters, self.INPUT, context)
        source_crs = source.sourceCrs()
        dest_crs = QgsCoordinateReferenceSystem("EPSG:4326")
        transform = QgsCoordinateTransform(source_crs, dest_crs, context.project())
        extent = source.sourceExtent()
        extent_wgs84 = transform.transformBoundingBox(extent)
        
        # Create polygon list (counter-clockwise)
        aoi = [
            {"lon": extent_wgs84.xMinimum(), "lat": extent_wgs84.yMinimum()},
            {"lon": extent_wgs84.xMaximum(), "lat": extent_wgs84.yMinimum()},
            {"lon": extent_wgs84.xMaximum(), "lat": extent_wgs84.yMaximum()},
            {"lon": extent_wgs84.xMinimum(), "lat": extent_wgs84.yMaximum()},
            {"lon": extent_wgs84.xMinimum(), "lat": extent_wgs84.yMinimum()} 
        ]

        # --- Read User Parameters ---
        rgt = self.parameterAsInt(parameters, self.RGT, context)
        cycle = self.parameterAsInt(parameters, self.CYCLE, context)
        beam = self.parameterAsString(parameters, self.BEAM, context)
        t0 = self.parameterAsString(parameters, self.TIME_START, context)
        t1 = self.parameterAsString(parameters, self.TIME_END, context)
        scenario = self.parameterAsEnum(parameters, self.SCENARIO, context)
        
        # Read Confidence Filter parameters
        use_conf = self.parameterAsBool(parameters, self.USE_CONF, context)
        conf_threshold = self.parameterAsDouble(parameters, self.CONF_THRESHOLD, context)

        # --- Build Base Parameters ---
        parms = {"poly": aoi}
        
        # Add Time Filters if provided
        if t0 and len(t0) > 5: parms["t0"] = f"{t0}T00:00:00Z"
        if t1 and len(t1) > 5: parms["t1"] = f"{t1}T23:59:59Z"
        
        # Add Track Filters if provided
        if rgt > 0: parms["rgt"] = rgt
        if cycle > 0: parms["cycle"] = cycle
        if beam: parms["beams"] = beam

        # Determine API Endpoint
        api_endpoint = "atl24x"
        
        # Scenario Logic
        if scenario == 3: # ATL03 Raw
            api_endpoint = "atl03x"
            # Initialize atl24 dict if not exists
            parms.setdefault("atl24", {})["class_ph"] = ["bathymetry", "sea_surface"]
            parms["cnf"] = -1
            
        elif scenario == 5: # ATL06 Surface
            api_endpoint = "atl03x"
            parms.setdefault("atl24", {})["class_ph"] = ["bathymetry"]
            parms.setdefault("fit", {})["res"] = 10
            parms["fit"]["len"] = 20
            parms["fit"]["pass_invalid"] = True

        # --- Apply Confidence Filter ---
        if use_conf:
            feedback.pushInfo(f"Adding Confidence Filter: Threshold >= {conf_threshold}")
            if "atl24" not in parms:
                parms["atl24"] = {}
            
            parms["atl24"]["confidence_threshold"] = conf_threshold
            
            # Ensure class_ph is set to allow filtering logic to work properly
            if "class_ph" not in parms["atl24"]:
                 parms["atl24"]["class_ph"] = ["bathymetry", "sea_surface", "unclassified"]

        # --- Execute Request ---
        feedback.pushInfo(f"Requesting {api_endpoint}...")
        
        try:
            gdf = sr.run(api_endpoint, parms)
        except Exception as e:
            msg = str(e)
            if "WinError 2" in msg:
                raise QgsProcessingException("Time Out Error: The area is too large or the time range is too long. Please reduce the AOI or the date range.")
            else:
                raise QgsProcessingException(f"API Error: {msg}")

        if gdf.empty:
            feedback.reportError("No Data Found.", True)
            return {self.OUTPUT: None}

        feedback.pushInfo(f"Received {len(gdf)} points. Saving...")

        # --- Convert and Save to QGIS Layer ---
        fields = QgsFields()
        valid_cols = []
        
        # Determine valid columns and types
        for col in gdf.columns:
            if col == 'geometry': continue
            # Skip complex types (lists, dicts)
            if len(gdf) > 0 and isinstance(gdf[col].iloc[0], (list, dict, np.ndarray)): continue
            
            valid_cols.append(col)
            if pd.api.types.is_float_dtype(gdf[col]): 
                fields.append(QgsField(col, QVariant.Double))
            elif pd.api.types.is_integer_dtype(gdf[col]): 
                fields.append(QgsField(col, QVariant.Int))
            else: 
                fields.append(QgsField(col, QVariant.String))

        # Initialize Output Sink
        (sink, dest_id) = self.parameterAsSink(parameters, self.OUTPUT, context, fields, QgsWkbTypes.Point, dest_crs)
        
        if sink is None:
            raise QgsProcessingException("Invalid Sink. Could not create output layer.")

        total = len(gdf)
        count = 0
        
        # Iterate over rows and add features
        for index, row in gdf.iterrows():
            if feedback.isCanceled(): break
            
            fet = QgsFeature()
            fet.setFields(fields)
            fet.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(row['geometry'].x, row['geometry'].y)))
            
            # Prepare attributes, handling numpy types
            attrs = [row[c].item() if hasattr(row[c], 'item') else row[c] for c in valid_cols]
            fet.setAttributes(attrs)
            
            sink.addFeature(fet)
            count += 1
            
            # Update progress bar every 500 features
            if count % 500 == 0: 
                feedback.setProgress(int(count/total * 100))

        return {self.OUTPUT: dest_id}