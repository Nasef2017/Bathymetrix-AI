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

# Attempt to import external libraries
try:
    import sliderule
    from sliderule import sliderule as sr
    import geopandas as gpd
except ImportError:
    pass

class SlideRuleFinalTool(QgsProcessingAlgorithm):
    # Constants
    INPUT = 'INPUT'
    OUTPUT = 'OUTPUT'
    SCENARIO = 'SCENARIO'
    RGT = 'RGT'; CYCLE = 'CYCLE'; BEAM = 'BEAM'
    TIME_START = 'TIME_START'; TIME_END = 'TIME_END'
    USE_CONF = 'USE_CONF'; CONF_THRESHOLD = 'CONF_THRESHOLD'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return SlideRuleFinalTool()

    def name(self):
        # تم تغيير الاسم البرمجي ليعكس وظيفتها العامة
        return 'sliderule_icesat2_downloader'

    def displayName(self):
        # تم تغيير الاسم المعروض للأداة
        return self.tr('ICESat-2 Data Downloader')

    def group(self):
        # تم حذف المجموعة لتظهر الأداة في القائمة الرئيسية
        return self.tr('')

    def groupId(self):
        # تم حذف المجموعة
        return ''

    def shortHelpString(self):
        return self.tr("Downloads ICESat-2 data directly from NASA SlideRule servers.")

    def helpString(self):
        return """
        <p><b>ICESat-2 Data Downloader Tool</b></p>
        <p>This tool connects to the NASA SlideRule server to fetch elevation data from the ICESat-2 satellite for a specified Area of Interest (AOI).</p>
        
        <h3>Inputs</h3>
        <p><b>Input AOI:</b> A polygon layer that defines the geographical area for which you want to download data. The layer's extent will be used to define the region.</p>
        
        <h3>Scenario / Data Product</h3>
        <p>Select the type of data you wish to download:</p>
        <ul>
            <li><b>1. Quick Access (All Data):</b> Provides quick access to all available data.</li>
            <li><b>2. Single Track (Requires RGT):</b> Downloads data for a single, specified track (requires RGT number).</li>
            <li><b>3. Detailed Track (Classified):</b> Detailed and classified track data.</li>
            <li><b>4. ATL03 Photons (Raw Data):</b> Raw photons (ATL03 data product).</li>
            <li><b>5. ATL03 + YAPC (Cleaned Raw Data):</b> Raw photons with the YAPC filter applied for noise removal.</li>
            <li><b>6. ATL06 Surface Height:</b> Processed surface height data (ATL06 data product).</li>
            <li><b>7. ATL24 Bathymetry (Filtered):</b> Filtered bathymetry data (ATL24 data product).</li>
        </ul>

        <h3>Filters</h3>
        <p><b>Time Filter:</b></p>
        <ul>
            <li><b>Start Time:</b> Specify the start of the time period for the search (e.g., 2019-10-01).</li>
            <li><b>End Time:</b> Specify the end of the time period for the search (e.g., 2019-11-01).</li>
        </ul>
        <p><b>Track Filter:</b></p>
        <ul>
            <li><b>RGT (Reference Ground Track):</b> The specific reference ground track number (e.g., 342).</li>
            <li><b>Cycle:</b> The orbit cycle number.</li>
            <li><b>Beam:</b> Specify the desired beam (e.g., gt1l, gt1r, gt2l, ...).</li>
        </ul>
        <p><b>Confidence Filter:</b></p>
        <ul>
            <li><b>Apply Confidence Filter?:</b> When enabled, points will be filtered based on their confidence score.</li>
            <li><b>Confidence Threshold:</b> The minimum confidence value to accept (e.g., 0.9).</li>
        </ul>

        <h3>Output</h3>
        <p><b>Output Layer:</b> A point layer containing the downloaded and filtered data. The layer will be saved in the WGS84 geographic coordinate system (EPSG:4326).</p>
        
        <p><b>Note:</b> This tool requires the <b>sliderule</b> and <b>geopandas</b> libraries to be installed in your QGIS Python environment.</p>
        """

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

        sr.init("slideruleearth.io", verbose=False)

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
        if t0: parms["t0"] = f"{t0}T00:00:00Z"
        if t1: parms["t1"] = f"{t1}T23:59:59Z"
        
        rgt = self.parameterAsInt(parameters, self.RGT, context)
        if rgt > 0: parms["rgt"] = rgt
        
        cycle = self.parameterAsInt(parameters, self.CYCLE, context)
        if cycle > 0: parms["cycle"] = cycle
        
        beam = self.parameterAsString(parameters, self.BEAM, context)
        if beam: parms["beams"] = beam

        scenario = self.parameterAsEnum(parameters, self.SCENARIO, context)
        use_conf = self.parameterAsBool(parameters, self.USE_CONF, context)
        
        # Endpoint Logic
        api_endpoint = "atl24x"
        if scenario == 3: # ATL03
            api_endpoint = "atl03x"
            parms.setdefault("atl24", {})["class_ph"] = ["bathymetry", "sea_surface"]
            parms["cnf"] = -1
        elif scenario == 5: # ATL03 + YAPC
            api_endpoint = "atl03x"
            parms.setdefault("atl24", {})["class_ph"] = ["bathymetry"]
            parms.setdefault("fit", {})["res"] = 10
            parms["fit"]["len"] = 20

        if use_conf:
             parms.setdefault("atl24", {})["confidence_threshold"] = self.parameterAsDouble(parameters, self.CONF_THRESHOLD, context)
             if "class_ph" not in parms["atl24"]:
                 parms["atl24"]["class_ph"] = ["bathymetry", "sea_surface", "unclassified"]

        # --- Request Data ---
        feedback.pushInfo(f"Requesting {api_endpoint} from SlideRule...")
        try:
            gdf = sr.run(api_endpoint, parms)
        except Exception as e:
            raise QgsProcessingException(f"API Error: {e}")

        if gdf.empty:
            feedback.reportError("No Data Found.", True)
            return {self.OUTPUT: None}

        # تم حذف الجزء الخاص بتصحيح المد والجزر
        feedback.pushInfo(f"Downloaded {len(gdf)} points.")
        
        # --- Save to QGIS ---
        fields = QgsFields()
        valid_cols = []
        
        for col in gdf.columns:
            if col == 'geometry': continue
            if len(gdf) > 0 and isinstance(gdf[col].iloc[0], (list, dict, np.ndarray)): continue
            
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
        for index, row in gdf.iterrows():
            if feedback.isCanceled(): break
            fet = QgsFeature()
            fet.setFields(fields)
            fet.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(row['geometry'].x, row['geometry'].y)))
            attrs = [row[c].item() if hasattr(row[c], 'item') else row[c] for c in valid_cols]
            fet.setAttributes(attrs)
            sink.addFeature(fet)
            count += 1
            if count % 500 == 0: feedback.setProgress(int(count/total * 100))

        return {self.OUTPUT: dest_id}