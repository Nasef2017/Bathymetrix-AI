# infrastructure/canvas.py
from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer, QgsProcessingContext


def add_raster_to_canvas(path, layer_name, context=None):
    if context:
        details = QgsProcessingContext.LayerDetails(
            layer_name, QgsProject.instance(), layer_name
        )
        context.addLayerToLoadOnCompletion(path, details)
    else:
        QgsProject.instance().addMapLayer(QgsRasterLayer(path, layer_name))


def add_vector_to_canvas(path, layer_name, provider="ogr", context=None):
    if context:
        details = QgsProcessingContext.LayerDetails(
            layer_name, QgsProject.instance(), layer_name
        )
        context.addLayerToLoadOnCompletion(path, details)
    else:
        QgsProject.instance().addMapLayer(QgsVectorLayer(path, layer_name, provider))
