# infrastructure/vector_io.py
import processing
from qgis.core import QgsVectorLayer


def reproject_layer_if_needed(
    vector_layer: QgsVectorLayer,
    target_crs,
    temp_output_path,
    context=None,
    feedback=None,
):
    if not vector_layer:
        return None
    if vector_layer.crs() == target_crs:
        return vector_layer.source()

    if feedback:
        feedback.pushInfo(f"Reprojecting '{vector_layer.name()}'...")
    result = processing.run(
        "native:reprojectlayer",
        {"INPUT": vector_layer, "TARGET_CRS": target_crs, "OUTPUT": temp_output_path},
        context=context,
        feedback=feedback,
        is_child_algorithm=True,
    )
    return result["OUTPUT"]


def filter_by_depth(layer_source, depth_field, max_depth, context=None, feedback=None):
    if not layer_source or not depth_field:
        return layer_source

    expr = (
        f'"{depth_field}" >= {max_depth}'
        if max_depth < 0
        else f'"{depth_field}" <= {max_depth}'
    )
    if feedback:
        feedback.pushInfo(f"Filtering points... keeping: {expr}")

    result = processing.run(
        "native:extractbyexpression",
        {"INPUT": layer_source, "EXPRESSION": expr, "OUTPUT": "memory:"},
        context=context,
        feedback=feedback,
        is_child_algorithm=True,
    )
    return result["OUTPUT"]
