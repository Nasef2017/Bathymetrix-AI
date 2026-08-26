# infrastructure/vector_io.py
try:
    import processing
    from qgis.core import QgsVectorLayer, QgsCoordinateReferenceSystem
except ImportError:
    processing = None
    QgsVectorLayer = None
    QgsCoordinateReferenceSystem = None



def resolve_depth_field(vector_layer, requested_field: str) -> str:
    """
    Finds the best matching depth field in vector_layer.
    If requested_field is present, returns it.
    Otherwise, inspects fields for common depth field names (e.g., ortho_h, h_mean, depth, z, h, elevation, value),
    or falls back to the first numeric field.
    """
    if not vector_layer or not requested_field:
        return requested_field

    if isinstance(vector_layer, str):
        vlayer = QgsVectorLayer(vector_layer, "temp_layer", "ogr")
    else:
        vlayer = vector_layer

    if not vlayer or not vlayer.isValid():
        return requested_field

    field_names = [f.name() for f in vlayer.fields()]
    if not field_names:
        return requested_field

    # Known bathymetry depth field candidates
    candidates = [
        "field_3",
        "field3",
        "final_level",
        "ortho_h",
        "h_mean",
        "depth",
        "z_msl",
        "z_ellip",
        "z",
        "h",
        "elevation",
        "value",
        "grid_code",
    ]

    # 1. Exact match (Strictly respect user choice unless it's a known bad default)
    # If the field is a known metadata field (usually auto-selected by QGIS as the first column),
    # we override it ONLY if a better depth candidate exists.
    bad_defaults = ["confidence", "fid", "id", "objectid", "track", "beam", "pair", "time", "date"]
    
    if requested_field.lower() in bad_defaults:
        # Smart Override: Try to find a real depth candidate instead of the bad default
        for cand in candidates:
            for f in field_names:
                if cand in f.lower():
                    # Found a better candidate! Override the bad default.
                    return f
                    
    # If not a bad default (e.g. user selected 'my_custom_depth'), or no better candidate was found,
    # we STRICTLY respect the user's choice.
    if requested_field in field_names:
        return requested_field

    # 2. Case-insensitive match
    req_lower = requested_field.lower()
    for f in field_names:
        if f.lower() == req_lower:
            return f

    # 3. Fallback to first candidate match
    for cand in candidates:
        for f in field_names:
            if cand in f.lower():
                return f

    # 4. Fallback to first numeric field if no candidate name matched
    for field in vlayer.fields():
        if field.typeName().lower() in ["real", "double", "float", "integer", "int", "int8", "int64"]:
            return field.name()

    return requested_field


def reproject_layer_if_needed(
    vector_layer,
    target_crs: QgsCoordinateReferenceSystem,
    temp_output_path: str,
    context=None,
    feedback=None,
):
    """
    Safely reprojects vector_layer to target_crs.
    Handles Lat/Long (WGS84), UTM, user-defined CRS, missing .prj files, and memory layers.
    """
    if not vector_layer:
        return None

    if isinstance(vector_layer, str):
        vlayer = QgsVectorLayer(vector_layer, "temp_reproject_source", "ogr")
    else:
        vlayer = vector_layer

    if not vlayer or not vlayer.isValid():
        return None

    # Fix missing CRS on vector layer if needed
    source_crs = vlayer.crs()
    current_layer = vlayer

    if not source_crs.isValid():
        if feedback:
            feedback.pushWarning(f"⚠️ Vector layer '{vlayer.name()}' missing CRS. Assigning default WGS 84 (EPSG:4326)...")
        source_crs = QgsCoordinateReferenceSystem("EPSG:4326")
        
        try:
            assign_res = processing.run(
                "native:assignprojection",
                {"INPUT": vlayer, "CRS": source_crs, "OUTPUT": "TEMPORARY_OUTPUT"},
                context=context,
                feedback=feedback,
                is_child_algorithm=True,
            )
            current_layer = assign_res["OUTPUT"]
        except Exception:
            vlayer.setCrs(source_crs)

    if not target_crs or not target_crs.isValid():
        return vlayer.source()

    if source_crs == target_crs:
        return vlayer.source()

    if feedback:
        feedback.pushInfo(
            f"🔄 Reprojecting '{vlayer.name()}' from {source_crs.authid() or source_crs.userFriendlyIdentifier()} "
            f"to target CRS {target_crs.authid() or target_crs.userFriendlyIdentifier()}..."
        )

    try:
        result = processing.run(
            "native:reprojectlayer",
            {"INPUT": current_layer, "TARGET_CRS": target_crs, "OUTPUT": temp_output_path},
            context=context,
            feedback=feedback,
            is_child_algorithm=True,
        )
        return result["OUTPUT"]
    except Exception as e:
        if feedback:
            feedback.pushWarning(f"Reprojection warning: {str(e)}. Proceeding with source layer.")
        return vlayer.source()


def filter_by_depth(layer_source, depth_field, max_depth, context=None, feedback=None):
    """
    Extracts features by depth magnitude while ensuring CRS remains 100% valid.
    """
    if not layer_source:
        return layer_source

    if isinstance(layer_source, str):
        vlayer = QgsVectorLayer(layer_source, "temp_layer", "ogr")
    else:
        vlayer = layer_source

    if not vlayer or not vlayer.isValid():
        return layer_source

    source_crs = vlayer.crs()

    actual_field = resolve_depth_field(vlayer, depth_field)
    if not actual_field:
        return vlayer.source() if hasattr(vlayer, "source") else layer_source

    abs_max = abs(float(max_depth)) if max_depth is not None else 30.0
    expr = f'abs("{actual_field}") <= {abs_max} AND "{actual_field}" != 0'

    if feedback:
        feedback.pushInfo(f"Filtering points... keeping: {expr} (Field: '{actual_field}')")

    try:
        out_dest = "TEMPORARY_OUTPUT"
        if isinstance(layer_source, str) and layer_source.endswith(".gpkg"):
            out_dest = layer_source.replace(".gpkg", "_filtered.gpkg")

        res = processing.run(
            "native:extractbyexpression",
            {"INPUT": vlayer, "EXPRESSION": expr, "OUTPUT": out_dest},
            context=context,
            feedback=feedback,
            is_child_algorithm=True,
        )
        out_layer = res["OUTPUT"]

        # Ensure source CRS is explicitly retained if layer object
        if isinstance(out_layer, QgsVectorLayer) and source_crs.isValid():
            out_layer.setCrs(source_crs)

        return out_layer
    except Exception as e:
        if feedback:
            feedback.pushWarning(f"Depth filtering expression failed: {str(e)}. Proceeding with input layer.")
        return vlayer
