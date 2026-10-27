# Bathymetrix-AI changelog

## v8.0 — previously introduced STAC downloader

Documentation synchronized 2026-10-27; metadata already declared version 8.0 at inspection.

- Document Sentinel-2A Level-2A and Landsat-8/9 Collection 2 Level-2 data acquisition through Earth Search.
- Record existing ROI, date and scene-cloud filtering, single-scene and temporal median workflows, water-only options, Float32 reflectance, directory-only outputs and quality layers.
- Synchronize the README release label with metadata and add source/product/method citations.
- Preserve all processing, machine-learning, bathymetry, menus and existing downloader implementation.
- Known issue: the supplied QGIS 4 Windows trace reports a native pyproj CRS/PROJ access violation in the original downloader. This documentation-only update does not fix or validate that runtime path.

Earlier version descriptions remain in metadata.txt.
