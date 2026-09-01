import os
import re
from typing import Dict, List, Optional, Tuple, Any
try:
    from qgis.core import QgsVectorLayer, QgsProcessingException
except ImportError:
    QgsVectorLayer = None
    QgsProcessingException = Exception


class TemporalDataScanner:
    """
    Scans multi-year directories and matches yearly datasets:
    - Auto-discovers subfolders named by year (e.g. /2020/, /2021/)
    - Satellite GeoTIFF imagery per year
    - ICESat-2 vector shapefiles per year (or fallback to single multi-year layer)
    - Optional Residual correction rasters/vectors per year
    - Optional Unseen validation test points per year
    """

    def __init__(self, root_img_dir: str, start_year: Optional[int] = None, end_year: Optional[int] = None):
        if root_img_dir:
            root_img_dir = os.path.normpath(root_img_dir)
            if os.path.isfile(root_img_dir):
                root_img_dir = os.path.dirname(root_img_dir)
            
            # If the user selected a 4-digit year folder directly (e.g. Dabaa/2019):
            if os.path.isdir(root_img_dir):
                base_name = os.path.basename(root_img_dir)
                if base_name.isdigit() and len(base_name) == 4:
                    parent_dir = os.path.dirname(root_img_dir)
                    if os.path.isdir(parent_dir):
                        sub_items = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d)) and d.isdigit() and len(d) == 4]
                        if sub_items:
                            root_img_dir = parent_dir
                            
        self.root_img_dir = root_img_dir
        self.start_year = start_year
        self.end_year = end_year

    def scan_yearly_datasets(
        self,
        icesat_layer: Optional[QgsVectorLayer] = None,
        icesat_year_field: str = "",
        unseen_layer: Optional[QgsVectorLayer] = None,
        unseen_year_field: str = "",
    ) -> Dict[int, Dict[str, Any]]:
        """
        Scans root directory and pairs per-year data.
        """
        if not os.path.exists(self.root_img_dir):
            raise QgsProcessingException(
                f"Dataset root directory does not exist: {self.root_img_dir}"
            )

        # Auto-discover target years from directory structure
        if self.start_year is not None and self.end_year is not None:
            target_years = list(range(self.start_year, self.end_year + 1))
        else:
            target_years = self._auto_discover_years()

        if not target_years:
            raise QgsProcessingException(
                f"No year subfolders (e.g. /2020/, /2021/) found inside '{self.root_img_dir}'!"
            )

        yearly_data = {}

        for yr in target_years:
            yr_str = str(yr)
            year_dir = os.path.join(self.root_img_dir, yr_str)
            
            # 1. Find Year Image
            img_path = self._find_satellite_image(yr_str)
            if not img_path:
                raise QgsProcessingException(
                    f"No valid GeoTIFF image found for year {yr} in '{self.root_img_dir}'! "
                    f"Ensure directory contains folder '{yr}' with satellite imagery (.tif/.jp2)."
                )

            # 2. Find Year ICESat-2 / Training File (check year subfolder first)
            year_icesat_path = self._find_year_icesat(yr_str)

            # 3. Find Year Control Points (check year subfolder)
            control_path = self._find_year_control(yr_str)

            # 4. Find Year Unseen Validation File
            year_unseen_path = self._find_year_unseen(yr_str, unseen_layer)

            yearly_data[yr] = {
                "year": yr,
                "image_path": img_path,
                "icesat_file_path": year_icesat_path,
                "icesat_layer": icesat_layer,
                "icesat_year_field": icesat_year_field,
                "control_path": control_path,
                "unseen_file_path": year_unseen_path,
                "unseen_layer": unseen_layer,
                "unseen_year_field": unseen_year_field,
            }

        return yearly_data

    def _auto_discover_years(self) -> List[int]:
        """Auto-discovers subfolders named by 4-digit years (e.g. /2020/, /2021/)."""
        years = []
        for item in os.listdir(self.root_img_dir):
            item_path = os.path.join(self.root_img_dir, item)
            if os.path.isdir(item_path) and item.isdigit() and len(item) == 4:
                years.append(int(item))

        if not years:
            for fname in os.listdir(self.root_img_dir):
                matches = re.findall(r"\b(20\d{2}|19\d{2})\b", fname)
                for m in matches:
                    years.append(int(m))

        return sorted(list(set(years)))
    def _find_satellite_image(self, year_str: str) -> Optional[str]:
        """Finds Satellite raster image for a specific year."""
        year_subfolder = os.path.join(self.root_img_dir, year_str)
        if os.path.exists(year_subfolder) and os.path.isdir(year_subfolder):
            for fname in os.listdir(year_subfolder):
                if fname.lower().endswith((".tif", ".tiff", ".jp2", ".vrt")):
                    return os.path.join(year_subfolder, fname)

        for fname in os.listdir(self.root_img_dir):
            fpath = os.path.join(self.root_img_dir, fname)
            if os.path.isfile(fpath) and fname.lower().endswith((".tif", ".tiff", ".jp2", ".vrt")):
                if year_str in fname:
                    return fpath

        return None
    def _find_year_icesat(self, year_str: str) -> Optional[str]:
        """Finds ICESat-2 or training shapefile/geopackage inside the year subfolder."""
        year_subfolder = os.path.join(self.root_img_dir, year_str)
        if os.path.exists(year_subfolder) and os.path.isdir(year_subfolder):
            candidate_files = []
            for fname in os.listdir(year_subfolder):
                fname_lower = fname.lower()
                if fname_lower.endswith((".shp", ".gpkg", ".geojson", ".csv", ".kml", ".tab")):
                    # Explicitly ignore validation and control files
                    if any(x in fname_lower for x in ["valid", "test", "unseen", "sonar", "control", "residual", "error", "eval"]):
                        continue
                    candidate_files.append(fname)
            
            if candidate_files:
                # 1st Priority: files containing known training/depth keywords or the year itself
                for fname in candidate_files:
                    fname_lower = fname.lower()
                    if any(k in fname_lower for k in ["icesat", "atl", "train", "bathy", "depth", "point", "survey", "lidar", "data", year_str]):
                        return os.path.join(year_subfolder, fname)
                
                # 2nd Priority: grab the first valid shapefile / vector in the folder
                return os.path.join(year_subfolder, candidate_files[0])
        return None

    def _find_year_unseen(self, year_str: str, unseen_layer: Optional[QgsVectorLayer]) -> Optional[str]:
        """Finds unseen validation point shapefile inside the year subfolder."""
        year_subfolder = os.path.join(self.root_img_dir, year_str)
        if os.path.exists(year_subfolder) and os.path.isdir(year_subfolder):
            for fname in os.listdir(year_subfolder):
                fname_lower = fname.lower()
                if fname_lower.endswith((".shp", ".gpkg", ".geojson", ".csv", ".kml", ".tab")) and any(k in fname_lower for k in ["unseen", "sonar", "test", "valid", "eval", "chk", "ground"]):
                    return os.path.join(year_subfolder, fname)
        return None

    def _find_year_control(self, year_str: str) -> Optional[str]:
        """Finds control points (for residual correction) inside the year subfolder."""
        year_subfolder = os.path.join(self.root_img_dir, year_str)
        if os.path.exists(year_subfolder) and os.path.isdir(year_subfolder):
            for fname in os.listdir(year_subfolder):
                fname_lower = fname.lower()
                if any(k in fname_lower for k in ["control", "ctrl", "residual", "datum", "error"]):
                    if fname_lower.endswith((".shp", ".gpkg", ".geojson", ".csv", ".tif", ".tiff")):
                        return os.path.join(year_subfolder, fname)
        return None
