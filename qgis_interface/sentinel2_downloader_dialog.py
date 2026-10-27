# -*- coding: utf-8 -*-
"""
qgis_interface/sentinel2_downloader_dialog.py
Interactive PyQt Dialog for Satellite Imagery Downloader (Earth Search STAC).
Supports exactly 3 satellites:
1. Sentinel-2A (MSI Level-2A BOA Surface Reflectance)
2. Landsat-8 (OLI Collection 2 Level-2 Surface Reflectance)
3. Landsat-9 (OLI-2 Collection 2 Level-2 Surface Reflectance)

Sentinel-2B and Sentinel-2C are strictly excluded and rejected.
Supports:
- 5 AOI Selection Modes (Polygon Layer, Extent Layer, Map Canvas, Draw AOI on Canvas, AOI File)
- Interactive STAC Search with Background QgsTask
- Sortable & Filterable Results Table with Scene Footprint Visualization on QGIS Canvas
- Preview Display with HTTP / HTTPS thumbnail fetching
- Sensor-Adaptive Spectral Band & Resolution Selection (10m / 20m / 30m)
- Single Scene & Temporal Median Composite Modes
- Water-Only Output & Advanced Masking Configuration (SCL & Landsat QA_PIXEL)
- Background Execution with Progress Tracking & Cancellation
"""

import os
import json
import urllib.request
from datetime import datetime, date

from qgis.PyQt.QtCore import (
    Qt,
    QDate,
    pyqtSignal,
    QObject
)
from qgis.PyQt.QtGui import (
    QColor,
    QPixmap,
    QIcon,
    QFont
)
from qgis.PyQt.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QButtonGroup,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QProgressBar,
    QFileDialog,
    QMessageBox,
    QSplitter,
    QTabWidget,
    QWidget,
    QScrollArea
)

from qgis.core import (
    QgsProject,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsPointXY,
    QgsRectangle,
    QgsMapLayerProxyModel,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsTask,
    QgsApplication,
    QgsWkbTypes
)
from qgis.gui import (
    QgsMapToolExtent,
    QgsRubberBand,
    QgsMapLayerComboBox
)

from ..algorithms.Sentinel2_Downloader import Sentinel2DownloaderAlgorithm
from ...core.stac.stac_client import (
    STACClient,
    STACSearchError,
    BAND_ASSET_MAP,
    ORDERED_BAND_LIST,
    LANDSAT_BAND_ASSET_MAP,
    LANDSAT_ORDERED_BAND_LIST,
    ALLOWED_PLATFORMS,
    DISALLOWED_PLATFORMS
)
from ...core.stac.processor import (
    process_single_scene,
    process_temporal_median_composite,
    ProcessorError
)


class BackgroundSearchTask(QgsTask):
    """Asynchronous background task for STAC search to keep UI 100% fluid."""
    search_completed = pyqtSignal(list, str)

    def __init__(self, search_params: dict):
        super().__init__("Searching Satellite STAC Scenes", QgsTask.CanCancel)
        self.params = search_params
        self.results = []
        self.error_msg = ""

    def run(self):
        try:
            client = STACClient()
            self.results = client.search_scenes(
                aoi_geometry=self.params.get("aoi_geometry"),
                bbox=self.params.get("bbox"),
                start_date=self.params.get("start_date", "2023-01-01"),
                end_date=self.params.get("end_date", "2023-12-31"),
                max_cloud_cover=self.params.get("max_cloud_cover", 10.0),
                platform=self.params.get("platform"),
                min_aoi_coverage=self.params.get("min_aoi_coverage", 0.0),
                max_scenes=self.params.get("max_scenes", 50),
                feedback=self
            )
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def finished(self, result):
        if result:
            self.search_completed.emit(self.results, "")
        else:
            self.search_completed.emit([], self.error_msg or "Search canceled or encountered an error.")

    def pushInfo(self, msg: str):
        self.setProgress(min(90.0, self.progress() + 10.0))


class BackgroundProcessingTask(QgsTask):
    """Asynchronous background task for downloading and compositing satellite data."""
    task_done = pyqtSignal(dict, str)
    status_updated = pyqtSignal(str)

    def __init__(self, mode: str, kwargs: dict):
        super().__init__(f"Satellite Processing: {mode}", QgsTask.CanCancel)
        self.mode = mode
        self.kwargs = kwargs
        self.results = {}
        self.error_msg = ""

    def run(self):
        try:
            self.kwargs["feedback"] = self
            if self.mode == "single":
                self.results = process_single_scene(**self.kwargs)
            else:
                self.results = process_temporal_median_composite(**self.kwargs)
            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def finished(self, result):
        if result:
            self.task_done.emit(self.results, "")
        else:
            self.task_done.emit({}, self.error_msg or "Processing was canceled or failed.")

    def pushInfo(self, msg: str):
        self.status_updated.emit(msg)

    def pushWarning(self, msg: str):
        self.status_updated.emit(f"Warning: {msg}")


class Sentinel2DownloaderDialog(QDialog):
    """
    Main GUI Dialog for Satellite Imagery Downloader and Temporal Compositor.
    Supports Sentinel-2A, Landsat-8, and Landsat-9 Surface Reflectance.
    """

    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.canvas = iface.mapCanvas() if iface else None

        self.setWindowTitle("Bathymetrix-AI | Satellite Imagery Downloader (Sentinel-2A & Landsat-8/9)")
        self.resize(1080, 800)
        self.setMinimumSize(880, 680)

        self.found_scenes = []
        self.current_aoi_geojson = None
        self.draw_tool = None
        self.rubber_band = None
        self.footprint_band = None
        self.active_task = None

        self._setup_ui()
        self._setup_rubber_bands()
        self._populate_initial_dates()

    def _setup_rubber_bands(self):
        if not self.canvas:
            return
        # AOI Drawn Box rubber band
        self.rubber_band = QgsRubberBand(self.canvas, QgsWkbTypes.PolygonGeometry)
        self.rubber_band.setColor(QColor(230, 81, 0, 70))
        self.rubber_band.setStrokeColor(QColor(230, 81, 0, 220))
        self.rubber_band.setWidth(2)

        # Scene footprint rubber band
        self.footprint_band = QgsRubberBand(self.canvas, QgsWkbTypes.PolygonGeometry)
        self.footprint_band.setColor(QColor(0, 150, 136, 40))
        self.footprint_band.setStrokeColor(QColor(0, 150, 136, 220))
        self.footprint_band.setWidth(2)

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # Header Title Banner
        header = QWidget()
        header.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1A5276, stop:1 #117864); border-radius: 6px; padding: 10px;")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(10, 6, 10, 6)

        title_lbl = QLabel("🛰️ Satellite Surface Reflectance Downloader (Sentinel-2A & Landsat-8/9)")
        title_font = QFont("Segoe UI", 13, QFont.Bold)
        title_lbl.setFont(title_font)
        title_lbl.setStyleSheet("color: white;")
        h_layout.addWidget(title_lbl)

        sub_lbl = QLabel("Element 84 Earth Search STAC | S2A Level-2A & Landsat-8/9 Collection 2 Level-2")
        sub_lbl.setStyleSheet("color: #E8F8F5; font-size: 11px;")
        h_layout.addWidget(sub_lbl, alignment=Qt.AlignRight)
        main_layout.addWidget(header)

        # Splitter dividing Settings and Results
        splitter = QSplitter(Qt.Vertical)

        # Top Control Widget (AOI, Mode, Search Criteria)
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(8)

        # --- Section 1: AOI Selection ---
        aoi_group = QGroupBox("1. Area of Interest (AOI) Definition")
        aoi_grid = QGridLayout(aoi_group)
        aoi_grid.setContentsMargins(10, 10, 10, 10)
        aoi_grid.setSpacing(8)

        self.aoi_btn_group = QButtonGroup(self)
        self.rb_polygon_layer = QRadioButton("Polygon Vector Layer")
        self.rb_extent_layer = QRadioButton("Layer Extent")
        self.rb_canvas_extent = QRadioButton("Map Canvas Extent")
        self.rb_draw_canvas = QRadioButton("Draw AOI on Canvas")
        self.rb_file_aoi = QRadioButton("AOI File (GeoJSON / GPKG / SHP)")
        self.rb_canvas_extent.setChecked(True)

        self.aoi_btn_group.addButton(self.rb_polygon_layer, 0)
        self.aoi_btn_group.addButton(self.rb_extent_layer, 1)
        self.aoi_btn_group.addButton(self.rb_canvas_extent, 2)
        self.aoi_btn_group.addButton(self.rb_draw_canvas, 3)
        self.aoi_btn_group.addButton(self.rb_file_aoi, 4)

        aoi_grid.addWidget(self.rb_canvas_extent, 0, 0)
        aoi_grid.addWidget(self.rb_polygon_layer, 0, 1)
        aoi_grid.addWidget(self.rb_extent_layer, 0, 2)
        aoi_grid.addWidget(self.rb_draw_canvas, 0, 3)
        aoi_grid.addWidget(self.rb_file_aoi, 0, 4)

        # Layer combos & file pickers
        self.layer_combo = QgsMapLayerComboBox()
        self.layer_combo.setFilters(QgsMapLayerProxyModel.PolygonLayer)
        self.layer_combo.setEnabled(False)
        aoi_grid.addWidget(self.layer_combo, 1, 0, 1, 2)

        self.cb_selected_only = QCheckBox("Selected features only")
        self.cb_selected_only.setEnabled(False)
        aoi_grid.addWidget(self.cb_selected_only, 1, 2)

        self.btn_draw = QPushButton("✏️ Draw Rectangle on Canvas")
        self.btn_draw.setEnabled(False)
        self.btn_draw.clicked.connect(self._activate_draw_tool)
        aoi_grid.addWidget(self.btn_draw, 1, 3)

        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Select AOI file...")
        self.file_edit.setEnabled(False)
        self.btn_file_browse = QPushButton("Browse...")
        self.btn_file_browse.setEnabled(False)
        self.btn_file_browse.clicked.connect(self._browse_aoi_file)

        file_box = QHBoxLayout()
        file_box.addWidget(self.file_edit)
        file_box.addWidget(self.btn_file_browse)
        aoi_grid.addLayout(file_box, 1, 4)

        self.aoi_status_lbl = QLabel("AOI: Current Map Canvas Extent ready.")
        self.aoi_status_lbl.setStyleSheet("color: #196F3D; font-weight: bold; font-size: 11px;")
        aoi_grid.addWidget(self.aoi_status_lbl, 2, 0, 1, 5)

        self.aoi_btn_group.buttonClicked.connect(self._on_aoi_mode_changed)
        top_layout.addWidget(aoi_group)

        # --- Section 2: Mode & Search Parameters ---
        search_group = QGroupBox("2. Satellite Platform & Search Criteria")
        search_grid = QGridLayout(search_group)
        search_grid.setContentsMargins(10, 10, 10, 10)
        search_grid.setSpacing(8)

        search_grid.addWidget(QLabel("Imagery Mode:"), 0, 0)
        self.combo_mode = QComboBox()
        self.combo_mode.addItems([
            "Single Scene (Best Cloud-Free Scene)",
            "Temporal Median Composite (Multi-Scene Stack)"
        ])
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        search_grid.addWidget(self.combo_mode, 0, 1)

        search_grid.addWidget(QLabel("Date Range:"), 0, 2)
        date_box = QHBoxLayout()
        self.date_start = QDateEdit()
        self.date_start.setCalendarPopup(True)
        self.date_end = QDateEdit()
        self.date_end.setCalendarPopup(True)
        date_box.addWidget(self.date_start)
        date_box.addWidget(QLabel("to"))
        date_box.addWidget(self.date_end)
        search_grid.addLayout(date_box, 0, 3)

        search_grid.addWidget(QLabel("Max Cloud Cover (%):"), 1, 0)
        self.spin_cloud = QDoubleSpinBox()
        self.spin_cloud.setRange(0.0, 100.0)
        self.spin_cloud.setValue(10.0)
        self.spin_cloud.setSingleStep(1.0)
        search_grid.addWidget(self.spin_cloud, 1, 1)

        # Satellite Selection: Exactly Sentinel-2A, Landsat-8, Landsat-9 (Sentinel-2B/2C removed)
        search_grid.addWidget(QLabel("Satellite:"), 1, 2)
        self.combo_platform = QComboBox()
        self.combo_platform.addItems([
            "Sentinel-2A",
            "Landsat-8",
            "Landsat-9",
            "Landsat-8 & Landsat-9 (Combined)"
        ])
        self.combo_platform.currentIndexChanged.connect(self._on_platform_changed)
        search_grid.addWidget(self.combo_platform, 1, 3)

        search_grid.addWidget(QLabel("Min AOI Coverage (%):"), 2, 0)
        self.spin_min_cov = QDoubleSpinBox()
        self.spin_min_cov.setRange(0.0, 100.0)
        self.spin_min_cov.setValue(10.0)
        search_grid.addWidget(self.spin_min_cov, 2, 1)

        search_grid.addWidget(QLabel("Max Scenes:"), 2, 2)
        self.spin_max_scenes = QSpinBox()
        self.spin_max_scenes.setRange(1, 150)
        self.spin_max_scenes.setValue(30)
        search_grid.addWidget(self.spin_max_scenes, 2, 3)

        self.btn_search = QPushButton("🔍 Search Scenes (Earth Search STAC)")
        self.btn_search.setStyleSheet("background-color: #2E86C1; color: white; font-weight: bold; padding: 6px 14px; border-radius: 4px;")
        self.btn_search.clicked.connect(self._run_stac_search)
        search_grid.addWidget(self.btn_search, 3, 2, 1, 2)

        top_layout.addWidget(search_group)
        splitter.addWidget(top_widget)

        # Bottom Control Widget: Table, Tabs for Advanced Configuration, and Run
        bot_widget = QWidget()
        bot_layout = QVBoxLayout(bot_widget)
        bot_layout.setContentsMargins(0, 0, 0, 0)
        bot_layout.setSpacing(8)

        # Tab Widget for Results & Configuration
        self.tabs = QTabWidget()

        # Tab 1: Interactive Results Table
        results_tab = QWidget()
        res_layout = QVBoxLayout(results_tab)
        res_layout.setContentsMargins(6, 6, 6, 6)

        table_header_box = QHBoxLayout()
        self.results_count_lbl = QLabel("No scenes queried yet. Click 'Search Scenes' above.")
        self.results_count_lbl.setStyleSheet("font-weight: bold; color: #5D6D7E;")
        table_header_box.addWidget(self.results_count_lbl)

        self.btn_select_all = QPushButton("Select All")
        self.btn_select_all.clicked.connect(self._select_all_rows)
        self.btn_select_none = QPushButton("Clear Selection")
        self.btn_select_none.clicked.connect(self._clear_selection_rows)
        table_header_box.addWidget(self.btn_select_all)
        table_header_box.addWidget(self.btn_select_none)
        res_layout.addLayout(table_header_box)

        # Splitter between Table and Preview
        table_preview_splitter = QSplitter(Qt.Horizontal)

        self.scene_table = QTableWidget()
        self.scene_table.setColumnCount(8)
        self.scene_table.setHorizontalHeaderLabels([
            "Select", "Scene ID", "Date", "Satellite", "Cloud %", "Tile", "Baseline / Coll", "AOI Cov %"
        ])
        self.scene_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.scene_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.scene_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.scene_table.itemSelectionChanged.connect(self._on_table_row_selected)
        table_preview_splitter.addWidget(self.scene_table)

        # Preview Widget
        self.preview_panel = QWidget()
        self.preview_panel.setMaximumWidth(280)
        p_layout = QVBoxLayout(self.preview_panel)
        p_layout.setContentsMargins(4, 4, 4, 4)
        p_layout.addWidget(QLabel("<b>Scene Quick Preview:</b>"))
        self.preview_img_lbl = QLabel("Select a scene to view thumbnail")
        self.preview_img_lbl.setAlignment(Qt.AlignCenter)
        self.preview_img_lbl.setStyleSheet("background: #EAEDED; border: 1px solid #BDC3C7; min-height: 200px;")
        p_layout.addWidget(self.preview_img_lbl)
        self.preview_info_lbl = QLabel("")
        self.preview_info_lbl.setWordWrap(True)
        self.preview_info_lbl.setStyleSheet("font-size: 10px; color: #34495E;")
        p_layout.addWidget(self.preview_info_lbl)
        p_layout.addStretch()
        table_preview_splitter.addWidget(self.preview_panel)

        res_layout.addWidget(table_preview_splitter)
        self.tabs.addTab(results_tab, "📋 Search Results & Footprints")

        # Tab 2: Spectral & Water Configuration
        config_tab = QWidget()
        cfg_layout = QGridLayout(config_tab)
        cfg_layout.setContentsMargins(10, 10, 10, 10)
        cfg_layout.setSpacing(10)

        # Resolution
        cfg_layout.addWidget(QLabel("Target Spatial Resolution:"), 0, 0)
        self.combo_target_res = QComboBox()
        self.combo_target_res.addItems([
            "Auto / Optimal (10m for S2A, 30m for Landsat)",
            "10 m",
            "20 m",
            "30 m"
        ])
        cfg_layout.addWidget(self.combo_target_res, 0, 1)

        # Water Masking
        self.cb_water_only = QCheckBox("Enable Water-Only Output (Exclude Land Pixels to NoData)")
        self.cb_water_only.setStyleSheet("font-weight: bold; color: #117864;")
        self.cb_water_only.setChecked(True)
        cfg_layout.addWidget(self.cb_water_only, 1, 0, 1, 2)

        cfg_layout.addWidget(QLabel("Water Masking Method:"), 2, 0)
        self.combo_water_method = QComboBox()
        self.combo_water_method.addItems([
            "Smart Hybrid (SCL/QA + NDWI/MNDWI + Cloud Mask) [Recommended]",
            "SCL / QA_PIXEL Water Classification Only",
            "NDWI Spectral Mask (Green - NIR)",
            "MNDWI Spectral Mask (Green - SWIR1)"
        ])
        cfg_layout.addWidget(self.combo_water_method, 2, 1)

        cfg_layout.addWidget(QLabel("Masking Strictness:"), 3, 0)
        self.combo_water_mode = QComboBox()
        self.combo_water_mode.addItems([
            "Permissive (Preserves Turbid Coastal & Shallow Water) [Recommended]",
            "Strict (High-Consensus Only)"
        ])
        cfg_layout.addWidget(self.combo_water_mode, 3, 1)

        cfg_layout.addWidget(QLabel("Temporal Water Occurrence Threshold (%):"), 4, 0)
        self.spin_water_occ = QDoubleSpinBox()
        self.spin_water_occ.setRange(1.0, 100.0)
        self.spin_water_occ.setValue(50.0)
        cfg_layout.addWidget(self.spin_water_occ, 4, 1)

        cfg_layout.addWidget(QLabel("Quality Rasters:"), 5, 0)
        self.cb_quality_rasters = QCheckBox("Export Valid Observation Count, Temp Std, and Water Occurrence Rasters")
        self.cb_quality_rasters.setChecked(True)
        cfg_layout.addWidget(self.cb_quality_rasters, 5, 1)

        cfg_layout.setRowStretch(6, 1)
        self.tabs.addTab(config_tab, "⚙️ Spectral & Water Settings")

        bot_layout.addWidget(self.tabs)

        # Output Destination Box (Requirement 16: Directory Only)
        out_box = QHBoxLayout()
        out_box.addWidget(QLabel("Output Directory:"))
        self.out_dir_edit = QLineEdit(os.path.join(os.path.expanduser("~"), "Bathymetrix_Satellite_Downloads"))
        self.btn_out_browse = QPushButton("Browse...")
        self.btn_out_browse.clicked.connect(self._browse_output_dir)
        out_box.addWidget(self.out_dir_edit)
        out_box.addWidget(self.btn_out_browse)
        bot_layout.addLayout(out_box)

        # Progress bar & Status
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        bot_layout.addWidget(self.progress_bar)

        self.status_lbl = QLabel("Ready.")
        self.status_lbl.setStyleSheet("color: #7F8C8D; font-size: 11px;")
        bot_layout.addWidget(self.status_lbl)

        # Action Buttons (Run & Cancel)
        action_box = QHBoxLayout()
        action_box.addStretch()

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_active_task)
        action_box.addWidget(self.btn_cancel)

        self.btn_run = QPushButton("🚀 Run Download & Processing")
        self.btn_run.setStyleSheet("background-color: #117864; color: white; font-weight: bold; padding: 8px 20px; font-size: 13px; border-radius: 4px;")
        self.btn_run.clicked.connect(self._run_processing)
        action_box.addWidget(self.btn_run)

        bot_layout.addLayout(action_box)
        splitter.addWidget(bot_widget)

        main_layout.addWidget(splitter)

    def _populate_initial_dates(self):
        today = QDate.currentDate()
        self.date_end.setDate(today)
        self.date_start.setDate(today.addDays(-30))

    def _on_aoi_mode_changed(self, btn):
        mode_id = self.aoi_btn_group.id(btn)
        self.layer_combo.setEnabled(mode_id in [0, 1])
        self.cb_selected_only.setEnabled(mode_id in [0, 1])
        self.btn_draw.setEnabled(mode_id == 3)
        self.file_edit.setEnabled(mode_id == 4)
        self.btn_file_browse.setEnabled(mode_id == 4)

        if mode_id == 2:
            self.aoi_status_lbl.setText("AOI: Current Map Canvas Extent selected.")
        elif mode_id == 0:
            self.aoi_status_lbl.setText("AOI: Selected Polygon Vector Layer.")
        elif mode_id == 1:
            self.aoi_status_lbl.setText("AOI: Layer Extent.")
        elif mode_id == 3:
            self.aoi_status_lbl.setText("AOI: Click 'Draw Rectangle on Canvas' to delineate area.")
        elif mode_id == 4:
            self.aoi_status_lbl.setText("AOI: File selection.")

    def _on_mode_changed(self, idx):
        if idx == 0:
            self.spin_water_occ.setEnabled(False)
        else:
            self.spin_water_occ.setEnabled(True)

    def _on_platform_changed(self, idx):
        # Hint resolution based on sensor
        if idx in [1, 2, 3]:  # Landsat
            self.combo_target_res.setItemText(0, "Auto / Optimal (30m for Landsat)")
        else:  # Sentinel-2A
            self.combo_target_res.setItemText(0, "Auto / Optimal (10m for Sentinel-2A)")

    def _activate_draw_tool(self):
        if not self.canvas:
            return
        self.draw_tool = QgsMapToolExtent(self.canvas)
        self.draw_tool.extentChanged.connect(self._on_draw_extent_changed)
        self.canvas.setMapTool(self.draw_tool)
        self.status_lbl.setText("Click and drag on the map canvas to draw the Area of Interest...")

    def _on_draw_extent_changed(self, rect: QgsRectangle):
        if not self.canvas:
            return
        canvas_crs = self.canvas.mapSettings().destinationCrs()
        wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
        tr = QgsCoordinateTransform(canvas_crs, wgs84, QgsProject.instance())
        rect_wgs = tr.transformBoundingBox(rect)

        self.current_aoi_geojson = {
            "type": "Polygon",
            "coordinates": [[
                [rect_wgs.xMinimum(), rect_wgs.yMinimum()],
                [rect_wgs.xMaximum(), rect_wgs.yMinimum()],
                [rect_wgs.xMaximum(), rect_wgs.yMaximum()],
                [rect_wgs.xMinimum(), rect_wgs.yMaximum()],
                [rect_wgs.xMinimum(), rect_wgs.yMinimum()]
            ]]
        }
        if self.rubber_band:
            self.rubber_band.setToGeometry(QgsGeometry.fromRect(rect), None)
            self.rubber_band.show()

        self.aoi_status_lbl.setText(
            f"AOI: Custom Rectangle Drawn [{rect_wgs.xMinimum():.3f}, {rect_wgs.yMinimum():.3f}, {rect_wgs.xMaximum():.3f}, {rect_wgs.yMaximum():.3f}]"
        )
        self.status_lbl.setText("AOI bounding box delineated successfully.")
        if self.draw_tool:
            self.canvas.unsetMapTool(self.draw_tool)

    def _browse_aoi_file(self):
        fpath, _ = QFileDialog.getOpenFileName(self, "Select AOI File", "", "Vector Files (*.geojson *.gpkg *.shp *.json)")
        if fpath:
            self.file_edit.setText(fpath)
            self._load_aoi_from_file(fpath)

    def _load_aoi_from_file(self, fpath: str):
        vlayer = QgsVectorLayer(fpath, "AOI_Temp", "ogr")
        if not vlayer.isValid():
            QMessageBox.critical(self, "Error", f"Failed to open AOI file: {fpath}")
            return
        wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
        tr = QgsCoordinateTransform(vlayer.crs(), wgs84, QgsProject.instance())
        geoms = []
        for feat in vlayer.getFeatures():
            g = feat.geometry()
            if g and not g.isEmpty():
                gw = QgsGeometry(g)
                gw.transform(tr)
                geoms.append(gw)
        if geoms:
            comb = QgsGeometry.unaryUnion(geoms) if len(geoms) > 1 else geoms[0]
            self.current_aoi_geojson = json.loads(comb.asJson())
            self.aoi_status_lbl.setText(f"AOI: File loaded ({os.path.basename(fpath)}, {len(geoms)} features)")

    def _browse_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.out_dir_edit.text())
        if d:
            self.out_dir_edit.setText(d)

    def _get_active_aoi_geojson(self) -> dict:
        mode_id = self.aoi_btn_group.checkedId()
        wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")

        if mode_id == 2:  # Map Canvas
            if not self.canvas:
                raise ValueError("Map canvas is not available.")
            ext = self.canvas.extent()
            tr = QgsCoordinateTransform(self.canvas.mapSettings().destinationCrs(), wgs84, QgsProject.instance())
            ext_w = tr.transformBoundingBox(ext)
            return {
                "type": "Polygon",
                "coordinates": [[
                    [ext_w.xMinimum(), ext_w.yMinimum()],
                    [ext_w.xMaximum(), ext_w.yMinimum()],
                    [ext_w.xMaximum(), ext_w.yMaximum()],
                    [ext_w.xMinimum(), ext_w.yMaximum()],
                    [ext_w.xMinimum(), ext_w.yMinimum()]
                ]]
            }

        elif mode_id == 0:  # Polygon Layer
            layer = self.layer_combo.currentLayer()
            if not layer:
                raise ValueError("Please select a polygon layer.")
            tr = QgsCoordinateTransform(layer.crs(), wgs84, QgsProject.instance())
            feats = layer.selectedFeatures() if self.cb_selected_only.isChecked() else layer.getFeatures()
            geoms = []
            for feat in feats:
                g = feat.geometry()
                if g and not g.isEmpty():
                    gw = QgsGeometry(g)
                    gw.transform(tr)
                    geoms.append(gw)
            if not geoms:
                raise ValueError("Selected layer has no valid polygon features.")
            comb = QgsGeometry.unaryUnion(geoms) if len(geoms) > 1 else geoms[0]
            return json.loads(comb.asJson())

        elif mode_id == 1:  # Layer Extent
            layer = self.layer_combo.currentLayer()
            if not layer:
                raise ValueError("Please select a layer to retrieve extent.")
            tr = QgsCoordinateTransform(layer.crs(), wgs84, QgsProject.instance())
            ext_w = tr.transformBoundingBox(layer.extent())
            return {
                "type": "Polygon",
                "coordinates": [[
                    [ext_w.xMinimum(), ext_w.yMinimum()],
                    [ext_w.xMaximum(), ext_w.yMinimum()],
                    [ext_w.xMaximum(), ext_w.yMaximum()],
                    [ext_w.xMinimum(), ext_w.yMaximum()],
                    [ext_w.xMinimum(), ext_w.yMinimum()]
                ]]
            }

        elif mode_id in [3, 4]:  # Draw or File
            if not self.current_aoi_geojson:
                raise ValueError("Please draw or specify an AOI first.")
            return self.current_aoi_geojson

        raise ValueError("Unknown AOI mode.")

    def _run_stac_search(self):
        try:
            aoi_geo = self._get_active_aoi_geojson()
        except Exception as e:
            QMessageBox.warning(self, "AOI Warning", str(e))
            return

        self.btn_search.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_lbl.setText("Querying Element 84 Earth Search STAC API...")

        plat_map = {
            0: "sentinel-2a",
            1: "landsat-8",
            2: "landsat-9",
            3: "landsat"
        }
        selected_plat = plat_map.get(self.combo_platform.currentIndex(), "sentinel-2a")

        search_params = {
            "aoi_geometry": aoi_geo,
            "start_date": self.date_start.date().toString("yyyy-MM-dd"),
            "end_date": self.date_end.date().toString("yyyy-MM-dd"),
            "max_cloud_cover": self.spin_cloud.value(),
            "platform": selected_plat,
            "min_aoi_coverage": self.spin_min_cov.value(),
            "max_scenes": self.spin_max_scenes.value()
        }

        self.active_task = BackgroundSearchTask(search_params)
        self.active_task.search_completed.connect(self._on_search_completed)
        self.btn_cancel.setEnabled(True)
        QgsApplication.taskManager().addTask(self.active_task)

    def _on_search_completed(self, scenes: list, error_msg: str):
        self.btn_search.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.active_task = None

        if error_msg:
            self.status_lbl.setText(f"Search failed: {error_msg}")
            QMessageBox.critical(self, "Search Error", f"STAC Query Error: {error_msg}")
            return

        # Double check that no Sentinel-2B or 2C scenes enter table
        valid_scenes = []
        for s in scenes:
            p = str(s.get("platform", "")).lower()
            if p in DISALLOWED_PLATFORMS or "s2b" in s["id"].lower() or "s2c" in s["id"].lower():
                continue
            if p in ALLOWED_PLATFORMS:
                valid_scenes.append(s)

        self.found_scenes = valid_scenes
        self._populate_results_table(valid_scenes)
        self.status_lbl.setText(f"Found {len(valid_scenes)} scenes matching criteria.")
        self.results_count_lbl.setText(f"Found {len(valid_scenes)} scenes on Earth Search (Collection 1 Level-2A / Landsat C2 L2):")

    def _populate_results_table(self, scenes: list):
        self.scene_table.setRowCount(len(scenes))
        is_single = (self.combo_mode.currentIndex() == 0)

        for row, sc in enumerate(scenes):
            cb_item = QTableWidgetItem()
            cb_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            cb_item.setCheckState(Qt.Checked if (row == 0 or not is_single) else Qt.Unchecked)
            self.scene_table.setItem(row, 0, cb_item)

            self.scene_table.setItem(row, 1, QTableWidgetItem(sc.get("id", "")))
            self.scene_table.setItem(row, 2, QTableWidgetItem(sc.get("date", "")))
            self.scene_table.setItem(row, 3, QTableWidgetItem(sc.get("satellite", sc.get("platform", ""))))
            self.scene_table.setItem(row, 4, QTableWidgetItem(f"{sc.get('cloud_cover', 0):.1f}%"))
            self.scene_table.setItem(row, 5, QTableWidgetItem(sc.get("mgrs_tile", "")))
            self.scene_table.setItem(row, 6, QTableWidgetItem(str(sc.get("processing_baseline", ""))))
            self.scene_table.setItem(row, 7, QTableWidgetItem(f"{sc.get('aoi_coverage_pct', 0):.1f}%"))

        if scenes:
            self.scene_table.selectRow(0)

    def _on_table_row_selected(self):
        sel = self.scene_table.selectedItems()
        if not sel:
            return
        row = self.scene_table.currentRow()
        if row < 0 or row >= len(self.found_scenes):
            return

        scene = self.found_scenes[row]

        # 1. Update Preview Panel
        preview_url = scene.get("preview_url")
        self.preview_info_lbl.setText(
            f"<b>ID:</b> {scene['id']}<br>"
            f"<b>Satellite:</b> {scene.get('satellite', scene.get('platform'))}<br>"
            f"<b>Date:</b> {scene['date']}<br>"
            f"<b>Tile:</b> {scene.get('mgrs_tile', '')}<br>"
            f"<b>Cloud:</b> {scene['cloud_cover']}%<br>"
            f"<b>Assets:</b> {len(scene.get('assets', {}))} bands"
        )
        if preview_url:
            self.preview_img_lbl.setText("Loading preview...")
            try:
                req = urllib.request.Request(preview_url, headers={"User-Agent": "Bathymetrix-AI"})
                with urllib.request.urlopen(req, timeout=8) as r:
                    data = r.read()
                    pix = QPixmap()
                    pix.loadFromData(data)
                    self.preview_img_lbl.setPixmap(pix.scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            except Exception:
                self.preview_img_lbl.setText("Preview unavailable")
        else:
            self.preview_img_lbl.setText("No preview link")

        # 2. Draw Footprint on QGIS Map Canvas
        if self.footprint_band and self.canvas and "geometry" in scene:
            geom_json = scene["geometry"]
            try:
                qgeom = QgsGeometry.fromGeoJson(json.dumps(geom_json))
                wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
                canvas_crs = self.canvas.mapSettings().destinationCrs()
                tr = QgsCoordinateTransform(wgs84, canvas_crs, QgsProject.instance())
                qgeom.transform(tr)
                self.footprint_band.setToGeometry(qgeom, None)
                self.footprint_band.show()
            except Exception:
                pass

    def _select_all_rows(self):
        for r in range(self.scene_table.rowCount()):
            item = self.scene_table.item(r, 0)
            if item:
                item.setCheckState(Qt.Checked)

    def _clear_selection_rows(self):
        for r in range(self.scene_table.rowCount()):
            item = self.scene_table.item(r, 0)
            if item:
                item.setCheckState(Qt.Unchecked)

    def _cancel_active_task(self):
        if self.active_task:
            self.active_task.cancel()
            self.status_lbl.setText("Canceling task...")
            self.btn_cancel.setEnabled(False)

    def _run_processing(self):
        try:
            aoi_geo = self._get_active_aoi_geojson()
        except Exception as e:
            QMessageBox.warning(self, "AOI Warning", str(e))
            return

        checked_scenes = []
        for r in range(self.scene_table.rowCount()):
            item = self.scene_table.item(r, 0)
            if item and item.checkState() == Qt.Checked:
                checked_scenes.append(self.found_scenes[r])

        if not checked_scenes:
            QMessageBox.warning(self, "No Scenes Selected", "Please select at least one scene from the table.")
            return

        out_dir = self.out_dir_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "Output Directory", "Please choose an output directory.")
            return
        os.makedirs(out_dir, exist_ok=True)

        is_single = (self.combo_mode.currentIndex() == 0)

        # Resolution handling
        res_idx = self.combo_target_res.currentIndex()
        if res_idx == 0:
            target_res = None  # Processor will pick sensor-optimal default
        elif res_idx == 1:
            target_res = 10.0
        elif res_idx == 2:
            target_res = 20.0
        else:
            target_res = 30.0

        water_only = self.cb_water_only.isChecked()
        method_map = {0: "hybrid", 1: "scl", 2: "ndwi", 3: "mndwi"}
        water_method = method_map.get(self.combo_water_method.currentIndex(), "hybrid")
        water_mode = "permissive" if self.combo_water_mode.currentIndex() == 0 else "strict"
        water_occ = self.spin_water_occ.value()
        save_ancillary = self.cb_quality_rasters.isChecked()

        # Disable Run, enable Cancel
        self.btn_run.setEnabled(False)
        self.btn_search.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(5)

        if is_single:
            target_scene = checked_scenes[0]
            sat_title = target_scene.get("satellite", "Satellite")
            self.status_lbl.setText(f"Starting Single Scene Processing: {target_scene['id']} ({sat_title})...")
            task_kwargs = {
                "scene_meta": target_scene,
                "aoi_geojson": aoi_geo,
                "output_dir": out_dir,
                "pixel_size": target_res,
                "water_only": water_only,
                "water_method": water_method,
                "water_mode": water_mode,
                "save_water_mask_raster": save_ancillary
            }
            self.active_task = BackgroundProcessingTask("single", task_kwargs)
        else:
            # Check sensor compatibility for composite
            has_s2a = any(s.get("platform") == "sentinel-2a" for s in checked_scenes)
            has_landsat = any(s.get("platform") in ["landsat-8", "landsat-9"] for s in checked_scenes)

            if has_s2a and has_landsat:
                self.btn_run.setEnabled(True)
                self.btn_search.setEnabled(True)
                self.btn_cancel.setEnabled(False)
                self.progress_bar.setVisible(False)
                QMessageBox.warning(
                    self,
                    "Sensor Harmonization Limit",
                    "Sentinel-2A and Landsat-8/9 cannot be combined in the same median composite automatically.\n\n"
                    "Cross-sensor harmonization is outside the scope of this update.\n"
                    "Please select either Sentinel-2A scenes or Landsat scenes for compositing."
                )
                return

            sat_label = "Landsat" if has_landsat else "Sentinel-2A"
            self.status_lbl.setText(f"Starting Temporal Composite across {len(checked_scenes)} {sat_label} scenes...")
            task_kwargs = {
                "scenes_meta": checked_scenes,
                "aoi_geojson": aoi_geo,
                "output_dir": out_dir,
                "pixel_size": target_res,
                "water_only": water_only,
                "water_temporal_policy": "occurrence_threshold",
                "water_occurrence_threshold": water_occ,
                "save_valid_count_raster": save_ancillary,
                "save_temp_std_raster": save_ancillary,
                "save_water_occurrence_raster": save_ancillary
            }
            self.active_task = BackgroundProcessingTask("composite", task_kwargs)

        self.active_task.status_updated.connect(lambda s: self.status_lbl.setText(s))
        self.active_task.task_done.connect(self._on_processing_finished)
        QgsApplication.taskManager().addTask(self.active_task)

    def _on_processing_finished(self, results: dict, error_msg: str):
        self.btn_run.setEnabled(True)
        self.btn_search.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.active_task = None

        if error_msg:
            self.status_lbl.setText(f"Processing failed: {error_msg}")
            QMessageBox.critical(self, "Processing Error", f"Processing Error: {error_msg}")
            return

        primary = results.get("surface_reflectance") or results.get("composite_reflectance")
        sat_name = results.get("satellite", "Satellite")
        self.status_lbl.setText("Processing finished successfully!")

        if primary and QgsProject.instance():
            lyr_name = os.path.splitext(os.path.basename(primary))[0]
            rlayer = QgsRasterLayer(primary, lyr_name)
            if rlayer.isValid():
                QgsProject.instance().addMapLayer(rlayer)

        QMessageBox.information(
            self,
            "Success",
            f"{sat_name} processing completed successfully!\n\n"
            f"Primary GeoTIFF:\n{primary}\n\n"
            f"The layer has been added to your QGIS Map Canvas."
        )

    def closeEvent(self, event):
        if self.rubber_band:
            self.rubber_band.reset()
        if self.footprint_band:
            self.footprint_band.reset()
        event.accept()
