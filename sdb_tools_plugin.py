import importlib.util
from qgis.core import QgsApplication
from qgis.PyQt.QtWidgets import QMessageBox

class SdbToolsPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.provider = None
        self.dependencies_met = True

    def check_dependencies(self):
        required_modules = {
            'numpy': 'numpy',
            'pandas': 'pandas',
            'rasterio': 'rasterio',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'sklearn': 'scikit-learn>=1.5.0',
            'scipy': 'scipy',
            'joblib': 'joblib',
            'skopt': 'scikit-optimize',
            'sliderule': 'sliderule',
            'icepyx': 'icepyx',
            'geopandas': 'geopandas',
            'pyarrow': 'pyarrow',
            'netCDF4': 'netCDF4'
        }
        missing = []
        for mod, pip_name in required_modules.items():
            if importlib.util.find_spec(mod) is None:
                missing.append(pip_name)
        return missing

    def initGui(self):
        missing = self.check_dependencies()
        if missing:
            self.dependencies_met = False
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setWindowTitle("Missing Dependencies")
            text = "<b>Bathymetrix-AI requires additional Python libraries.</b><br><br>"
            text += "Please open the <b>OSGeo4W Shell</b> as Administrator and run the following command:<br><br>"
            text += f"<code>pip install {' '.join(missing)}</code><br><br>"
            text += "After installation, please restart QGIS."
            msg.setText(text)
            try:
                msg.exec()
            except AttributeError:
                msg.exec_()
            return

        from .qgis_interface.provider import SdbProvider
        self.provider = SdbProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def unload(self):
        if self.provider and self.dependencies_met:
            QgsApplication.processingRegistry().removeProvider(self.provider)
