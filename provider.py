# <<< تم التعديل: إضافة Qgis إلى سطر الاستيراد
from qgis.core import QgsProcessingProvider, QgsProcessingAlgorithm, QgsMessageLog, Qgis
import os
import inspect
import importlib.util
import sys

# تحديد اسم للرسائل التشخيصية
LOG_TAG = "SDB_Tools_Provider"

class SdbProvider(QgsProcessingProvider):

    def __init__(self):
        """Constructor."""
        super().__init__()
        # طباعة رسالة عند بدء التشغيل
        QgsMessageLog.logMessage("Initializing SdbProvider...", LOG_TAG, level=Qgis.Info)
        self._algorithms = self._load_algorithms()

    def loadAlgorithms(self, *args, **kwargs):
        """Loads all processing algorithms from this provider."""
        for alg in self._algorithms:
            self.addAlgorithm(alg)
            
    def unload(self):
        """Unloads the provider."""
        super().unload()
        self._algorithms = []

    def id(self):
        return 'sdb_tools'

    def name(self):
        return 'Bathymetrix-AI'

    def longName(self):
        return self.name()

    def _load_algorithms(self):
        """Dynamically loads all algorithm classes from .py files."""
        algorithms = []
        alg_folder = os.path.join(os.path.dirname(__file__), 'algorithms')
        
        # طباعة المسار الذي يبحث فيه
        QgsMessageLog.logMessage(f"Searching for algorithms in folder: {alg_folder}", LOG_TAG, level=Qgis.Info)

        if not os.path.isdir(alg_folder):
            QgsMessageLog.logMessage("Algorithm folder not found.", LOG_TAG, level=Qgis.Warning)
            return []

        # طباعة قائمة الملفات التي يجدها
        found_files = [f.name for f in os.scandir(alg_folder) if f.is_file() and f.name.endswith('.py') and f.name != '__init__.py']
        QgsMessageLog.logMessage(f"Found .py files: {found_files}", LOG_TAG, level=Qgis.Info)

        for f_name in found_files:
            f_path = os.path.join(alg_folder, f_name)
            module_name = f'sdb_tools.algorithms.{f_name[:-3]}'
            
            if module_name in sys.modules:
                del sys.modules[module_name]

            try:
                # طباعة اسم الملف الذي يحاول تحميله الآن
                QgsMessageLog.logMessage(f"  --> Attempting to load module from: {f_name}", LOG_TAG, level=Qgis.Info)
                spec = importlib.util.spec_from_file_location(module_name, f_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                loaded_from_this_file = False
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, QgsProcessingAlgorithm) and obj is not QgsProcessingAlgorithm:
                        algorithms.append(obj())
                        loaded_from_this_file = True
                        # طباعة اسم الكلاس الذي نجح في تحميله
                        QgsMessageLog.logMessage(f"      SUCCESS: Loaded algorithm class '{name}'", LOG_TAG, level=Qgis.Success)
                
                if not loaded_from_this_file:
                    QgsMessageLog.logMessage(f"      WARNING: No algorithm class found in {f_name}", LOG_TAG, level=Qgis.Warning)

            except Exception as e:
                # طباعة أي خطأ يحدث أثناء التحميل
                QgsMessageLog.logMessage(f"      ERROR loading algorithm from {f_name}: {e}", LOG_TAG, level=Qgis.Critical)
        
        QgsMessageLog.logMessage(f"Total algorithms loaded: {len(algorithms)}", LOG_TAG, level=Qgis.Info)
        return algorithms