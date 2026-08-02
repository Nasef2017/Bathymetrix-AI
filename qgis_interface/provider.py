from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon
import os

# --- منطقة الاستيراد (IMPORTS) ---


# 1. Master Flow
from .algorithms.SDB_MasterFlow import SDBMasterOrchestrator

# 2. Pre-processing
from .algorithms.SDB_01_Preprocessing import SDBPhase1Preprocessing

# 3. Data Filtering
from .algorithms.SDB_02_Data_Filtering import SDBModule02

# 4. Initial Modeling
from .algorithms.SDB_03_Initial_Modeling import SDBModule03

# 5. Spatial Retraining
from .algorithms.SDB_04_Spatial_Retraining import SDBPhase4Adaptive

# 6. Final Reporting
from .algorithms.SDB_05_Final_Reporting import SDBModule05

# 7. Evaluate Model (Removed)

# 8. ICESAT
from .algorithms.ICESAT_V25 import SlideRuleFinalTool

# 9. Tidal Datum
from .algorithms.Tidal_Datum_Converter import TidalDatumConverter

# 10. Temporal Intelligence (Multi-Year System)
from .algorithms.SDB_Temporal_Intelligence import SDBTemporalIntelligence


class SdbProvider(QgsProcessingProvider):

    def __init__(self):
        """Constructor."""
        super().__init__()

    def loadAlgorithms(self, *args, **kwargs):
        """
        هنا يتم تسجيل الأدوات يدوياً.
        لو حدث خطأ هنا، سيعطيك بايثون رسالة واضحة بدلاً من إخفاء الأداة.
        """
        self.addAlgorithm(SDBMasterOrchestrator())
        self.addAlgorithm(SDBPhase1Preprocessing())
        self.addAlgorithm(SDBModule02())
        self.addAlgorithm(SDBModule03())
        self.addAlgorithm(SDBPhase4Adaptive())
        self.addAlgorithm(SDBModule05())

        self.addAlgorithm(SlideRuleFinalTool())
        self.addAlgorithm(TidalDatumConverter())
        self.addAlgorithm(SDBTemporalIntelligence())

    def id(self):
        return "sdb_tools"

    def name(self):
        return "Bathymetrix-AI"

    def icon(self):
        return QIcon(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "icon.png")
        )

    def longName(self):
        return self.name()
