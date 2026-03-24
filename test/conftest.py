import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from borebar_model import BoreBarModel

@pytest.fixture(scope='session')
def qapp():
    qtwidgets = pytest.importorskip('PyQt5.QtWidgets')
    app = qtwidgets.QApplication.instance()
    if app is None:
        app = qtwidgets.QApplication([])
    return app

@pytest.fixture
def model():
    return BoreBarModel()
