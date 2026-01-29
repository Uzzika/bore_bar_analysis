# test/conftest.py
import sys
from pathlib import Path

# Добавляем корень проекта (папку, где лежит borebar_model.py) в sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../bore_bar_analysis
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
from borebar_model import BoreBarModel  # теперь найдётся


@pytest.fixture
def model():
    # у тебя модель статическая — можно возвращать класс
    return BoreBarModel


def assert_finite_array(arr, *, allow_nan=False):
    arr = np.asarray(arr, dtype=float)
    if allow_nan:
        ok = np.isfinite(arr) | np.isnan(arr)
    else:
        ok = np.isfinite(arr)
    assert ok.all()
