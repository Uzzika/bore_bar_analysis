import csv
import json
from pathlib import Path

import numpy as np
import pytest

from app.core.borebar_model import BoreBarModel
from app.utils.presets import get_torsional_presets


PyQt5 = pytest.importorskip('PyQt5')

from PyQt5.QtWidgets import QFileDialog
from app.ui.longitudinal_page import LongitudinalPage
from app.ui.torsional_page import TorsionalPage
from app.ui.transverse_page import TransversePage


@pytest.mark.parametrize(
    'preset_name, positive_start',
    [
        ('Крутильные — симметричная display-проверка', 2.0),
        ('Крутильные — локальный диапазон около ω*', 0.25),
        ('Крутильные — почти без демпфирования', 0.1),
    ],
)
def test_torsional_negative_requested_range_uses_same_physical_branch_as_positive_only(model, preset_name, positive_start):
    negative_params = dict(get_torsional_presets()[preset_name])
    positive_params = dict(negative_params)
    positive_params['omega_start'] = positive_start

    negative_res = model.calculate_torsional(negative_params)
    positive_res = model.calculate_torsional(positive_params)

    neg_omega = np.asarray(negative_res['physical_omega'], dtype=float)
    pos_omega = np.asarray(positive_res['physical_omega'], dtype=float)
    neg_re = np.asarray(negative_res['physical_sigma_real'], dtype=float)
    pos_re = np.asarray(positive_res['physical_sigma_real'], dtype=float)
    neg_im = np.asarray(negative_res['physical_sigma_imag'], dtype=float)
    pos_im = np.asarray(positive_res['physical_sigma_imag'], dtype=float)

    assert np.all(neg_omega > 0.0), 'физическая ветвь не должна вычисляться на отрицательных частотах'

    step = float(negative_params['omega_step'])
    omega_tol = max(1e-6, step * 1e-4)

    # При построении сетки через np.arange старт из отрицательной области может
    # давать микросмещение положительной части относительно запуска только на ω>0.
    # Это не должно считаться сменой физической ветви, пока расхождение мало
    # относительно шага дискретизации.
    assert neg_omega.shape == pos_omega.shape
    assert np.allclose(neg_omega, pos_omega, rtol=0.0, atol=omega_tol)

    # Значения на почти совпадающих сетках тоже могут немного отличаться,
    # особенно в режимах с очень резким изменением Im(σ) возле малых ω.
    # Смысл проверки — подтвердить ту же физическую ветвь, а не побитово
    # совпадающие значения на каждой точке.
    value_rtol = 1e-5
    value_atol = max(1e-8, omega_tol)
    assert np.allclose(neg_re, pos_re, rtol=value_rtol, atol=value_atol, equal_nan=True)
    assert np.allclose(neg_im, pos_im, rtol=value_rtol, atol=value_atol, equal_nan=True)


@pytest.mark.parametrize(
    'preset_name',
    [
        'Крутильные — симметричная display-проверка',
        'Крутильные — локальный диапазон около ω*',
        'Крутильные — почти без демпфирования',
    ],
)
def test_torsional_negative_display_curve_is_exact_mirror_without_extra_outliers(model, preset_name):
    params = dict(get_torsional_presets()[preset_name])
    result = model.calculate_torsional(params)

    display_omega = np.asarray(result['display_omega'], dtype=float)
    display_re = np.asarray(result['display_sigma_real'], dtype=float)
    display_im = np.asarray(result['display_sigma_imag'], dtype=float)

    neg = display_omega < 0.0
    pos = display_omega > 0.0

    assert np.any(neg) and np.any(pos)
    assert np.array_equal(display_omega[neg], -display_omega[pos][::-1])
    assert np.array_equal(np.isnan(display_re[neg]), np.isnan(display_re[pos][::-1]))
    assert np.array_equal(np.isnan(display_im[neg]), np.isnan(display_im[pos][::-1]))
    assert np.allclose(display_re[neg], display_re[pos][::-1], equal_nan=True)
    assert np.allclose(display_im[neg], -display_im[pos][::-1], equal_nan=True)


@pytest.mark.parametrize(
    'page_cls, expected_type',
    [
        (TorsionalPage, 'torsional'),
        (LongitudinalPage, 'longitudinal'),
        (TransversePage, 'transverse'),
    ],
)
def test_export_results_writes_json_file_for_each_analysis(monkeypatch, qapp, tmp_path, page_cls, expected_type):
    page = page_cls(None)
    out_file = tmp_path / f'{expected_type}.json'

    monkeypatch.setattr(
        QFileDialog,
        'getSaveFileName',
        lambda *args, **kwargs: (str(out_file), 'JSON (*.json)'),
    )

    page.export_results()

    assert out_file.exists()
    payload = json.loads(out_file.read_text(encoding='utf-8'))
    assert payload['analysis_type'] == expected_type
    assert payload['export_schema_version'] == 4
    assert payload['preset_name'] == 'custom'
    point_count = payload['curve_summary'].get('point_count', payload['curve_summary'].get('finite_point_count'))
    assert point_count == len(payload['curve'])
    assert 'invalid_point_count' in payload['numerics']
    assert 'invalid_reason_counts' in payload['numerics']


@pytest.mark.parametrize(
    'page_cls, expected_type',
    [
        (TorsionalPage, 'torsional'),
        (LongitudinalPage, 'longitudinal'),
        (TransversePage, 'transverse'),
    ],
)
def test_export_results_writes_csv_file_with_metadata_for_each_analysis(monkeypatch, qapp, tmp_path, page_cls, expected_type):
    page = page_cls(None)
    out_file = tmp_path / f'{expected_type}.csv'

    monkeypatch.setattr(
        QFileDialog,
        'getSaveFileName',
        lambda *args, **kwargs: (str(out_file), 'CSV (*.csv)'),
    )

    page.export_results()

    assert out_file.exists()
    with out_file.open('r', encoding='utf-8', newline='') as fh:
        rows = list(csv.reader(fh))

    flat = [' | '.join(row) for row in rows if row]
    assert any(f'# analysis_type | {expected_type}' == row for row in flat)
    assert any('# export_schema_version | 4' == row for row in flat)
    assert any('# preset_name | custom' == row for row in flat)
    assert any(row == '# curve' for row in flat)
    assert any('omega | re | im' == row for row in flat)
    assert any(row == '# special_points_im0' for row in flat)
