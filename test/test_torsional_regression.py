import pytest
import numpy as np

from app.core.borebar_model import BoreBarModel
from app.utils.presets import get_torsional_presets


def _coth(z):
    return np.cosh(z) / np.sinh(z)


def _sigma_torsional_reference(omega, *, rho, G, Jp, Jr, delta1, length, multiplier):
    p = 1j * omega
    lam1 = np.sqrt(rho * G) * Jp / Jr
    lam2 = length * np.sqrt(rho / G)
    d1 = delta1 * multiplier
    expr = np.sqrt(1.0 + d1 * p)
    arg = lam2 * p / expr
    return -p - lam1 * expr * _coth(arg)


def test_torsional_sigma_matches_formula_at_control_nodes(model):
    params = dict(
        rho=7800.0,
        G=8.0e10,
        Jr=2.57e-2,
        Jp=1.9e-5,
        delta1=3.44e-6,
        multiplier=1.0,
        length=3.0,
        omega_start=1000.0,
        omega_end=15000.0,
        omega_step=1.0,
    )

    res = model.calculate_torsional(params)
    omega = np.asarray(res['omega'], dtype=float)
    sigma = np.asarray(res['sigma_real'], dtype=float) + 1j * np.asarray(res['sigma_imag'], dtype=float)
    by_omega = {float(w): s for w, s in zip(omega, sigma)}

    for w in (1000.0, 5000.0, 10000.0, 14999.0):
        s_model = by_omega[w]
        s_ref = _sigma_torsional_reference(w, **{k: params[k] for k in ('rho', 'G', 'Jp', 'Jr', 'delta1', 'length', 'multiplier')})
        assert np.isfinite(s_model.real) and np.isfinite(s_model.imag)
        assert np.isclose(s_model.real, s_ref.real, rtol=1e-9, atol=1e-9)
        assert np.isclose(s_model.imag, s_ref.imag, rtol=1e-9, atol=1e-9)


def test_torsional_find_im0_points_matches_sign_change_count(model):
    params = dict(
        rho=7800.0,
        G=8.0e10,
        Jr=2.57e-2,
        Jp=1.9e-5,
        delta1=3.44e-6,
        multiplier=1.0,
        length=3.0,
        omega_start=1000.0,
        omega_end=15000.0,
        omega_step=1.0,
    )

    res = model.calculate_torsional(params)
    omega = np.asarray(res['physical_omega'], dtype=float)
    sigma_re = np.asarray(res['physical_sigma_real'], dtype=float)
    sigma_im = np.asarray(res['physical_sigma_imag'], dtype=float)

    expected_intervals = BoreBarModel._sign_change_intervals(omega, sigma_im)
    im0 = model.find_torsional_im0_points(params)
    points = im0['points']
    critical = im0['critical']

    assert len(points) == len(expected_intervals) > 0
    assert critical is not None
    assert critical['re'] == min(p['re'] for p in points)

    for point, (i, j) in zip(points, expected_intervals):
        assert omega[i] <= point['omega'] <= omega[j]
        assert min(sigma_re[i], sigma_re[j]) <= point['re'] <= max(sigma_re[i], sigma_re[j])
        assert point['frequency'] == pytest.approx(point['omega'] / (2 * np.pi))


def test_torsional_display_curve_is_conjugate_mirror_for_negative_range(model):
    params = get_torsional_presets()['Крутильные — симметричная display-проверка']
    res = model.calculate_torsional(params)

    display_omega = np.asarray(res['display_omega'], dtype=float)
    display_re = np.asarray(res['display_sigma_real'], dtype=float)
    display_im = np.asarray(res['display_sigma_imag'], dtype=float)

    assert np.any(display_omega < 0)
    assert np.any(display_omega > 0)

    neg_mask = display_omega < 0
    pos_mask = display_omega > 0

    omega_neg = display_omega[neg_mask]
    omega_pos = display_omega[pos_mask]
    re_neg = display_re[neg_mask]
    re_pos = display_re[pos_mask]
    im_neg = display_im[neg_mask]
    im_pos = display_im[pos_mask]

    assert np.allclose(omega_neg, -omega_pos[::-1])
    assert np.allclose(re_neg, re_pos[::-1], equal_nan=True)
    assert np.allclose(im_neg, -im_pos[::-1], equal_nan=True)
    assert res['negative_frequency_policy'] == 'display_curve_is_built_as_conjugate_mirror_of_positive_branch'
