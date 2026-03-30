import pytest
import numpy as np


def test_transverse_returns_finite_curve_and_modal_metadata(model):
    params = dict(
        E=2.1e11,
        rho=7800.0,
        length=2.7,
        mu=0.6,
        tau=0.1,
        R=0.04,
        r=0.035,
        K_cut=6e5,
        h=3.02141544835e-05,
        omega_start=0.0,
        omega_end=220.0,
        omega_step=0.1,
    )
    res = model.calculate_transverse(params)

    Wre = np.asarray(res['W_real'], dtype=float)
    Wim = np.asarray(res['W_imag'], dtype=float)
    assert Wre.size > 0
    assert Wim.size > 0
    assert np.all(np.isfinite(Wre))
    assert np.all(np.isfinite(Wim))
    assert res['beta'] == pytest.approx(res['h'] * res['gamma'])
    assert res['modal_shape_source'] == 'verified_cantilever_first_mode_phi'
    assert res['shape_normalization'] == 'phi(L)=1'


def test_transverse_default_h_is_used_when_h_not_provided(model):
    params = dict(
        E=2.1e11,
        rho=7800.0,
        length=2.7,
        mu=0.6,
        tau=0.1,
        R=0.04,
        r=0.035,
        K_cut=6e5,
        omega_start=0.0,
        omega_end=220.0,
        omega_step=0.1,
    )
    res = model.calculate_transverse(params)
    assert res['beta'] == pytest.approx(res['h'] * res['gamma'])
    assert res['modal_shape_source'] == 'verified_cantilever_first_mode_phi'
    assert res['shape_normalization'] == 'phi(L)=1'
    assert res['phi_L'] == pytest.approx(1.0)


def test_transverse_im0_points_have_consistent_critical_point(model):
    params = dict(
        E=2.1e11,
        rho=7800.0,
        length=2.7,
        mu=0.6,
        tau=0.1,
        R=0.04,
        r=0.035,
        K_cut=6e5,
        omega_start=0.1,
        omega_end=220.0,
        omega_step=0.1,
    )
    im0 = model.find_transverse_im0_points(params)
    points = im0['points']
    research_critical = im0['research_critical_point']

    assert len(points) > 0
    assert research_critical is not None
    assert research_critical['re'] == min(p['re'] for p in points)
    assert all(p['im'] == 0.0 for p in points)


def test_transverse_rejects_removed_legacy_variant(model):
    params = dict(
        E=2.1e11,
        rho=7800.0,
        length=2.7,
        mu=0.6,
        tau=0.1,
        R=0.04,
        r=0.035,
        K_cut=6e5,
        h=3.02141544835e-05,
        omega_start=0.0,
        omega_end=220.0,
        omega_step=0.1,
        transverse_modal_shape_variant="project_maple_compatible_phi",
    )
    with pytest.raises(ValueError, match="только verified_cantilever_first_mode_phi"):
        model.calculate_transverse(params)
