import numpy as np


def test_longitudinal_zero_frequency_limits_are_correct(model):
    params = dict(E=2.1e11, rho=7800.0, S=2e-4, length=3.0, tau=0.06, mu=0.6)
    res = model.calculate_longitudinal(params)

    K1_0_ref = (params['E'] * params['S']) / (params['length'] * (1.0 - params['mu']))
    delta_0_ref = -(params['E'] * params['S'] * params['mu'] * params['tau']) / (params['length'] * (1.0 - params['mu']))

    assert res['K1_0'] == K1_0_ref
    assert res['delta_0'] == delta_0_ref


def test_longitudinal_grid_and_output_shapes_are_consistent(model):
    params = dict(
        E=2.1e11,
        rho=7800.0,
        S=2e-4,
        length=3.0,
        tau=0.06,
        mu=0.6,
        omega_start=1e-3,
        omega_end=400.0,
        omega_step=0.1,
    )
    res = model.calculate_longitudinal(params)
    omega = np.asarray(res['omega'], dtype=float)
    K1 = np.asarray(res['K1'], dtype=float)
    delta = np.asarray(res['delta'], dtype=float)

    assert omega.shape == K1.shape == delta.shape
    assert omega[0] == params['omega_start']
    assert omega[-1] == params['omega_end']
    assert np.all(np.diff(omega) > 0)
    assert np.all(np.isfinite(K1) | np.isnan(K1))
    assert np.all(np.isfinite(delta) | np.isnan(delta))


def test_longitudinal_without_regeneration_has_zero_delta_everywhere_valid(model):
    params = dict(
        E=2.0e11,
        rho=7800.0,
        S=2.0e-4,
        length=2.5,
        mu=0.0,
        tau=0.0,
        omega_start=0.001,
        omega_end=400.0,
        omega_step=0.1,
    )
    res = model.calculate_longitudinal(params)
    delta = np.asarray(res['delta'], dtype=float)
    finite = np.isfinite(delta)
    assert finite.any()
    assert np.allclose(delta[finite], 0.0)


def test_longitudinal_im0_points_have_consistent_critical_point(model):
    params = dict(
        E=2.1e11,
        rho=7800.0,
        S=2e-4,
        length=3.0,
        tau=0.06,
        mu=0.6,
        omega_start=1e-3,
        omega_end=400.0,
        omega_step=0.1,
    )
    im0 = model.find_longitudinal_im0_points(params)
    points = im0['points']
    critical = im0['critical']

    assert len(points) > 0
    assert critical is not None
    assert critical['K1'] == min(p['K1'] for p in points)
    for point in points:
        assert point['delta'] == 0.0
        assert np.isfinite(point['omega'])
        assert np.isfinite(point['K1'])


def test_longitudinal_rejects_mu_near_one(model):
    params = dict(E=2.1e11, rho=7800.0, S=2e-4, length=3.0, tau=0.06, mu=1.0)
    import pytest
    with pytest.raises(ValueError, match="μ≈1"):
        model.calculate_longitudinal(params)
