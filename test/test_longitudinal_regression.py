# test/test_longitudinal_regression.py
import numpy as np


def test_longitudinal_zero_frequency_limits_are_correct(model):
    """
    Регрессия: справочные пределы ω→0 должны быть ровно по формулам,
    которые возвращает модель (K1_0, delta_0).
    """
    params = dict(
        E=2.1e11,
        rho=7800.0,
        S=2e-4,        # ВАЖНО: это именно то, что ты подаёшь в модель сейчас
        length=3.0,
        tau=0.06,
        mu=0.6,
    )

    res = model.calculate_longitudinal(params)

    E, S, L, mu, tau = params["E"], params["S"], params["length"], params["mu"], params["tau"]

    K1_0_ref = (E * S) / (L * (1.0 - mu))
    delta_0_ref = -(E * S * mu * tau) / (L * (1.0 - mu))

    assert np.isclose(res["K1_0"], K1_0_ref, rtol=0.0, atol=0.0)
    assert np.isclose(res["delta_0"], delta_0_ref, rtol=0.0, atol=0.0)

def test_longitudinal_outputs_have_valid_shape_and_no_infs(model):
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

    omega = np.asarray(res["omega"], dtype=float)
    K1 = np.asarray(res["K1"], dtype=float)
    delta = np.asarray(res["delta"], dtype=float)

    assert omega.shape == K1.shape == delta.shape

    expected_omega = np.arange(
        params["omega_start"],
        params["omega_end"] + params["omega_step"],
        params["omega_step"],
    )

    expected_size = expected_omega.size

    assert omega.size == expected_size

    assert np.all(np.diff(omega) > 0)
    assert np.all(np.isfinite(K1) | np.isnan(K1))
    assert np.all(np.isfinite(delta) | np.isnan(delta))

def test_longitudinal_no_infs_on_working_range(model):
    """
    Регрессия: на диапазоне ω модель не должна выдавать inf/-inf.
    NaN допускаются (они используются как разрывы).
    """
    params = dict(
        E=2.1e11,
        rho=7800.0,
        S=2e-4,
        length=3.0,
        tau=0.06,
        mu=0.6,
        omega_max_longitudinal=400.0,
        omega_points_longitudinal=12000,
    )

    res = model.calculate_longitudinal(params)
    K1 = np.asarray(res["K1"], dtype=float)
    delta = np.asarray(res["delta"], dtype=float)

    assert np.all(np.isfinite(K1) | np.isnan(K1))
    assert np.all(np.isfinite(delta) | np.isnan(delta))

def test_torsional_leftmost_point(model):
    """
    Проверяем, что кривая уходит влево
    и достигает значений порядка -100...-1000 (как в исследовании).
    """
    params = dict(
        rho=7800.0,
        G=8.0e10,
        Jr=2.57e-2,
        Jp=1.9e-5,
        delta1=3.44e-6,
        multiplier=1,
        length=3.0,
        omega_start=1000.0,
        omega_end=15000.0,
        omega_step=1.0,
    )

    res = model.calculate_torsional(params)
    re_sigma = np.asarray(res["sigma_real"], dtype=float)

    re_min = np.nanmin(re_sigma)

    print("\nLeftmost Re =", re_min)

    # ожидаем масштаб как в исследовании
    assert re_min < -50.0


def test_longitudinal_intersection_diagnostics(model):
    """
    Диагностика точки δ(ω)=0.
    Это граница устойчивости продольных колебаний.
    """

    params = dict(
        E=2.1e11,
        rho=7800.0,
        S=2e-4,   # важно! не 2.0
        length=3.0,
        tau=0.06,
        mu=0.6,
        omega_start=1e-3,
        omega_end=400.0,
        omega_step=0.1,
    )

    res = model.calculate_longitudinal(params)

    omega = np.asarray(res["omega"])
    delta = np.asarray(res["delta"])
    K1 = np.asarray(res["K1"])

    # ищем смену знака δ
    sign_changes = np.where(np.diff(np.sign(delta)) != 0)[0]

    print("\n===== Продольные =====")
    print("Количество пересечений δ=0:", len(sign_changes))

    assert len(sign_changes) > 0

    # берём первое пересечение
    idx = sign_changes[0]
    omega_star = omega[idx]
    K1_star = K1[idx]
    delta_star = delta[idx]

    print("omega* =", omega_star)
    print("K1* =", K1_star)
    print("delta* ≈", delta_star)
    print("======================")

    assert abs(delta_star) < 1e3  # допускаем близость к 0