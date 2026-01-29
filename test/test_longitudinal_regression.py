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
        S=2.0,        # ВАЖНО: это именно то, что ты подаёшь в модель сейчас
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
    """
    Регрессия: продольный расчёт возвращает массивы согласованной формы,
    без inf/-inf. NaN допускаются (это механизм разрывов).
    """
    params = dict(
        E=2.1e11,
        rho=7800.0,
        S=2.0,
        length=3.0,
        tau=0.06,
        mu=0.6,
        omega_max_longitudinal=0.5,
        omega_points_longitudinal=4000,
    )

    res = model.calculate_longitudinal(params)

    omega = np.asarray(res["omega"], dtype=float)
    K1 = np.asarray(res["K1"], dtype=float)
    delta = np.asarray(res["delta"], dtype=float)

    assert omega.shape == K1.shape == delta.shape
    assert omega.size == params["omega_points_longitudinal"]

    # ω должен быть строго возрастающим
    assert np.all(np.diff(omega) > 0)

    # Inf запрещаем, NaN разрешаем
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
        S=2.0,
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
