# test/test_transverse_regression.py
import numpy as np


def test_transverse_returns_finite_w(model):
    """
    Регрессия: W(p) должен быть конечным (inf/-inf недопустимы),
    NaN в норме не ожидаем после фильтрации.
    """
    params = dict(
        E=2.1e11,
        rho=7800.0,
        length=2.7,
        mu=0.6,
        tau=0.1,
        R=0.04,
        r=0.035,
        K_cut=6e5,
        beta=0.3,
    )

    res = model.calculate_transverse(params)

    Wre = np.asarray(res["W_real"], dtype=float)
    Wim = np.asarray(res["W_imag"], dtype=float)

    assert Wre.size > 0 and Wim.size > 0
    assert np.all(np.isfinite(Wre))
    assert np.all(np.isfinite(Wim))

    # sanity: параметры должны быть конечны
    assert np.isfinite(res["alpha"])
    assert np.isfinite(res["gamma"])
    assert np.isfinite(res["phi_L"])

def test_torsional_crosses_real_axis(model):
    """
    Проверяем, что существует ω,
    при котором Im(σ) меняет знак.
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

    im_sigma = np.asarray(res["sigma_imag"], dtype=float)

    sign_changes = np.where(np.diff(np.sign(im_sigma)) != 0)[0]

    print("Sign changes in Im(σ):", len(sign_changes))

    assert len(sign_changes) > 0

def test_transverse_intersection_diagnostics(model):
    """
    Диагностика точки Im(W)=0.
    Это граница устойчивости поперечных колебаний.
    """

    params = dict(
        E=2.1e11,
        rho=7800.0,
        length=2.7,
        mu=0.6,
        tau=0.1,
        R=0.04,
        r=0.035,
        K_cut=6e5,
        beta=0.3,
        omega_start=0.1,
        omega_end=220.0,
        omega_step=0.1,
    )

    res = model.calculate_transverse(params)

    omega = np.asarray(res["omega"])
    Wre = np.asarray(res["W_real"])
    Wim = np.asarray(res["W_imag"])

    sign_changes = np.where(np.diff(np.sign(Wim)) != 0)[0]

    print("\n===== Поперечные =====")
    print("Количество пересечений Im(W)=0:", len(sign_changes))

    assert len(sign_changes) > 0

    idx = sign_changes[0]
    omega_star = omega[idx]
    re_star = Wre[idx]
    im_star = Wim[idx]

    print("omega* =", omega_star)
    print("Re(W*) =", re_star)
    print("Im(W*) ≈", im_star)
    print("======================")

    assert Wim[idx] * Wim[idx+1] < 0