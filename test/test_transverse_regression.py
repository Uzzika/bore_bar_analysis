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
