import numpy as np
import pytest

from app.core.borebar_model import BoreBarModel


def test_model_validation_rejects_nonphysical_direct_calls():
    model = BoreBarModel()

    with pytest.raises(ValueError, match=r"R > r"):
        model.calculate_transverse({
            "E": 2.1e11,
            "rho": 7800.0,
            "length": 2.7,
            "mu": 0.6,
            "tau": 0.1,
            "R": 0.04,
            "r": 0.04,
            "K_cut": 6e5,
            "h": 3.02141544835e-05,
            "omega_start": 0.0,
            "omega_end": 220.0,
            "omega_step": 0.1,
        })

    with pytest.raises(ValueError, match=r"μ≈1"):
        model.calculate_longitudinal({
            "E": 2.0e11,
            "rho": 7800.0,
            "S": 2.0e-4,
            "length": 2.5,
            "mu": 1.0,
            "tau": 0.06,
            "omega_start": 0.001,
            "omega_end": 400.0,
            "omega_step": 0.1,
        })

    with pytest.raises(ValueError, match=r"δ₁"):
        model.calculate_torsional({
            "rho": 7800.0,
            "G": 8.0e10,
            "length": 3.0,
            "delta1": -1.0e-6,
            "multiplier": 1.0,
            "Jr": 2.57e-2,
            "Jp": 1.9e-5,
            "omega_start": 1000.0,
            "omega_end": 15000.0,
            "omega_step": 1.0,
        })


def test_torsional_research_length_growth_reduces_stability_boundary():
    """По тексту исследования с ростом длины борштанги область устойчивости уменьшается."""
    model = BoreBarModel()
    base = {
        "rho": 7800.0,
        "G": 8.0e10,
        "Jr": 2.57e-2,
        "Jp": 1.9e-5,
        "delta1": 3.44e-6,
        "multiplier": 1.0,
        "omega_start": 1000.0,
        "omega_end": 15000.0,
        "omega_step": 1.0,
    }

    critical_re = []
    for length in (2.5, 3.0, 4.0, 5.0, 6.0):
        crit = model.find_torsional_im0_points({**base, "length": length})["critical"]
        assert crit is not None
        critical_re.append(float(crit["re"]))

    assert all(a > b for a, b in zip(critical_re, critical_re[1:]))


def test_torsional_research_internal_damping_moves_boundary_toward_stability():
    """По исследованию увеличение внутреннего трения заметно влияет на область устойчивости."""
    model = BoreBarModel()
    base = {
        "rho": 7800.0,
        "G": 8.0e10,
        "Jr": 2.57e-2,
        "Jp": 1.9e-5,
        "length": 3.0,
        "delta1": 3.44e-6,
        "omega_start": 1000.0,
        "omega_end": 15000.0,
        "omega_step": 1.0,
    }

    critical_re = []
    for multiplier in (1.0, 2.0, 3.0, 4.0, 6.0, 10.0):
        crit = model.find_torsional_im0_points({**base, "multiplier": multiplier})["critical"]
        assert crit is not None
        critical_re.append(float(crit["re"]))

    assert all(a < b for a, b in zip(critical_re, critical_re[1:]))
    assert critical_re[-1] < 0.0


def test_longitudinal_regenerative_delay_zero_crossings_follow_pi_over_tau_grid():
    """Для продольной модели пересечения δ(ω)=0 должны идти по сетке ω≈nπ/τ, пока нет вырождения."""
    model = BoreBarModel()
    params = {
        "E": 2.0e11,
        "rho": 7800.0,
        "S": 2.0e-4,
        "length": 2.5,
        "mu": 0.1,
        "tau": 0.06,
        "omega_start": 0.001,
        "omega_end": 400.0,
        "omega_step": 0.1,
    }

    points = model.find_longitudinal_im0_points(params)["points"]
    assert len(points) >= 6

    tau = params["tau"]
    expected = [n * np.pi / tau for n in range(1, len(points) + 1)]
    actual = [p["omega"] for p in points]

    for act, exp in zip(actual, expected):
        assert act == pytest.approx(exp, abs=0.2)


def test_transverse_maple_reference_mode_preserves_research_parameters_and_root_pattern():
    """Режим project_maple_compatible_phi должен сохранять параметры эталонного Maple-эксперимента."""
    model = BoreBarModel()
    params = {
        "E": 2.1e11,
        "rho": 7800.0,
        "length": 2.7,
        "R": 0.04,
        "r": 0.035,
        "K_cut": 6.0e5,
        "beta": 0.3,
        "mu": 0.6,
        "tau": 0.1,
        "omega_start": 0.0,
        "omega_end": 220.0,
        "omega_step": 0.1,
        "transverse_modal_shape_variant": "project_maple_compatible_phi",
    }

    res = model.calculate_transverse(params)
    im0 = model.find_transverse_im0_points(params)

    assert res["modal_shape_source"] == "project_maple_compatible_phi"
    assert res["beta"] == pytest.approx(0.3)
    assert res["model_variant"] == "galerkin_one_mode_project_shape"

    points = im0["points"]
    critical = im0["critical"]
    assert len(points) >= 8
    assert critical is not None
    assert critical["re"] < 0.0
    assert 150.0 <= critical["omega"] <= 200.0
