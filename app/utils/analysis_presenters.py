import numpy as np

from app.utils.export_utils import curve_rows_with_gaps, curve_summary, finite_curve_rows


def format_nonzero_reason_counts(reason_counts: dict) -> str:
    items = []
    for key, value in (reason_counts or {}).items():
        try:
            ivalue = int(value)
        except Exception:
            continue
        if ivalue > 0:
            items.append(f"{key}={ivalue}")
    return ", ".join(items) if items else "нет"


# ---------------------- export builders ----------------------

def build_torsional_export_data(*, params: dict, preset_name: str, result: dict, im0: dict) -> dict:
    research_critical = im0.get("research_critical_point")

    curve_omega = np.asarray(result["physical_omega"], dtype=float)
    curve_re = np.asarray(result["physical_sigma_real"], dtype=float)
    curve_im = np.asarray(result["physical_sigma_imag"], dtype=float)

    return {
        "export_schema_version": 4,
        "analysis_type": "torsional",
        "preset_name": preset_name or "custom",
        "params": params,
        "model_info": {
            "model_variant": result.get("model_variant", "torsional_physical_positive_plus_model_display_symmetry"),
            "curve_semantics": "curve stores the physical torsional branch: omega, Re(sigma), Im(sigma)",
            "delta1_effective": float(result.get("delta1_effective", params["delta1"] * params.get("multiplier", 1.0))),
            "negative_frequency_policy": result.get(
                "negative_frequency_policy",
                "display_curve_is_built_as_conjugate_mirror_of_positive_branch",
            ),
            "source_curve_for_special_points": im0.get("source_curve", "physical_positive_branch"),
            "critical_point_semantics": "research_critical_point_first_im0_in_research_window",
            "minimum_re_point_semantics": "diagnostic_only_not_used_as_primary_critical_point",
        },
        "numerics": {
            "solver_variant": "torsional_direct_curve_sampling_with_im0_detection",
            "export_variant": "compact_unified_v4",
            "omega_step": float(params["omega_step"]),
            "invalid_point_count": int(result.get("invalid_point_count", 0)),
            "invalid_reason_counts": dict(result.get("invalid_reason_counts", {})),
            "numerics_metadata": dict(result.get("numerics_metadata", {})),
            "curve_saved_kind": "physical_only",
        },
        "curve_summary": curve_summary(curve_omega, curve_re, curve_im, include_total_count=True),
        "special_points": {
            "im0_points": im0.get("points", []),
            "critical_point": research_critical,
            "research_critical_point": research_critical,
            "minimum_re_critical_point": im0.get("minimum_re_critical_point"),
            "critical_selection_policy": im0.get("critical_selection_policy"),
        },
        "curve": curve_rows_with_gaps(curve_omega, curve_re, curve_im),
    }



def build_longitudinal_export_data(*, params: dict, preset_name: str, omega: np.ndarray, result: dict, im0: dict) -> dict:
    K1 = np.asarray(result["K1"], dtype=float)
    delta = np.asarray(result["delta"], dtype=float)
    curve_rows = finite_curve_rows(omega, K1, delta)
    research_critical = im0.get("research_critical_point") or im0.get("critical")

    return {
        "export_schema_version": 4,
        "analysis_type": "longitudinal",
        "preset_name": preset_name or "custom",
        "params": params,
        "model_info": {
            "model_variant": result.get("model_variant", "si_wave_speed"),
            "model_regime": result.get("longitudinal_model_regime"),
            "model_regime_label": result.get("longitudinal_model_regime_label"),
            "model_scope": result.get("longitudinal_model_scope"),
            "research_alignment_status": result.get("research_alignment_status"),
            "model_note": result.get("longitudinal_model_note"),
            "interpretation_note": result.get(
                "interpretation_note",
                "Текущий продольный режим является интерпретацией исследовательской постановки.",
            ),
            "curve_semantics": "curve stores omega, K1(omega), delta(omega)",
            "curve_parameterization": result.get("curve_parameterization", "omega -> (K1(omega), delta(omega))"),
            "wave_speed_a": float(result.get("a", 0.0)),
            "x_definition": "omega * L / a",
            "zero_frequency_limit_policy": result.get("zero_frequency_limit_policy"),
            "source_curve_for_special_points": im0.get("source_curve", "direct_longitudinal_curve"),
        },
        "numerics": {
            "solver_variant": "longitudinal_direct_curve_sampling_with_zero_crossing_detection",
            "export_variant": "compact_unified_v4",
            "omega_step": float(params["omega_step"]),
            "invalid_point_count": int(result.get("invalid_point_count", 0)),
            "invalid_reason_counts": dict(result.get("invalid_reason_counts", {})),
            "numerics_metadata": dict(result.get("numerics_metadata", {})),
            "curve_saved_kind": "direct_curve",
        },
        "curve_summary": curve_summary(omega, K1, delta, include_total_count=False),
        "special_points": {
            "im0_points": im0.get("points", []),
            "critical_point": research_critical,
            "research_critical_point": research_critical,
            "minimum_re_critical_point": im0.get("minimum_re_critical_point"),
            "critical_selection_policy": im0.get("critical_selection_policy"),
        },
        "curve": curve_rows,
    }



def build_transverse_export_data(*, params: dict, preset_name: str, result: dict, im0: dict) -> dict:
    curve_omega = np.asarray(result["omega"], dtype=float)
    curve_re = np.asarray(result["W_real"], dtype=float)
    curve_im = np.asarray(result["W_imag"], dtype=float)
    research_critical = im0.get("research_critical_point") or im0.get("critical")

    return {
        "export_schema_version": 4,
        "analysis_type": "transverse",
        "preset_name": preset_name or "custom",
        "params": params,
        "model_info": {
            "model_variant": result.get("model_variant", "galerkin_one_mode_unknown"),
            "curve_semantics": "curve stores omega, Re(W), Im(W) on the full physical grid; invalid points are kept as null/NaN gaps",
            "transverse_model": "Galerkin one-mode model",
            "transverse_model_regime": result.get("transverse_model_regime"),
            "transverse_model_regime_label": result.get("transverse_model_regime_label"),
            "transverse_model_scope": result.get("transverse_model_scope"),
            "transverse_model_note": result.get("transverse_model_note"),
            "research_alignment_status": result.get("research_alignment_status"),
            "interpretation_note": result.get(
                "interpretation_note",
                "Проектная одномодовая модель в виде верифицированной первой собственной формы консольной балки.",
            ),
            "modal_shape_variant": result.get("modal_shape_variant"),
            "modal_shape_source": result.get("modal_shape_source"),
            "modal_shape_description": result.get("modal_shape_description"),
            "shape_normalization": result.get("shape_normalization"),
            "shape_scale_C": float(result["shape_scale_C"]),
            "k1": float(result["k1"]),
            "lambda1": float(result["lambda1"]),
            "shape_eta": result.get("shape_eta"),
            "alpha": float(result["alpha"]),
            "beta": float(result["beta"]),
            "gamma": float(result["gamma"]),
            "h": float(result["h"]),
            "damping_source": result["damping_source"],
            "modal_mass_integral": float(result["modal_mass_integral"]),
            "modal_curvature_integral": float(result["modal_curvature_integral"]),
        },
        "numerics": {
            "solver_variant": "transverse_direct_curve_sampling_with_zero_crossing_detection",
            "export_variant": "compact_unified_v4",
            "omega_step": float(params["omega_step"]),
            "invalid_point_count": int(result.get("invalid_point_count", 0)),
            "invalid_reason_counts": dict(result.get("invalid_reason_counts", {})),
            "numerics_metadata": dict(result.get("numerics_metadata", {})),
            "curve_saved_kind": "full_curve_with_nan_gaps",
        },
        "curve_summary": curve_summary(curve_omega, curve_re, curve_im, include_total_count=True),
        "special_points": {
            "im0_points": im0.get("points", []),
            "critical_point": research_critical,
            "research_critical_point": research_critical,
            "minimum_re_critical_point": im0.get("minimum_re_critical_point"),
            "critical_selection_policy": im0.get("critical_selection_policy"),
        },
        "curve": curve_rows_with_gaps(curve_omega, curve_re, curve_im),
    }


# ---------------------- summary builders ----------------------

def build_torsional_summary_text(*, result: dict, critical: dict | None, im0: dict | None = None, plot_curve: dict | None = None, elapsed_seconds: float | None = None) -> str:
    lines = [
        "Крутильная модель",
        "",
        "Ключевые параметры:",
        f"δ₁,эфф = {float(result.get('delta1_effective', float('nan'))):.6g}",
    ]
    if im0 is not None:
        points = im0.get('points', []) or []
        lines += [
            "",
            f"Найдено точек пересечения с осью Im(σ)=0: {len(points)}",
        ]

    invalid_count = int(result.get('invalid_point_count', 0))
    lines += [
        "",
        f"Отбраковано точек: {invalid_count}",
        f"Причины: {format_nonzero_reason_counts(result.get('invalid_reason_counts', {}))}",
    ]
    if plot_curve is not None:
        lines.append(f"Обрезано display-сегментов у начала координат: {int(plot_curve.get('clipped_count', 0))}")
    if elapsed_seconds is not None:
        lines.append(f"Время расчёта и построения графика: {elapsed_seconds:.3f} с")
    return "\n".join(lines)



def build_longitudinal_summary_text(*, result: dict, im0: dict | None = None, elapsed_seconds: float | None = None) -> str:
    lines = [
        "Продольная модель",
        "",
        "Ключевые параметры:",
        f"a = √(E/ρ) = {float(result.get('a', float('nan'))):.6g} м/с",
        f"K₁(0) = {float(result.get('K1_0', float('nan'))):.6g}",
        f"δ(0) = {float(result.get('delta_0', float('nan'))):.6g}",
        f"ω₁ ≈ πa/L = {float(result.get('omega_main', float('nan'))):.6g} рад/с",
    ]

    if im0 is not None:
        points = im0.get('points', []) or []
        lines += [
            "",
            f"Найдено точек пересечения с осью δ(ω)=0: {len(points)}",
        ]

    invalid_count = int(result.get('invalid_point_count', 0))
    lines += [
        "",
        f"Отбраковано точек: {invalid_count}",
        f"Причины: {format_nonzero_reason_counts(result.get('invalid_reason_counts', {}))}",
    ]
    if elapsed_seconds is not None:
        lines.append(f"Время расчёта и построения графика: {elapsed_seconds:.3f} с")
    return "\n".join(lines)



def build_transverse_summary_text(*, result: dict, im0: dict | None = None, display_curve: dict | None = None, elapsed_seconds: float | None = None) -> str:
    if display_curve is None:
        omega = np.asarray(result.get('omega', []), dtype=float)
        display_curve = {'display_point_count': int(omega.size), 'base_point_count': int(omega.size)}

    lines = [
        "Поперечная модель",
        "",
        "Ключевые параметры:",
        f"α = {float(result.get('alpha', float('nan'))):.6g}",
        f"β = {float(result.get('beta', float('nan'))):.6g}",
        f"γ = {float(result.get('gamma', float('nan'))):.6g}",
        f"h = {float(result.get('h', float('nan'))):.6g} с",
        f"β = h·γ = {float(result.get('h', float('nan'))) * float(result.get('gamma', float('nan'))):.6g}",
    ]

    if im0 is not None:
        points = im0.get('points', []) or []
        lines += [
            "",
            f"Найдено точек пересечения с осью Im(W)=0: {len(points)}",
        ]

    invalid_count = int(result.get('invalid_point_count', 0))
    lines += [
        "",
        f"Отбраковано точек: {invalid_count}",
        f"Причины: {format_nonzero_reason_counts(result.get('invalid_reason_counts', {}))}",
    ]
    base_n = int(display_curve.get('base_point_count', 0))
    disp_n = int(display_curve.get('display_point_count', 0))
    ## if base_n > 0:
    ##     lines.append(f"Точек на физической сетке / display-сетке: {base_n} / {disp_n}")
    if elapsed_seconds is not None:
        lines.append(f"Время расчёта и построения графика: {elapsed_seconds:.3f} с")
    return "\n".join(lines)
