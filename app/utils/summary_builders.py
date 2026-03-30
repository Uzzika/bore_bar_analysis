import numpy as np


def build_torsional_summary(result: dict, critical: dict | None) -> str:
    d1_eff = float(result.get("delta1_effective", np.nan))
    invalid_counts = dict(result.get("invalid_reason_counts", {}))
    lines = [f"Эффективное демпфирование δ₁,эфф = {d1_eff:.6g} с"]
    if critical:
        lines.extend([
            "Исследовательская критическая точка:",
            f"ω* = {critical['omega']:.6g} рад/с",
            f"f* = {critical['frequency']:.6g} Гц",
            f"Re σ* = {critical['re']:.6g}",
        ])
    lines.extend([
        f"Отбраковано точек: {int(result.get('invalid_point_count', 0))}",
        "Причины: "
        f"arg_nonfinite={int(invalid_counts.get('arg_nonfinite', 0))}, "
        f"arg_too_small={int(invalid_counts.get('arg_too_small', 0))}, "
        f"near_coth_pole={int(invalid_counts.get('near_coth_pole', 0))}, "
        f"sigma_nonfinite={int(invalid_counts.get('sigma_nonfinite', 0))}, "
        f"sigma_clip={int(invalid_counts.get('sigma_clip', 0))}",
    ])
    return "\n".join(lines)



def build_longitudinal_summary(result: dict) -> str:
    invalid_counts = dict(result.get("invalid_reason_counts", {}))
    return "\n".join([
        "Продольная модель:",
        f"Режим: {result.get('longitudinal_model_regime_label', 'SI-интерпретация исследовательской постановки')}",
        f"Статус относительно исследования: {result.get('research_alignment_status', 'si_interpretation_of_research_formulas')}",
        f"Назначение режима: {result.get('longitudinal_model_note', 'Физически согласованная SI-реализация формул K₁–δ.')}",
        f"a = √(E/ρ) = {result['a']:.6g} м/с",
        f"K₁(0+) = {result['K1_0']:.6g}",
        f"δ(0+) = {result['delta_0']:.6g}",
        f"ω₁ ≈ πa/L = {result['omega_main']:.6g} рад/с",
        f"Отбраковано точек: {int(result.get('invalid_point_count', 0))}",
        "Причины: "
        f"omega_nonfinite={int(invalid_counts.get('omega_nonfinite', 0))}, "
        f"cot_singularity={int(invalid_counts.get('cot_singularity', 0))}, "
        f"denominator_too_small={int(invalid_counts.get('denominator_too_small', 0))}, "
        f"response_clip={int(invalid_counts.get('response_clip', 0))}, "
        f"response_nonfinite={int(invalid_counts.get('response_nonfinite', 0))}",
    ])



def build_transverse_summary(result: dict, display_curve: dict) -> str:
    invalid_counts = dict(result.get("invalid_reason_counts", {}))
    return "\n".join([
        "Модель поперечных колебаний:",
        f"α = {result['alpha']:.6g}",
        f"γ = {result['gamma']:.6g}",
        f"h = {result['h']:.6g} c",
        f"β = h·γ = {result['beta']:.6g}",
        f"Источник диссипации: {result['damping_source']}",
        f"Форма φ(x): {result.get('modal_shape_source', 'unknown')}",
        f"Интерпретация режима: {result.get('modal_shape_description', 'unknown')}",
        f"Нормировка: {result.get('shape_normalization', 'unknown')}",
        f"Отбраковано точек: {int(result.get('invalid_point_count', 0))}",
        "Причины: "
        f"omega_nonfinite={int(invalid_counts.get('omega_nonfinite', 0))}, "
        f"denom_too_small={int(invalid_counts.get('denom_too_small', 0))}, "
        f"response_nonfinite={int(invalid_counts.get('response_nonfinite', 0))}, "
        f"response_clip={int(invalid_counts.get('response_clip', 0))}",
        f"Точек для отображения: {int(display_curve.get('display_point_count', 0))} "
        f"(базовая сетка: {int(display_curve.get('base_point_count', 0))})",
    ])
