import csv
import json
from typing import Iterable

import numpy as np


def curve_rows_with_gaps(omega: np.ndarray, re: np.ndarray, im: np.ndarray) -> list[dict]:
    rows = []
    for o, r, i in zip(np.asarray(omega), np.asarray(re), np.asarray(im)):
        rows.append({
            "omega": None if not np.isfinite(o) else float(o),
            "re": None if not np.isfinite(r) else float(r),
            "im": None if not np.isfinite(i) else float(i),
        })
    return rows



def finite_curve_rows(omega: np.ndarray, re: np.ndarray, im: np.ndarray) -> list[dict]:
    omega = np.asarray(omega, dtype=float)
    re = np.asarray(re, dtype=float)
    im = np.asarray(im, dtype=float)
    mask = np.isfinite(omega) & np.isfinite(re) & np.isfinite(im)
    return [
        {"omega": float(o), "re": float(r), "im": float(i)}
        for o, r, i in zip(omega[mask], re[mask], im[mask])
    ]



def curve_summary(omega: np.ndarray, re: np.ndarray, im: np.ndarray, *, include_total_count: bool) -> dict:
    omega = np.asarray(omega, dtype=float)
    re = np.asarray(re, dtype=float)
    im = np.asarray(im, dtype=float)
    mask = np.isfinite(omega) & np.isfinite(re) & np.isfinite(im)

    count_key = "finite_point_count" if include_total_count else "point_count"
    summary = {
        count_key: int(mask.sum()),
        "omega_min": None,
        "omega_max": None,
        "re_min": None,
        "re_max": None,
        "im_min": None,
        "im_max": None,
    }
    if include_total_count:
        summary["total_point_count"] = int(omega.size)

    if not np.any(mask):
        return summary

    oo = omega[mask]
    rr = re[mask]
    ii = im[mask]
    summary.update({
        "omega_min": float(np.min(oo)),
        "omega_max": float(np.max(oo)),
        "re_min": float(np.min(rr)),
        "re_max": float(np.max(rr)),
        "im_min": float(np.min(ii)),
        "im_max": float(np.max(ii)),
    })
    return summary



def export_analysis_data(data: dict, filename: str, file_format: str) -> None:
    if file_format == "json":
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["# export_schema_version", data["export_schema_version"]])
        writer.writerow(["# analysis_type", data["analysis_type"]])
        writer.writerow(["# preset_name", data["preset_name"]])

        for key, value in data.get("model_info", {}).items():
            writer.writerow([f"# model_info.{key}", value])

        numerics = data.get("numerics", {})
        for key, value in numerics.items():
            if key in ("invalid_reason_counts", "numerics_metadata"):
                continue
            writer.writerow([f"# numerics.{key}", value])

        for key, value in numerics.get("invalid_reason_counts", {}).items():
            writer.writerow([f"# numerics.invalid_reason_counts.{key}", value])

        for key, value in numerics.get("numerics_metadata", {}).items():
            writer.writerow([f"# numerics.metadata.{key}", value])

        writer.writerow([])
        writer.writerow(["# curve_summary"])
        for key, value in data.get("curve_summary", {}).items():
            writer.writerow([key, value])

        writer.writerow([])
        writer.writerow(["# curve"])
        writer.writerow(["omega", "re", "im"])
        for row in data.get("curve", []):
            writer.writerow([row.get("omega"), row.get("re"), row.get("im")])

        writer.writerow([])
        writer.writerow(["# special_points_im0"])
        writer.writerow(["omega", "re", "im", "frequency"])
        for p in data.get("special_points", {}).get("im0_points", []):
            writer.writerow([p.get("omega"), p.get("re"), p.get("im"), p.get("frequency")])

        critical = data.get("special_points", {}).get("critical_point")
        if critical:
            writer.writerow([])
            writer.writerow(["# critical_point"])
            writer.writerow(["omega", "re", "im", "frequency"])
            writer.writerow([
                critical.get("omega"),
                critical.get("re"),
                critical.get("im"),
                critical.get("frequency"),
            ])
