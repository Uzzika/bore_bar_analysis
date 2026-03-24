"""Типовые пресеты по видам колебаний.

Пресеты ниже нужны не только для красивых демонстраций, а для покрытия
критически важных режимов каждой модели:
- базовый рабочий режим;
- режимы, где проявляется симметрия/асимметрия представления;
- слабое и сильное демпфирование;
- влияние длины борштанги;
- регенеративные и почти-сингулярные случаи.

Замечание:
- крутильная модель физически считается на положительной ветви ω>0, но часть
  пресетов специально оставляет симметричный диапазон [-ω, +ω] для проверки
  корректного display-представления;
- пресеты содержат полные наборы параметров страниц, чтобы их можно было
  использовать как единый источник типовых конфигураций.
"""


def _torsional_base() -> dict:
    return {
        "rho": 7800.0,
        "G": 8.0e10,
        "length": 3.0,
        "delta1": 3.44e-6,
        "multiplier": 1.0,
        "Jr": 2.57e-2,
        "Jp": 1.9e-5,
        "omega_start": 1000.0,
        "omega_end": 15000.0,
        "omega_step": 1.0,
    }



def get_torsional_presets() -> dict:
    base = _torsional_base()
    return {
        "Крутильные — физическая положительная ветвь": {
            **base,
            "omega_start": 1000.0,
            "omega_end": 15000.0,
            "omega_step": 1.0,
        },
        "Крутильные — симметричная display-проверка": {
            **base,
            "omega_start": -15000.0,
            "omega_end": 15000.0,
            "omega_step": 2.0,
        },
        "Крутильные — локальный диапазон около ω*": {
            **base,
            "omega_start": -2500.0,
            "omega_end": 2500.0,
            "omega_step": 0.25,
        },
        "Крутильные — почти без демпфирования": {
            **base,
            "delta1": 1.0e-8,
            "multiplier": 1.0,
            "omega_start": -20000.0,
            "omega_end": 20000.0,
            "omega_step": 0.1,
        },
        "Крутильные — усиленное внутреннее демпфирование": {
            **base,
            "delta1": 8.0e-6,
            "multiplier": 3.0,
            "omega_start": -15000.0,
            "omega_end": 15000.0,
            "omega_step": 2.0,
        },
        "Крутильные — длинная борштанга": {
            **base,
            "length": 6.0,
            "multiplier": 2.0,
            "omega_start": 500.0,
            "omega_end": 15000.0,
            "omega_step": 1.0,
        },
        "Крутильные — повышенная инерция головки": {
            **base,
            "Jr": 4.5e-2,
            "omega_start": -12000.0,
            "omega_end": 12000.0,
            "omega_step": 2.0,
        },
    }



def _longitudinal_base() -> dict:
    return {
        "E": 2.0e11,
        "rho": 7800.0,
        "S": 2.0e-4,
        "length": 2.5,
        "mu": 0.10,
        "tau": 60e-3,
        "omega_start": 0.001,
        "omega_end": 400.0,
        "omega_step": 0.1,
    }



def get_longitudinal_presets() -> dict:
    base = _longitudinal_base()
    return {
        "Продольные — базовый рабочий режим": {
            **base,
        },
        "Продольные — без регенерации (μ=0)": {
            **base,
            "mu": 0.0,
            "tau": 0.0,
        },
        "Продольные — умеренная регенерация": {
            **base,
            "length": 3.0,
            "mu": 0.45,
            "tau": 0.05,
            "omega_end": 500.0,
        },
        "Продольные — почти критическая связь": {
            **base,
            "length": 3.0,
            "mu": 0.80,
            "tau": 0.03,
            "omega_end": 500.0,
        },
        "Продольные — около сингулярности знаменателя": {
            **base,
            "mu": 0.95,
            "tau": 0.02,
            "omega_end": 600.0,
            "omega_step": 0.05,
        },
        "Продольные — длинная борштанга": {
            **base,
            "length": 5.0,
            "mu": 0.50,
            "tau": 0.05,
            "omega_end": 300.0,
        },
        "Продольные — малая площадь сечения": {
            **base,
            "S": 1.0e-4,
            "mu": 0.35,
            "tau": 0.06,
            "omega_end": 500.0,
        },
    }



def _transverse_base() -> dict:
    return {
        "E": 2.1e11,
        "rho": 7800.0,
        "length": 2.7,
        "R": 0.04,
        "r": 0.035,
        "K_cut": 6.0e5,
        "h": 3.0214154483500606e-05,
        "mu": 0.6,
        "tau": 0.1,
        "omega_start": 0.0,
        "omega_end": 220.0,
        "omega_step": 0.1,
    }



def get_transverse_presets() -> dict:
    base = _transverse_base()
    return {
        "Поперечные — базовый рабочий режим": {
            **base,
            "transverse_modal_shape_variant": "verified_cantilever_first_mode_phi",
        },
        "Поперечные — без регенерации": {
            **base,
            "mu": 0.0,
            "tau": 0.0,
            "omega_start": -220.0,
            "omega_end": 220.0,
            "omega_step": 0.1,
            "transverse_modal_shape_variant": "verified_cantilever_first_mode_phi",
        },
        "Поперечные — симметричный диапазон для годографа": {
            **base,
            "omega_start": -250.0,
            "omega_end": 250.0,
            "omega_step": 0.2,
            "transverse_modal_shape_variant": "verified_cantilever_first_mode_phi",
        },
        "Поперечные — сильное внутреннее трение": {
            **base,
            "h": 6.042830896700121e-05,
            "mu": 0.0,
            "tau": 0.0,
            "omega_start": -300.0,
            "omega_end": 300.0,
            "omega_step": 0.5,
            "transverse_modal_shape_variant": "verified_cantilever_first_mode_phi",
        },
        "Поперечные — длинная борштанга": {
            **base,
            "length": 4.0,
            "h": 9.824243697322762e-05,
            "mu": 0.0,
            "tau": 0.0,
            "omega_start": -250.0,
            "omega_end": 250.0,
            "omega_step": 0.5,
            "transverse_modal_shape_variant": "verified_cantilever_first_mode_phi",
        },
        "Поперечные — тонкостенная геометрия": {
            **base,
            "R": 0.04,
            "r": 0.038,
            "mu": 0.5,
            "tau": 0.08,
            "omega_end": 260.0,
            "omega_step": 0.1,
            "transverse_modal_shape_variant": "verified_cantilever_first_mode_phi",
        },
        "Поперечные — усиленная жёсткость резания": {
            **base,
            "K_cut": 1.0e6,
            "mu": 0.7,
            "tau": 0.1,
            "omega_end": 260.0,
            "omega_step": 0.1,
            "transverse_modal_shape_variant": "verified_cantilever_first_mode_phi",
        },
        "Поперечные — project approximation (режим совместимости)": {
            **base,
            "transverse_modal_shape_variant": "project_maple_compatible_phi",
        },
    }



def get_presets(kind: str | None = None) -> dict:
    presets_by_kind = {
        "torsional": get_torsional_presets(),
        "longitudinal": get_longitudinal_presets(),
        "transverse": get_transverse_presets(),
    }
    if kind is None:
        merged = {}
        for group in presets_by_kind.values():
            merged.update(group)
        return merged
    if kind not in presets_by_kind:
        raise ValueError(f"Неизвестный тип пресетов: {kind}")
    return presets_by_kind[kind]
