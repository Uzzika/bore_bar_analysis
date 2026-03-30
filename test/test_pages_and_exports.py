import pytest

PyQt5 = pytest.importorskip('PyQt5')

from app.main_window import MainWindow
from app.ui.torsional_page import TorsionalPage
from app.ui.longitudinal_page import LongitudinalPage
from app.ui.transverse_page import TransversePage


def test_main_window_registers_all_pages(qapp):
    window = MainWindow()
    assert window.stack.count() == 6
    assert window.stack.currentWidget() is window.menu

    window.switch(window.torsional)
    assert window.stack.currentWidget() is window.torsional
    window.switch(window.longitudinal)
    assert window.stack.currentWidget() is window.longitudinal
    window.switch(window.transverse)
    assert window.stack.currentWidget() is window.transverse


def test_torsional_page_build_export_data_contains_unified_schema(qapp):
    page = TorsionalPage(None)
    params = page.get_parameters()
    data = page._build_export_data(params)

    assert data['export_schema_version'] == 4
    assert data['analysis_type'] == 'torsional'
    assert data['model_info']['source_curve_for_special_points'] == 'physical_positive_branch'
    assert data['numerics']['curve_saved_kind'] == 'physical_only'
    assert set(data['curve'][0]) == {'omega', 're', 'im'}
    

def test_longitudinal_page_preset_application_and_export_schema(qapp):
    page = LongitudinalPage(None)
    preset_name = next(name for name in page.presets if 'умеренная регенерация' in name)
    page.apply_preset(preset_name)
    params = page.get_parameters()
    data = page._build_export_data(params)

    assert page.current_preset_name == preset_name
    assert float(page.mu_input.text()) == params['mu']
    assert data['analysis_type'] == 'longitudinal'
    assert data['model_info']['model_variant'] == 'si_wave_speed'
    assert data['model_info']['research_alignment_status'] == 'si_interpretation_of_research_formulas'
    assert 'SI' in data['model_info']['model_regime_label']
    assert data['numerics']['curve_saved_kind'] == 'direct_curve'
    assert data['special_points']['research_critical_point'] is not None
    assert data['curve_summary']['point_count'] == len(data['curve'])


def test_transverse_page_export_contains_modal_metadata(qapp):
    page = TransversePage(None)
    params = page.get_parameters()
    data = page._build_export_data(params)

    assert data['analysis_type'] == 'transverse'
    assert data['model_info']['model_variant'] == 'galerkin_one_mode_verified_shape'
    assert data['model_info']['modal_shape_source'] == 'verified_cantilever_first_mode_phi'
    assert data['model_info']['shape_normalization'] == 'phi(L)=1'
    assert data['model_info']['beta'] == pytest.approx(data['model_info']['h'] * data['model_info']['gamma'])
    assert data['special_points']['research_critical_point'] is not None

def test_transverse_page_summary_mentions_computed_beta(qapp):
    page = TransversePage(None)
    result = page.model.calculate_transverse(page.get_parameters())
    page._update_result_summary(result)
    text = page.results_label.text()

    assert 'β = h·γ' in text
    assert 'Источник диссипации' in text
