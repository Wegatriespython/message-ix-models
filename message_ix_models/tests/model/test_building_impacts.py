"""Tests for model.buildings.impacts — building energy CID integration."""

import numpy as np
import pandas as pd

from message_ix_models.model.buildings.impacts import (
    _demand_to_final_energy_iamc,
    predict_building_ei,
    prepare_building_demand,
)

# ---------------------------------------------------------------------------
# predict_building_ei
# ---------------------------------------------------------------------------


def test_warming_increases_cooling_ei():
    gmt = np.array([1.0, 1.5, 2.0])
    ei_cool = predict_building_ei(gmt, "cool")
    mean_per_year = np.nanmean(ei_cool, axis=(0, 1, 2))
    assert mean_per_year[2] > mean_per_year[0]


def test_warming_decreases_heating_ei():
    gmt = np.array([1.0, 1.5, 2.0])
    ei_heat = predict_building_ei(gmt, "heat")
    mean_per_year = np.nanmean(ei_heat, axis=(0, 1, 2))
    assert mean_per_year[2] < mean_per_year[0]


def test_building_ei_selects_gwl_window_response_percentile():
    gmt = np.array([1.5, 2.0])
    low = predict_building_ei(gmt, "cool", percentile=10)
    high = predict_building_ei(gmt, "cool", percentile=90)

    assert np.nanmean(high) > np.nanmean(low)


# ---------------------------------------------------------------------------
# IAMC packaging helpers
# ---------------------------------------------------------------------------


def test_demand_to_final_energy_iamc_converts_gwa_to_ej_per_year():
    demand = pd.DataFrame(
        {
            "node": ["R12_AFR", "R12_AFR"],
            "year": [2030, 2035],
            "value": [1.0, 2.0],
        }
    )

    result = _demand_to_final_energy_iamc(
        demand, "Final Energy|Residential and Commercial|Cooling"
    )

    assert result["unit"].unique().tolist() == ["EJ/yr"]
    assert result["variable"].unique().tolist() == [
        "Final Energy|Residential and Commercial|Cooling"
    ]
    np.testing.assert_allclose(result["value"], [0.0315576, 0.0631152])


# ---------------------------------------------------------------------------
# prepare_building_demand — substitution arithmetic
# ---------------------------------------------------------------------------

# Minimal columns required by prepare_building_demand / _substitute_climate_component.
_RC_SPEC = pd.DataFrame(
    {"node": ["R12_AFR"], "year": [2050], "commodity": ["rc_spec"], "value": [100.0]}
)
_RC_THERM = pd.DataFrame(
    {"node": ["R12_AFR"], "year": [2050], "commodity": ["rc_therm"], "value": [200.0]}
)
_FRACTIONS = pd.DataFrame(
    {
        "node": ["R12_AFR"],
        "year": [2050],
        "frac_resid_cool": [0.2],
        "frac_comm_cool": [0.1],
        "frac_resid_heat": [0.3],
        "frac_comm_heat": [0.1],
    }
)


def test_prepare_building_demand_substitution_arithmetic():
    """new_value = old_value * (1 - frac_total) + cid_value."""
    cooling_cids = pd.DataFrame({"node": ["R12_AFR"], "year": [2050], "value": [15.0]})
    heating_cids = pd.DataFrame({"node": ["R12_AFR"], "year": [2050], "value": [30.0]})

    new_spec, new_therm = prepare_building_demand(
        _RC_SPEC.copy(),
        _RC_THERM.copy(),
        cooling_cids,
        heating_cids,
        fractions=_FRACTIONS,
    )

    # rc_spec: 100 * (1 - 0.3) + 15 = 85.0
    np.testing.assert_allclose(new_spec["value"].values, [85.0])
    # rc_therm: 200 * (1 - 0.4) + 30 = 150.0
    np.testing.assert_allclose(new_therm["value"].values, [150.0])


def test_prepare_building_demand_missing_cid_zerofills():
    """When CID has no rows for a (node, year), its contribution is zero."""
    empty_cids = pd.DataFrame(
        {
            "node": pd.Series(dtype=str),
            "year": pd.Series(dtype=int),
            "value": pd.Series(dtype=float),
        }
    )

    new_spec, new_therm = prepare_building_demand(
        _RC_SPEC.copy(),
        _RC_THERM.copy(),
        empty_cids,
        empty_cids,
        fractions=_FRACTIONS,
    )

    # rc_spec: 100 * (1 - 0.3) + 0 = 70.0
    np.testing.assert_allclose(new_spec["value"].values, [70.0])
    # rc_therm: 200 * (1 - 0.4) + 0 = 120.0
    np.testing.assert_allclose(new_therm["value"].values, [120.0])


# ---------------------------------------------------------------------------
# Scenario data directory
# ---------------------------------------------------------------------------


def _gmt():
    from message_ix_models.tools.impacts import GmtArray

    years = np.arange(2020, 2101, 5)
    return GmtArray(np.linspace(1.1, 2.6, len(years)), years)


def test_scenario_data_rejects_unknown_files(tmp_path):
    import pytest

    from message_ix_models.model.buildings.impacts import check_scenario_data

    with pytest.raises(FileNotFoundError):
        check_scenario_data(tmp_path / "absent")
    with pytest.raises(ValueError, match="no data files"):
        check_scenario_data(tmp_path)
    (tmp_path / "region_EI_cool_gwl_binned_passive.nc").write_text("")
    with pytest.raises(ValueError, match="Unexpected files"):
        check_scenario_data(tmp_path)


def test_scenario_data_replaces_only_what_it_holds(tmp_path):
    """Packaged copies reproduce the default; a scaled gamma table scales its part."""
    import shutil

    from message_ix_models.model.buildings.impacts import (
        _buildings_data_path,
        compute_building_cids,
    )
    from message_ix_models.util import package_data_path

    years = [2030, 2050, 2100]
    cool0, heat0 = compute_building_cids(_gmt(), years, "SSP2")

    same = tmp_path / "same"
    same.mkdir()
    shutil.copy(
        package_data_path("impacts", "rime", "region_EI_cool_gwl_binned.nc"), same
    )
    name = "correction_coefficients_cool_SSP2_resid.csv"
    shutil.copy(_buildings_data_path("correction_coefficients", name), same)
    cool1, heat1 = compute_building_cids(_gmt(), years, "SSP2", data_dir=same)
    pd.testing.assert_frame_equal(cool0, cool1)
    pd.testing.assert_frame_equal(heat0, heat1)

    double = tmp_path / "double"
    double.mkdir()
    cc = pd.read_csv(_buildings_data_path("correction_coefficients", name), comment="#")
    cc["correction_coeff"] *= 2
    cc.to_csv(double / name, index=False)
    cool2, heat2 = compute_building_cids(_gmt(), years, "SSP2", data_dir=double)
    pd.testing.assert_frame_equal(heat0, heat2)
    ratio = (cool2["value"] / cool0["value"]).dropna()
    # Residential doubles, commercial does not: every node rises, none doubles past 2x.
    assert (ratio > 1).all() and (ratio <= 2 + 1e-9).all()
