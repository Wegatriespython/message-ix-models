"""Tests for model.buildings.impacts — building energy CID integration."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from message_ix_models.model.buildings.impacts import (
    _ei_to_dataframe,
    _mfh_weighted_ei,
    compute_building_cids,
    load_correction_coefficients,
    load_floor_areas,
    load_sector_fractions,
    load_theta,
    predict_building_ei,
)
from message_ix_models.tools.impacts import impacts_data_path


class TestDataLoading:
    def test_correction_coefficients_resid(self):
        df = load_correction_coefficients("cool", "S1", "resid")
        expected = {
            "region",
            "arch",
            "urt",
            "year",
            "correction_coeff",
            "floor_Mm2",
        }
        assert expected <= set(df.columns)
        assert len(df) > 0
        # Regions are short codes (no R12_ prefix)
        assert df["region"].str.startswith("R12_").sum() == 0

    def test_correction_coefficients_comm(self):
        df = load_correction_coefficients("cool", "S1", "comm")
        # Commercial archetypes start with comm_
        assert df["arch"].str.startswith("comm_").all()

    def test_sector_fractions(self):
        df = load_sector_fractions()
        assert {"node", "year", "frac_resid_cool", "frac_comm_cool"} <= set(df.columns)
        # Fractions should be in [0, 1]
        for col in ("frac_resid_cool", "frac_comm_cool"):
            assert df[col].between(0, 1).all(), f"{col} out of [0, 1]"
        # Nodes have R12_ prefix
        assert df["node"].str.startswith("R12_").all()
        # 12 regions
        assert df["node"].nunique() == 12

    def test_floor_areas(self):
        df = load_floor_areas("resid")
        assert {"region", "year", "arch", "urt", "floor_Mm2"} == set(df.columns)
        assert (df["floor_Mm2"] >= 0).all()


class TestPrediction:
    @pytest.fixture
    def ei_cool(self):
        """Predict EI for 3 GMT values (single trajectory)."""
        gmt = np.array([1.0, 1.5, 2.0])
        return predict_building_ei(gmt, "cool")

    def test_shape(self, ei_cool):
        # (12 regions, 10 archs, 3 urts, 3 years)
        assert ei_cool.shape == (12, 10, 3, 3)

    def test_positive(self, ei_cool):
        # EI should be non-negative (cooling energy intensity)
        assert np.nanmin(ei_cool) >= 0

    def test_warming_increases_cooling(self, ei_cool):
        # Higher GMT should increase cooling EI (globally averaged)
        mean_per_year = np.nanmean(ei_cool, axis=(0, 1, 2))
        assert mean_per_year[2] > mean_per_year[0], (
            "Cooling EI should increase with warming"
        )

    def test_heat_warming_decreases(self):
        gmt = np.array([1.0, 1.5, 2.0])
        ei_heat = predict_building_ei(gmt, "heat")
        mean_per_year = np.nanmean(ei_heat, axis=(0, 1, 2))
        assert mean_per_year[2] < mean_per_year[0], (
            "Heating EI should decrease with warming"
        )

    def test_ensemble_shape(self):
        # 2D input: (5 runs, 3 years)
        gmt_2d = np.tile([1.0, 1.5, 2.0], (5, 1))
        ei = predict_building_ei(gmt_2d, "cool")
        # Same output shape — ensemble averaged
        assert ei.shape == (12, 10, 3, 3)


class TestEiToDataframe:
    def test_roundtrip(self):
        ds = xr.open_dataset(
            str(impacts_data_path("rime", "region_EI_cool_gwl_binned.nc"))
        )
        gmt = np.array([1.0, 2.0])
        ei_all = predict_building_ei(gmt, "cool")
        df = _ei_to_dataframe(ei_all, ds, [2020, 2100])

        assert set(df.columns) == {"region", "arch", "urt", "year", "ei"}
        assert len(df) == 12 * 10 * 3 * 2  # 720
        assert df["region"].nunique() == 12
        assert df["arch"].nunique() == 10


class TestMfhWeightedEi:
    def test_produces_output(self):
        ds = xr.open_dataset(
            str(impacts_data_path("rime", "region_EI_cool_gwl_binned.nc"))
        )
        gmt = np.array([1.0, 2.0])
        ei_all = predict_building_ei(gmt, "cool")
        ei_df = _ei_to_dataframe(ei_all, ds, [2020, 2100])
        resid_floor = load_floor_areas("resid")

        mfh_df = _mfh_weighted_ei(ei_df, resid_floor)
        assert set(mfh_df.columns) == {"region", "urt", "year", "ei"}
        assert len(mfh_df) > 0
        assert (mfh_df["ei"] > 0).all()


class TestComputeBuildingCids:
    _MODEL_YEARS = [
        2020,
        2025,
        2030,
        2035,
        2040,
        2045,
        2050,
        2055,
        2060,
        2070,
        2080,
        2090,
        2100,
        2110,
    ]

    @pytest.fixture
    def cids(self):
        # Synthetic ensemble: 10 runs, uniform GMT ramp over annual 2020-2100
        years = np.arange(2020, 2101)
        gmt_base = np.linspace(1.0, 2.5, len(years))
        gmt_2d = gmt_base[np.newaxis, :] + np.random.default_rng(42).normal(
            0, 0.05, (10, len(years))
        )
        return compute_building_cids(gmt_2d, years, self._MODEL_YEARS)

    def test_output_format(self, cids):
        cooling, heating = cids
        for df in (cooling, heating):
            assert set(df.columns) == {"node", "year", "value"}
            assert df["node"].str.startswith("R12_").all()
            assert (df["value"] >= 0).all()

    def test_has_all_regions(self, cids):
        cooling, _ = cids
        assert cooling["node"].nunique() == 12

    def test_has_2110(self, cids):
        cooling, heating = cids
        assert 2110 in cooling["year"].values
        assert 2110 in heating["year"].values

    def test_units_are_gwa(self, cids):
        cooling, _ = cids
        # Output starts at the first year covered by the theta calibration.
        first_year = int(
            min(
                load_theta("cool", "SSP2")["year"].min(),
                load_theta("heat", "SSP2")["year"].min(),
            )
        )
        total_first_year = cooling.loc[cooling["year"] == first_year, "value"].sum()
        assert 0.1 < total_first_year < 500, (
            f"Unexpected {first_year} cooling: {total_first_year} GWa"
        )

    def test_theta_reproduces_calibrated_demand_at_gwl_1_1(self):
        years = np.arange(2020, 2101)
        model_years = self._MODEL_YEARS
        gmt = np.full(len(years), 1.1)

        cooling, heating = compute_building_cids(
            gmt,
            years,
            model_years,
            reference_scenario="SSP2",
        )

        fractions = load_sector_fractions("SSP2")
        rc_spec = pd.read_csv("_staging/rc_spec_baseline.csv")[
            ["node", "year", "value"]
        ]
        rc_therm = pd.read_csv("_staging/rc_therm_baseline.csv")[
            ["node", "year", "value"]
        ]

        for cid_df, scenario_df, frac_cols in (
            (cooling, rc_spec, ["frac_resid_cool", "frac_comm_cool"]),
            (heating, rc_therm, ["frac_resid_heat", "frac_comm_heat"]),
        ):
            calibrated = scenario_df.merge(
                fractions[["node", "year"] + frac_cols],
                on=["node", "year"],
                how="inner",
            )
            calibrated = calibrated.assign(
                calibrated=lambda df: df["value"] * df[frac_cols].sum(axis=1)
            )[["node", "year", "calibrated"]]

            merged = cid_df.merge(calibrated, on=["node", "year"], how="inner")
            np.testing.assert_allclose(
                merged["value"],
                merged["calibrated"],
                atol=1e-10,
                rtol=0,
            )
