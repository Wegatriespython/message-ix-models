"""Tests for model.buildings.impacts — building energy CID integration."""

import numpy as np
import pandas as pd

from message_ix_models.model.buildings.impacts import (
    compute_building_cids,
    load_sector_fractions,
    predict_building_ei,
)


class TestPrediction:
    def test_warming_increases_cooling(self):
        gmt = np.array([1.0, 1.5, 2.0])
        ei_cool = predict_building_ei(gmt, "cool")
        mean_per_year = np.nanmean(ei_cool, axis=(0, 1, 2))
        assert mean_per_year[2] > mean_per_year[0]

    def test_heat_warming_decreases(self):
        gmt = np.array([1.0, 1.5, 2.0])
        ei_heat = predict_building_ei(gmt, "heat")
        mean_per_year = np.nanmean(ei_heat, axis=(0, 1, 2))
        assert mean_per_year[2] < mean_per_year[0]

    def test_ensemble_averages(self):
        gmt_2d = np.tile([1.0, 1.5, 2.0], (5, 1))
        ei = predict_building_ei(gmt_2d, "cool")
        # Ensemble averaging collapses runs → same spatial shape
        assert ei.shape == (12, 10, 3, 3)


class TestComputeBuildingCids:
    _MODEL_YEARS = [
        2020, 2025, 2030, 2035, 2040, 2045, 2050,
        2055, 2060, 2070, 2080, 2090, 2100, 2110,
    ]

    def test_theta_reproduces_calibrated_demand_at_gwl_1_1(self):
        """At GWL 1.1 (present-day), CID output must match beta * rc."""
        years = np.arange(2020, 2101)
        gmt = np.full(len(years), 1.1)

        cooling, heating = compute_building_cids(
            gmt, years, self._MODEL_YEARS, reference_scenario="SSP2",
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
