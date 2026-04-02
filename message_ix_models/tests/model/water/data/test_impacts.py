"""Tests for model.water.data.impacts -- water-domain RIME transformations."""

import numpy as np
import pandas as pd
import pytest

from message_ix_models.model.water.utils import (
    N_MESSAGE_BASINS,
    N_RIME_BASINS,
    NAN_BASIN_IDS,
    load_basin_mapping,
    split_basin_macroregion,
)
from message_ix_models.tools.impacts import impacts_data_path


# Integration tests requiring RIME NetCDF files (for basin-to-index mapping)
_RIME_DIR = impacts_data_path("rime")
_HAS_RIME_DATA = (
    _RIME_DIR / "rime_regionarray_qtot_mean_CWatM_annual_window11.nc"
).exists()


@pytest.mark.skipif(not _HAS_RIME_DATA, reason="RIME NetCDF datasets not available")
class TestSplitBasinMacroregion:
    """Requires RIME data for basin-to-index mapping."""

    def test_nan_preservation(self):
        """Basins 0, 141, 154 should remain NaN (no RIME data)."""
        basin_mapping = load_basin_mapping()
        fake_rime = np.ones((N_RIME_BASINS, 3))
        result = split_basin_macroregion(fake_rime, basin_mapping)
        for idx, row in basin_mapping.iterrows():
            if row["BASIN_ID"] in NAN_BASIN_IDS:
                assert np.all(np.isnan(result[idx])), (
                    f"Row {idx} (BASIN_ID={row['BASIN_ID']}) should be NaN"
                )


@pytest.mark.skipif(not _HAS_RIME_DATA, reason="RIME NetCDF datasets not available")
class TestPredictWaterRime:
    """End-to-end: predict + basin expansion."""

    def test_annual_qtot(self):
        from message_ix_models.model.water.data.impacts import predict_water_rime

        gmt = np.linspace(1.0, 2.5, 10)
        result = predict_water_rime(gmt, "qtot_mean")
        assert result.shape == (N_MESSAGE_BASINS, 10)

    def test_annual_qr(self):
        from message_ix_models.model.water.data.impacts import predict_water_rime

        gmt = np.linspace(1.0, 2.5, 10)
        result = predict_water_rime(gmt, "qr")
        assert result.shape == (N_MESSAGE_BASINS, 10)

    def test_ensemble(self):
        from message_ix_models.model.water.data.impacts import predict_water_rime

        rng = np.random.default_rng(42)
        gmt_2d = rng.normal(1.5, 0.2, size=(5, 10))
        gmt_2d = np.clip(gmt_2d, 0.6, 7.4)
        result = predict_water_rime(gmt_2d, "qtot_mean")
        assert result.shape == (N_MESSAGE_BASINS, 10)

    def test_seasonal(self):
        from message_ix_models.model.water.data.impacts import predict_water_rime

        gmt = np.linspace(1.0, 2.5, 10)
        dry, wet = predict_water_rime(gmt, "qtot_mean", temporal_res="seasonal2step")
        assert dry.shape == (N_MESSAGE_BASINS, 10)
        assert wet.shape == (N_MESSAGE_BASINS, 10)


class TestBuildWaterCids:
    def test_output_structure_and_variable_coverage(self, monkeypatch):
        from message_ix_models.model.water.data import impacts

        basin_mapping = pd.DataFrame({"BCU_name": ["1", "2"]})
        qtot = np.array([[1.0, 2.0], [3.0, 4.0]])
        qr = np.array([[0.2, 0.4], [0.3, 0.8]])

        def fake_predict_water_rime(
            gmt_array,
            variable,
            temporal_res="annual",
            **kwargs,
        ):
            assert temporal_res == "annual"
            return qtot if variable == "qtot_mean" else qr

        def fake_sample_to_model_years(df, id_cols, msg_years):
            return df[id_cols + msg_years].copy()

        monkeypatch.setattr(impacts, "predict_water_rime", fake_predict_water_rime)
        monkeypatch.setattr(impacts, "load_basin_mapping", lambda: basin_mapping.copy())
        monkeypatch.setattr(
            impacts, "sample_to_model_years", fake_sample_to_model_years
        )

        sw_old = pd.DataFrame(
            {
                "node": ["B1", "B1", "B2", "B2"],
                "commodity": ["surfacewater_basin"] * 4,
                "level": ["water_avail_basin"] * 4,
                "year": [2020, 2021, 2020, 2021],
                "time": ["year"] * 4,
                "value": [-1.0, -1.0, -1.0, -1.0],
                "unit": ["MCM/year"] * 4,
            }
        )
        gw_old = sw_old.assign(commodity="groundwater_basin")
        share_old = pd.DataFrame(
            {
                "shares": ["share_low_lim_GWat"] * 4,
                "node_share": ["B1", "B1", "B2", "B2"],
                "year_act": [2020, 2021, 2020, 2021],
                "time": ["year"] * 4,
                "value": [0.0, 0.0, 0.0, 0.0],
                "unit": ["-"] * 4,
            }
        )

        result = impacts.build_water_cids(
            gmt_array=np.array([1.1, 1.2]),
            sw_old=sw_old,
            gw_old=gw_old,
            share_old=share_old,
            msg_years=[2020, 2021],
        )

        assert set(result) == {"sw", "gw", "share"}
        assert result["sw"][0] is sw_old
        assert result["gw"][0] is gw_old
        assert result["share"][0] is share_old

        sw_new = result["sw"][1].sort_values(["node", "year"]).reset_index(drop=True)
        gw_new = result["gw"][1].sort_values(["node", "year"]).reset_index(drop=True)
        share_new = (
            result["share"][1]
            .sort_values(["node_share", "year_act"])
            .reset_index(drop=True)
        )

        assert sw_new.columns.tolist() == [
            "node",
            "commodity",
            "level",
            "year",
            "time",
            "value",
            "unit",
        ]
        assert gw_new.columns.tolist() == sw_new.columns.tolist()
        assert share_new.columns.tolist() == [
            "shares",
            "node_share",
            "year_act",
            "time",
            "value",
            "unit",
        ]

        assert set(sw_new["commodity"]) == {"surfacewater_basin"}
        assert set(gw_new["commodity"]) == {"groundwater_basin"}
        assert set(share_new["shares"]) == {"share_low_lim_GWat"}

        np.testing.assert_allclose(sw_new["value"], [-800.0, -1600.0, -2700.0, -3200.0])
        np.testing.assert_allclose(gw_new["value"], [-200.0, -400.0, -300.0, -800.0])
        np.testing.assert_allclose(share_new["value"], [0.19, 0.19, 0.095, 0.19])
