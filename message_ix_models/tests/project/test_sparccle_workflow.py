"""Unit tests for :mod:`message_ix_models.project.sparccle.workflow`."""

import pytest

from message_ix_models.project.sparccle.workflow import _resolve_magicc_dir


STARTER = {
    "model": "SSP_SSP3_v6.6_sp",
    "scenario": "NPiREF_GDP_CI_95_ensemble_2_PHY",
    "ssp": "SSP3",
}


class TestResolveMagiccDir:
    def test_root_composes_per_starter(self):
        result = _resolve_magicc_dir(
            STARTER,
            magicc_root="/data/magicc",
            magicc_file=None,
            magicc_model_suffix="",
            n_starters=2,
        )
        assert result == (
            "/data/magicc/SSP_SSP3_v6.6_sp/NPiREF_GDP_CI_95_ensemble_2"
        )

    def test_model_suffix_appends_to_model_component(self):
        result = _resolve_magicc_dir(
            STARTER,
            magicc_root="/data/magicc",
            magicc_file=None,
            magicc_model_suffix="_p95",
            n_starters=2,
        )
        assert result == (
            "/data/magicc/SSP_SSP3_v6.6_sp_p95/NPiREF_GDP_CI_95_ensemble_2"
        )

    def test_scenario_without_phy_suffix_preserved(self):
        starter = {**STARTER, "scenario": "NPiREF_GDP_CI_95_ensemble_2"}
        result = _resolve_magicc_dir(
            starter,
            magicc_root="/data/magicc",
            magicc_file=None,
            magicc_model_suffix="",
            n_starters=1,
        )
        assert result.endswith("/NPiREF_GDP_CI_95_ensemble_2")

    def test_file_overrides_root_and_yaml(self, tmp_path):
        f = tmp_path / "x_IAMC_climateassessment.xlsx"
        f.touch()
        starter = {**STARTER, "magicc_output_dir": "/should/be/ignored"}
        result = _resolve_magicc_dir(
            starter,
            magicc_root="/also/ignored",
            magicc_file=f,
            magicc_model_suffix="",
            n_starters=1,
        )
        assert result == str(tmp_path)

    def test_file_rejected_for_multi_starter(self, tmp_path):
        f = tmp_path / "x.xlsx"
        f.touch()
        with pytest.raises(ValueError, match="exactly one starter"):
            _resolve_magicc_dir(
                STARTER,
                magicc_root=None,
                magicc_file=f,
                magicc_model_suffix="",
                n_starters=3,
            )

    def test_yaml_fallback_when_no_flags(self):
        starter = {**STARTER, "magicc_output_dir": "/legacy/path"}
        result = _resolve_magicc_dir(
            starter,
            magicc_root=None,
            magicc_file=None,
            magicc_model_suffix="",
            n_starters=1,
        )
        assert result == "/legacy/path"

    def test_root_overrides_yaml(self):
        starter = {**STARTER, "magicc_output_dir": "/should/be/ignored"}
        result = _resolve_magicc_dir(
            starter,
            magicc_root="/data/magicc",
            magicc_file=None,
            magicc_model_suffix="",
            n_starters=1,
        )
        assert result.startswith("/data/magicc/")

    def test_missing_path_raises(self):
        with pytest.raises(ValueError, match="lacks magicc_output_dir"):
            _resolve_magicc_dir(
                STARTER,
                magicc_root=None,
                magicc_file=None,
                magicc_model_suffix="",
                n_starters=1,
            )
