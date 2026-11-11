"""Unit tests for :mod:`message_ix_models.project.sparccle.workflow`."""

from pathlib import Path

import pytest

from message_ix_models.project.sparccle import workflow
from message_ix_models.project.sparccle.workflow import (
    MAX_SCENARIO_NAME,
    _resolve_magicc_dir,
    load_config,
    name_token,
    validate_names,
)

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
        assert result == ("/data/magicc/SSP_SSP3_v6.6_sp/NPiREF_GDP_CI_95_ensemble_2")

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


class TestNameToken:
    """The token that keeps two response arms from sharing one scenario name."""

    def test_absent_label_leaves_names_unchanged(self):
        # A config that does not opt in must keep producing the names it always
        # has, so no already-delivered row moves.
        assert name_token({"climate_label": None, "response_percentile": None}) == ""
        assert name_token({"climate_label": None, "response_percentile": 90}) == ""

    def test_mean_response_reads_as_E(self):
        token = name_token({"climate_label": "c95", "response_percentile": None})
        assert token == "_c95_iE"

    def test_percentile_response_carries_its_number(self):
        token = name_token({"climate_label": "c95", "response_percentile": 90})
        assert token == "_c95_i90"

    def test_two_arms_of_one_starter_get_different_names(self):
        # The defect this exists to prevent: the same starter built at two
        # response percentiles wrote one name, and whichever ran last won.
        starter = "INDC2030i_600f_GDP_CI_95_ensemble_1_PHY_cooling_CI_bp"
        expectation = starter + name_token(
            {"climate_label": "c95", "response_percentile": None}
        )
        tail = starter + name_token({"climate_label": "c95", "response_percentile": 90})
        assert expectation != tail

    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("scenario_config.yaml", ""),
            ("scenario_config_p95.yaml", "_c95_iE"),
            ("scenario_config_sts1_p95.yaml", "_c95_i90"),
        ],
    )
    def test_packaged_configs(self, filename, expected):
        path = Path(workflow.__file__).parent / filename
        assert name_token(load_config(path)["cid"]) == expected


PACKAGED_CONFIGS = [
    "scenario_config.yaml",
    "scenario_config_p95.yaml",
    "scenario_config_sts1_p95.yaml",
]


class TestValidateNames:
    """Names too long for the database must fail before anything is cloned."""

    @staticmethod
    def _config(scenario):
        return {"starters": [{"scenario": scenario}]}

    def test_name_within_the_limit_passes(self):
        # 39 characters plus the longest suffix and a 11-character token is the
        # widest thing the shipping p95 configs ask for.
        validate_names(
            self._config("INDC2030i_600f_GDP_CI_95_ensemble_1_PHY"), "_" + "x" * 10
        )

    def test_over_long_name_names_itself_and_its_length(self):
        with pytest.raises(ValueError) as exc:
            validate_names(
                self._config("INDC2030i_600f_GDP_CI_95_ensemble_1_PHY"),
                "_climp95_imp90",
            )
        message = str(exc.value)
        over_long = (
            "INDC2030i_600f_GDP_CI_95_ensemble_1_PHY_cooling_CI_bp_climp95_imp90"
        )
        assert over_long in message
        assert "67 characters" in message
        assert str(MAX_SCENARIO_NAME) in message

    def test_every_suffix_is_checked(self):
        # A token long enough that only the shortest suffix survives: the
        # buildings-only name must still be reported.
        with pytest.raises(ValueError) as exc:
            validate_names(self._config("x" * 55), "_" + "y" * 5)
        assert "_CI_b" in str(exc.value)

    @pytest.mark.parametrize("filename", PACKAGED_CONFIGS)
    def test_packaged_configs_fit(self, filename):
        config = load_config(Path(workflow.__file__).parent / filename)
        validate_names(config, name_token(config["cid"]))


class TestBuildingsToken:
    """Buildings scenario data must change the buildings names, and only those."""

    def test_absent_data_leaves_names_unchanged(self):
        assert workflow.buildings_token({"buildings_data": None}) == ""

    def test_label_names_the_data(self):
        cfg = {"buildings_data": "/some/dir", "buildings_label": "PAS"}
        assert workflow.buildings_token(cfg) == "_bPAS"

    def test_data_without_label_is_refused(self):
        with pytest.raises(ValueError, match="buildings_label"):
            workflow.buildings_token({"buildings_data": "/some/dir"})

    def test_cooling_only_name_is_not_checked_against_the_buildings_label(self):
        # 52 + "_cooling_CI_p" fits; with the label only the CI_b/CI_bp names
        # grow, so only those can be reported.
        config = {"starters": [{"scenario": "x" * 46}]}
        with pytest.raises(ValueError) as exc:
            validate_names(config, "", "_bPOSA")
        assert "_cooling_CI_bp_bPOSA" in str(exc.value)
        assert "_cooling_CI_p " not in str(exc.value)
