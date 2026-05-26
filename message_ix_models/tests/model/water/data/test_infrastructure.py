import pandas as pd
import pytest

from message_ix_models.model.water.data.infrastructure import (
    add_desalination,
    add_infrastructure_techs,
)
from message_ix_models.tests.model.water.conftest import water_params

DESAL_TECS = ("membrane", "distillation")


@pytest.mark.parametrize(
    "water_context",
    [
        water_params("R11", SDG="baseline"),
        water_params("R11", SDG="not_baseline"),
        water_params("R12", SDG="baseline"),
        water_params("R12", SDG="not_baseline"),
        water_params("ZMB", SDG="baseline"),
        water_params("ZMB", SDG="not_baseline"),
        water_params("R12", reduced_basin=True, SDG="baseline"),
    ],
    indirect=True,
)
def test_add_infrastructure_techs(
    water_context, water_scenario, assert_message_params, assert_input_output_structure
):
    """Test add_infrastructure_techs with global and country model configurations.

    Also tests start_creating_input_dataframe() and prepare_input_dataframe()
    since they are called by add_infrastructure_techs().
    """
    result = add_infrastructure_techs(context=water_context)

    # Standard MESSAGE parameter validation
    assert_message_params(result, expected_keys=["input", "output"])
    assert_input_output_structure(result)


@pytest.mark.parametrize(
    "water_context",
    [
        water_params("R11", RCP="6p0", ssp="SSP2"),
        water_params("R12", RCP="7p0", ssp="SSP2"),
        water_params("ZMB", RCP="7p0", ssp="SSP2"),
        water_params("R12", reduced_basin=True, RCP="7p0", ssp="SSP2"),
    ],
    indirect=True,
)
def test_add_desalination(
    water_context, water_scenario, assert_message_params, assert_input_output_structure
):
    """Test add_desalination with global and country model configurations."""
    result = add_desalination(context=water_context)

    # Standard MESSAGE parameter validation
    assert_message_params(result, expected_keys=["input", "output"])
    assert_input_output_structure(result)


@pytest.mark.parametrize(
    "water_context",
    [
        water_params("R12", RCP="2p6", ssp="SSP2"),
        water_params("R11", RCP="6p0", ssp="SSP2"),
        water_params("R12", reduced_basin=True, RCP="2p6", ssp="SSP2"),
    ],
    indirect=True,
)
def test_add_desalination_extraction_invariant(water_context, water_scenario):
    """Sum of bound_activity_lo across desal techs must not exceed
    bound_total_capacity_up on extract_salinewater_basin at any (basin, year).

    Membrane and distillation both consume salinewater_basin produced by
    extract_salinewater_basin; if the forced per-tech activity floors sum to
    more than the upstream extraction cap, the LP is structurally infeasible.
    See ticket #571 for the regression this guards against.
    """
    firstyear = water_context.get_scenario().firstmodelyear
    result = add_desalination(context=water_context)

    blo = result["bound_activity_lo"]
    bup = result["bound_total_capacity_up"]

    # bound_lo must be confined to model years; pre-firstmodelyear activity
    # is governed by historical_new_capacity vintaging and cannot be forced
    # by an activity floor without contradicting it at low-CAP basins.
    assert (blo["year_act"] >= firstyear).all(), (
        f"bound_activity_lo includes pre-firstmodelyear rows: "
        f"{sorted(blo.loc[blo['year_act'] < firstyear, 'year_act'].unique())}"
    )

    desal_lo = blo[blo["technology"].isin(DESAL_TECS)]
    extract_up = bup[bup["technology"] == "extract_salinewater_basin"]

    lo_sum = (
        desal_lo.groupby(["node_loc", "year_act"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "lo_sum"})
    )
    cap = extract_up[["node_loc", "year_act", "value"]].rename(
        columns={"value": "cap"}
    )
    joined = lo_sum.merge(cap, on=["node_loc", "year_act"], how="inner")
    headroom = joined["cap"] - joined["lo_sum"]
    # 1e-9 tolerance for floating-point scaling artefacts
    bad = joined[headroom < -1e-9]
    assert bad.empty, (
        "sum(bound_activity_lo[desal]) exceeds bound_total_capacity_up"
        f"[extract_salinewater_basin] at:\n{bad.to_string(index=False)}"
    )


@pytest.mark.parametrize(
    "water_context",
    [
        water_params("R12", RCP="2p6", ssp="SSP2"),
        water_params("R11", RCP="6p0", ssp="SSP2"),
        water_params("R12", reduced_basin=True, RCP="2p6", ssp="SSP2"),
    ],
    indirect=True,
)
def test_add_desalination_no_nan(water_context, water_scenario):
    """No water parameter from add_desalination may carry NaN values."""
    result = add_desalination(context=water_context)
    for key, df in result.items():
        if isinstance(df, pd.DataFrame) and "value" in df.columns:
            assert not df["value"].isna().any(), f"{key}: NaN in value column"


@pytest.mark.parametrize(
    "water_context",
    [water_params("R12", RCP="2p6", ssp="SSP2")],
    indirect=True,
)
def test_add_desalination_bound_lo_le_bound_up(water_context, water_scenario):
    """Per (node, technology, year_act, mode, time), bound_activity_lo must
    not exceed bound_activity_up where both are set."""
    result = add_desalination(context=water_context)
    blo = result.get("bound_activity_lo")
    bup = result.get("bound_activity_up")
    if blo is None or bup is None or blo.empty or bup.empty:
        pytest.skip("bound_activity_up not produced by add_desalination")
    keys = ["node_loc", "technology", "year_act", "mode", "time"]
    j = blo.merge(bup, on=keys, suffixes=("_lo", "_up"))
    bad = j[j["value_lo"] > j["value_up"] + 1e-9]
    assert bad.empty, f"bound_activity_lo > bound_activity_up at:\n{bad}"
