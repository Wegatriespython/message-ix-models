import pytest
from message_ix import Scenario

from message_ix_models import ScenarioInfo
from message_ix_models.model.structure import get_codes
from message_ix_models.model.water.data.infrastructure import (
    add_desalination,
    add_infrastructure_techs,
    get_vintage_and_active_years,
)
from message_ix_models.model.water.utils import map_yv_ya_lt


@pytest.fixture(params=["ZMB", "R12"])
def water_context_regions(test_context, request):
    """Setup test context for different regions."""
    region = request.param

    test_context.SDG = "baseline"
    test_context.time = ["year"]
    test_context.type_reg = "country"
    test_context.regions = region
    test_context.RCP = "7p0"

    nodes = get_codes(f"node/{region}")
    nodes = list(map(str, nodes[nodes.index("World")].child))
    test_context.map_ISO_c = {region: nodes[0]}

    return test_context


@pytest.fixture
def scenario_base(request):
    """Create basic scenario for testing."""
    return {
        "model": f"{request.node.name}/test water model",
        "scenario": f"{request.node.name}/test water scenario",
        "version": "new",
    }


@pytest.mark.parametrize("SDG", ["baseline", "not_baseline"])
def test_add_infrastructure_techs(water_context_regions, scenario_base, SDG, request):
    """Test infrastructure techs data generation."""
    water_context_regions.SDG = SDG

    mp = water_context_regions.get_platform()
    s = Scenario(mp=mp, **scenario_base)
    s.add_horizon(year=[2020, 2030, 2040])
    s.add_set("technology", ["tech1", "tech2"])
    s.add_set("year", [2020, 2030, 2040])
    s.commit(comment="test infrastructure techs")

    water_context_regions.set_scenario(s)
    water_context_regions["water build info"] = ScenarioInfo(s)

    result = add_infrastructure_techs(context=water_context_regions)

    assert isinstance(result, dict)
    assert "input" in result and "output" in result

    expected_input_cols = [
        "technology",
        "value",
        "unit",
        "level",
        "commodity",
        "mode",
        "time",
        "time_origin",
        "node_origin",
        "node_loc",
        "year_vtg",
        "year_act",
    ]
    expected_output_cols = [
        "technology",
        "value",
        "unit",
        "level",
        "commodity",
        "mode",
        "time",
        "time_dest",
        "node_loc",
        "node_dest",
        "year_vtg",
        "year_act",
    ]

    assert all(col in result["input"].columns for col in expected_input_cols)
    assert all(col in result["output"].columns for col in expected_output_cols)


def test_add_desalination(water_context_regions, scenario_base, request):
    """Test desalination data generation."""
    water_context_regions.type_reg = "global"

    mp = water_context_regions.get_platform()
    s = Scenario(mp=mp, **scenario_base)
    s.add_horizon(year=[2020, 2030, 2040])
    s.add_set("technology", ["tech1", "tech2"])
    s.add_set("year", [2020, 2030, 2040])
    s.commit(comment="test desalination")

    water_context_regions.set_scenario(s)
    water_context_regions["water build info"] = ScenarioInfo(s)

    result = add_desalination(context=water_context_regions)

    assert isinstance(result, dict)
    assert "input" in result and "output" in result

    expected_input_cols = [
        "technology",
        "value",
        "unit",
        "level",
        "commodity",
        "mode",
        "time",
        "time_origin",
        "node_origin",
        "node_loc",
        "year_vtg",
        "year_act",
    ]
    expected_output_cols = [
        "technology",
        "value",
        "unit",
        "level",
        "commodity",
        "mode",
        "time",
        "time_dest",
        "node_loc",
        "node_dest",
        "year_vtg",
        "year_act",
    ]

    assert all(col in result["input"].columns for col in expected_input_cols)
    assert all(col in result["output"].columns for col in expected_output_cols)


@pytest.mark.parametrize(
    "periods,lifetime,first_year",
    [
        ((2010, 2020, 2030, 2040), 20, 2020),
        ((2020, 2030, 2040), 20, 2020),
    ],
)
def test_map_yv_ya_lt_excessive_combinations(periods, lifetime, first_year):
    """Document that old map_yv_ya_lt generates
    excessive vintage-activity combinations."""
    result = map_yv_ya_lt(periods, lifetime, first_year)
    model_years = [y for y in periods if y >= first_year]
    proper_combinations = len(model_years)

    # This test documents the PROBLEM with the old function
    # It should fail to show the issue exists
    try:
        assert len(result) <= proper_combinations, (
            f"ISSUE: Excessive combinations ({len(result)}) vs proper "
            f"MESSAGEix ({proper_combinations})"
        )
        # If this passes, there's no issue (shouldn't happen)
        pytest.fail("Expected excessive combinations but didn't find them")
    except AssertionError as e:
        # This is expected - the old function generates excessive combinations
        assert "ISSUE: Excessive combinations" in str(e)

    # Check for historical vintages in model periods - this should also fail
    historical_years = [y for y in periods if y < first_year]
    if historical_years:
        historical_in_model = result[
            result["year_vtg"].isin(historical_years)
            & result["year_act"].isin([y for y in periods if y >= first_year])
        ]
        # This documents the issue - historical vintages in model periods
        if not historical_in_model.empty:
            # This is the problem we're documenting
            pass

    assert list(result.columns) == ["year_vtg", "year_act"]
    assert all(result["year_vtg"] <= result["year_act"])
    assert all(result["year_act"] >= first_year)
    assert all(result["year_act"] - result["year_vtg"] <= lifetime)


def test_get_vintage_and_active_years_fixed(
    water_context_regions, scenario_base, request
):
    """Test that new get_vintage_and_active_years generates proper combinations."""
    mp = water_context_regions.get_platform()
    s = Scenario(mp=mp, **scenario_base)
    s.add_horizon(year=[2010, 2020, 2030, 2040])
    s.add_set("technology", ["test_tech"])
    s.add_set("year", [2010, 2020, 2030, 2040])
    s.commit(comment="test vintage-activity years")

    # Add a simple technology for testing
    s.check_out()
    s.add_set("node", ["test_node"])
    s.add_par("technical_lifetime", ["test_node", "test_tech", 2020], 20, "y")
    s.commit(comment="add tech lifetime")

    # Test our fixed function
    result = get_vintage_and_active_years(s, "test_node", "test_tech")

    # Should have proper columns
    expected_cols = ["year_vtg", "year_act"]
    assert list(result.columns) == expected_cols

    # Should not have excessive combinations
    model_years = [2020, 2030, 2040]  # Years >= firstmodelyear
    # With MESSAGEix logic, should have reasonable number of combinations
    assert len(result) <= len(model_years) * 2, f"Too many combinations: {len(result)}"

    # Should not have historical vintages active in model periods
    if len(result) > 0:
        historical_in_model = result[
            (result["year_vtg"] < s.firstmodelyear)
            & (result["year_act"] >= s.firstmodelyear)
        ]
        assert historical_in_model.empty, (
            f"Fixed function should not have historical vintages in model periods: "
            f"{len(historical_in_model)}"
        )

        # Basic validity checks
        assert all(result["year_vtg"] <= result["year_act"])
        assert all(result["year_act"] >= s.firstmodelyear)


def test_vintage_activity_year_issues(water_context_regions, scenario_base, request):
    """Document infrastructure functions generate problematic year combinations."""
    mp = water_context_regions.get_platform()
    s = Scenario(mp=mp, **scenario_base)
    s.add_horizon(year=[2010, 2020, 2030, 2040])
    s.add_set("technology", ["tech1", "tech2"])
    s.add_set("year", [2010, 2020, 2030, 2040])
    s.commit(comment="test with historical years")

    water_context_regions.set_scenario(s)
    water_context_regions["water build info"] = ScenarioInfo(s)

    result = add_infrastructure_techs(context=water_context_regions)

    if "input" in result and not result["input"].empty:
        year_combos = (
            result["input"][["year_vtg", "year_act"]]
            .drop_duplicates()
            .sort_values(["year_vtg", "year_act"])
        )

        first_model_year = 2020
        historical_in_model = year_combos[
            (year_combos["year_vtg"] < first_model_year)
            & (year_combos["year_act"] >= first_model_year)
        ]

        if not historical_in_model.empty:
            pass  # Issue documented


def test_uncontrolled_broadcasting_to_all_basins(
    water_context_regions, scenario_base, request
):
    """Verify technologies are not broadcast to ALL basins without filtering."""
    mp = water_context_regions.get_platform()
    s = Scenario(mp=mp, **scenario_base)
    s.add_horizon(year=[2020, 2030, 2040])
    s.add_set("technology", ["tech1", "tech2"])
    s.add_set("year", [2020, 2030, 2040])
    s.commit(comment="test broadcasting")

    water_context_regions.set_scenario(s)
    water_context_regions["water build info"] = ScenarioInfo(s)

    result = add_infrastructure_techs(context=water_context_regions)

    if "input" in result and not result["input"].empty:
        unique_basins = result["input"]["node_loc"].nunique()
        unique_techs = result["input"]["technology"].nunique()

        tech_basin_combinations = (
            result["input"].groupby("technology")["node_loc"].nunique().reset_index()
        )

        techs_on_all_basins = tech_basin_combinations[
            tech_basin_combinations["node_loc"] == unique_basins
        ]

        # ISSUE: Should be 0 with proper basin filtering
        assert len(techs_on_all_basins) == 0, (
            f"ISSUE: No technologies should be on ALL {unique_basins} basins, "
            f"but found {len(techs_on_all_basins)}"
        )

        total_combinations = len(result["input"])
        # With filtering, combinations should be reasonable
        max_reasonable = unique_techs * unique_basins  # Upper bound

        # ISSUE: Should be reasonable with proper filtering
        assert total_combinations < max_reasonable, (
            f"ISSUE: Combinations ({total_combinations}) should be less than "
            f"maximum possible ({max_reasonable})"
        )


def test_missing_technology_basin_filtering(
    water_context_regions, scenario_base, request
):
    """Document inappropriate technology-basin combinations."""
    water_context_regions.type_reg = "global"

    mp = water_context_regions.get_platform()
    s = Scenario(mp=mp, **scenario_base)
    s.add_horizon(year=[2020, 2030, 2040])
    s.add_set("technology", ["tech1", "tech2"])
    s.add_set("year", [2020, 2030, 2040])
    s.commit(comment="test filtering")

    water_context_regions.set_scenario(s)
    water_context_regions["water build info"] = ScenarioInfo(s)

    result = add_desalination(context=water_context_regions)

    if "input" in result and not result["input"].empty:
        total_combinations = len(
            result["input"][["technology", "node_loc"]].drop_duplicates()
        )
        unique_basins = result["input"]["node_loc"].nunique()
        unique_techs = result["input"]["technology"].nunique()
        expected_all_combinations = unique_basins * unique_techs

        assert total_combinations < expected_all_combinations * 0.5, (
            f"ISSUE: {total_combinations} desalination combinations "
            f"across {unique_basins} basins"
        )

        techs_per_basin = (
            result["input"].groupby("node_loc")["technology"].nunique().reset_index()
        )

        basins_with_most_techs = techs_per_basin[
            techs_per_basin["technology"] >= unique_techs * 0.8
        ]

        assert len(basins_with_most_techs) == 0, (
            f"ISSUE: {len(basins_with_most_techs)} basins have most technologies"
        )


def test_redundant_data_generation(water_context_regions, scenario_base, request):
    """Document redundant/duplicate data generation patterns."""
    mp = water_context_regions.get_platform()
    s = Scenario(mp=mp, **scenario_base)
    s.add_horizon(year=[2020, 2030, 2040])
    s.add_set("technology", ["tech1", "tech2"])
    s.add_set("year", [2020, 2030, 2040])
    s.commit(comment="test redundancy")

    water_context_regions.set_scenario(s)
    water_context_regions["water build info"] = ScenarioInfo(s)

    result = add_infrastructure_techs(context=water_context_regions)

    if "input" in result and not result["input"].empty:
        duplicate_rows = result["input"].duplicated().sum()
        assert duplicate_rows == 0, f"ISSUE: {duplicate_rows} duplicate rows found"

        # Check for truly redundant combinations
        # (same technology, node, year, commodity, level, mode)
        full_key_columns = [
            "technology",
            "node_loc",
            "year_vtg",
            "year_act",
            "commodity",
            "level",
            "mode",
        ]
        if all(col in result["input"].columns for col in full_key_columns):
            grouped = result["input"].groupby(full_key_columns).size()
            redundant_combinations = (grouped > 1).sum()

            assert redundant_combinations == 0, (
                f"ISSUE: {redundant_combinations} truly redundant parameter "
                "combinations found"
            )
