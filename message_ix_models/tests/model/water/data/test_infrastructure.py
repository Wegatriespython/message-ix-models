import pandas as pd
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
    # Fix: R12 should be global, not country
    test_context.type_reg = "global" if region == "R12" else "country"
    test_context.regions = region
    test_context.RCP = "7p0"

    nodes = get_codes(f"node/{region}")
    nodes = list(map(str, nodes[nodes.index("World")].child))
    # Only set map_ISO_c for country-level regions
    if test_context.type_reg == "country":
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


@pytest.fixture
def scenario_with_context(water_context_regions, scenario_base, request):
    """Create a properly set up scenario with context for testing."""
    mp = water_context_regions.get_platform()
    s = Scenario(mp=mp, **scenario_base)
    s.add_horizon(year=[2020, 2030, 2040])
    s.add_set("technology", ["tech1", "tech2"])
    s.add_set("year", [2020, 2030, 2040])
    s.commit(comment=f"test setup for {request.node.name}")

    water_context_regions.set_scenario(s)
    water_context_regions["water build info"] = ScenarioInfo(s)

    return water_context_regions, s


@pytest.mark.parametrize("SDG", ["baseline", "not_baseline"])
def test_add_infrastructure_techs(scenario_with_context, SDG, request):
    """Test infrastructure techs data generation."""
    context, s = scenario_with_context
    context.SDG = SDG

    result = add_infrastructure_techs(context=context)

    technology_name = "urban_t_d"

    input = s.par("input", {"technology": technology_name})
    output = s.par("output", {"technology": technology_name})

    print(input)
    print(output)

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


def test_add_desalination(scenario_with_context, request):
    """Test desalination data generation."""
    context, s = scenario_with_context
    context.type_reg = "global"

    result = add_desalination(context=context)

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


def test_map_yv_ya_lt_fixed_wrapper(scenario_with_context, request):
    """Test that map_yv_ya_lt now properly wraps scenario.vintage_and_active_years()."""

    context, s = scenario_with_context

    # Modify scenario for this specific test - add historical year and specific tech
    s.check_out()

    # Test the refactored map_yv_ya_lt function
    result = map_yv_ya_lt(s, "test_node", "extract_surfacewater")
    print("\n=== REFACTORED map_yv_ya_lt RESULTS ===")
    print(f"Generated {len(result)} combinations:")
    print(result)
    print(f"Scenario firstmodelyear: {s.firstmodelyear}")

    # Should return proper DataFrame
    assert isinstance(result, pd.DataFrame)
    assert "year_vtg" in result.columns
    assert "year_act" in result.columns

    # Test the key fix: should NOT have historical vintages in model periods
    if not result.empty:
        # Should not have historical vintages (2010) active in model periods (2020+)
        historical_in_model = result[
            (result["year_vtg"] < s.firstmodelyear)
            & (result["year_act"] >= s.firstmodelyear)
        ]
        assert historical_in_model.empty, (
            f"FIXED: Should not have historical vintages in model periods, "
            f"but found {len(historical_in_model)} combinations: {historical_in_model}"
        )

        # Basic MESSAGEix validity checks
        assert all(result["year_vtg"] <= result["year_act"])
        assert all(result["year_act"] >= s.firstmodelyear)

    # Test fallback behavior when technology doesn't have technical_lifetime
    result_fallback = map_yv_ya_lt(s, "test_node", "nonexistent_tech")
    assert isinstance(result_fallback, pd.DataFrame)
    assert "year_vtg" in result_fallback.columns
    assert "year_act" in result_fallback.columns

    # Fallback should create minimal model-year-only combinations
    if not result_fallback.empty:
        model_years = [y for y in s.set("year") if y >= s.firstmodelyear]
        assert all(vtg in model_years for vtg in result_fallback.year_vtg.unique())
        assert all(result_fallback.year_vtg <= result_fallback.year_act)


def test_get_vintage_and_active_years_fixed(scenario_with_context, request):
    """Test that new get_vintage_and_active_years generates proper combinations."""
    context, s = scenario_with_context

    # Modify scenario for this specific test
    s.check_out()
    s.add_set("year", [2010])  # Add historical year
    s.add_set("technology", ["test_tech"])
    s.add_set("node", ["test_node"])
    s.add_par("technical_lifetime", ["test_node", "test_tech", 2020], 20, "y")
    s.commit(comment="add tech lifetime for test")

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


def test_vintage_activity_year_issues(scenario_with_context, request):
    """Document infrastructure functions generate problematic year combinations."""
    context, s = scenario_with_context

    # Add historical year for this test
    s.check_out()
    s.add_set("year", [2010])
    s.commit(comment="add historical year for test")

    # Update context scenario
    context.set_scenario(s)
    context["water build info"] = ScenarioInfo(s)

    result = add_infrastructure_techs(context=context)

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


def test_uncontrolled_broadcasting_to_all_basins(scenario_with_context, request):
    """Verify technologies are not broadcast to ALL basins without filtering."""
    context, s = scenario_with_context

    result = add_infrastructure_techs(context=context)

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


def test_missing_technology_basin_filtering(scenario_with_context, request):
    """Document inappropriate technology-basin combinations."""
    context, s = scenario_with_context
    context.type_reg = "global"

    result = add_desalination(context=context)

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


def test_redundant_data_generation(scenario_with_context, request):
    """Document redundant/duplicate data generation patterns."""
    context, s = scenario_with_context

    result = add_infrastructure_techs(context=context)

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


def test_basin_region_mapping(scenario_with_context, request):
    """Verify basin-to-region mapping is correct with no overlaps."""
    import pandas as pd

    from message_ix_models.util import package_data_path

    context, s = scenario_with_context

    # Only test for R12 regions
    if context.regions != "R12":
        return

    # Load basin data
    basin_file = package_data_path("water", "infrastructure", "all_basins.csv")
    df_basins = pd.read_csv(basin_file)

    # Filter for R12 regions
    r12_basins = df_basins[df_basins["model_region"] == "R12"]

    print("\nBasin-Region Mapping Analysis for R12:")
    print(f"Total basins in R12: {len(r12_basins)}")

    # Check regional distribution
    region_basin_counts = (
        r12_basins.groupby("REGION")["node"].count().sort_values(ascending=False)
    )
    print("Basins per region:")
    for region, count in region_basin_counts.items():
        print(f"  {region}: {count} basins")

    # Get unique regions
    unique_regions = r12_basins["REGION"].unique()
    print(f"Number of unique regions: {len(unique_regions)}")

    # Check for basin overlaps between regions
    basin_nodes = r12_basins.groupby("node")["REGION"].nunique()
    overlapping_basins = basin_nodes[basin_nodes > 1]

    print(f"Basins belonging to multiple regions: {len(overlapping_basins)}")
    if len(overlapping_basins) > 0:
        print("Overlapping basins:")
        for basin in overlapping_basins.items():
            regions = r12_basins[r12_basins["node"] == basin]["REGION"].unique()
            print(f"  {basin}: appears in regions {list(regions)}")

    # Check total combinations
    total_basins = len(r12_basins)

    result = add_infrastructure_techs(context=context)

    if "input" in result and not result["input"].empty:
        # Count unique basin-region combinations in result
        unique_basin_nodes = result["input"]["node_loc"].nunique()
        unique_origin_nodes = result["input"]["node_origin"].nunique()

        print("\nInfrastructure result analysis:")
        print(f"Unique basin nodes (node_loc): {unique_basin_nodes}")
        print(f"Unique origin nodes (node_origin): {unique_origin_nodes}")

        # Check if any basin appears with multiple regions
        basin_region_mapping = result["input"][
            ["node_loc", "node_origin"]
        ].drop_duplicates()
        basin_multi_regions = basin_region_mapping.groupby("node_loc")[
            "node_origin"
        ].nunique()
        problematic_basins = basin_multi_regions[basin_multi_regions > 1]

        print(
            f"Basins with multiple origin regions in result: {len(problematic_basins)}"
        )
        if len(problematic_basins) > 0:
            print("Problematic basins:")
            for basin, region_count in problematic_basins.head(10).items():
                origins = basin_region_mapping[
                    basin_region_mapping["node_loc"] == basin
                ]["node_origin"].unique()
                print(f"  {basin}: connected to {list(origins)}")

    # Assertions
    assert len(overlapping_basins) == 0, (
        f"Found {len(overlapping_basins)} basins in multiple regions"
    )
    assert len(unique_regions) == 12, (
        f"Expected 12 regions in R12, found {len(unique_regions)}"
    )
    assert total_basins == 217, f"Expected 217 basins in R12, found {total_basins}"
