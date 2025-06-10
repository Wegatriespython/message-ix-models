import pandas as pd
import pytest
from message_ix import Scenario

from message_ix_models import ScenarioInfo
from message_ix_models.model.structure import get_codes
from message_ix_models.model.water.data.demands import (
    add_irrigation_demand,
    add_sectoral_demands,
    add_water_availability,
    get_basin_sizes,
    set_target_rate,
)


@pytest.fixture(params=["ZMB"])
def water_context_country(test_context, request):
    """Setup test context for country-level regions."""
    region = request.param

    # Create scenario
    mp = test_context.get_platform()
    s = Scenario(
        mp=mp,
        model=f"test_water_model_{region}",
        scenario=f"test_water_scenario_{region}",
        version="new",
    )
    s.add_horizon(year=[2020, 2030, 2040, 2050])
    s.add_set("technology", ["tech1", "tech2"])
    s.add_set("year", [2020, 2030, 2040, 2050])

    # Setup context
    test_context.type_reg = "country"
    test_context.regions = region
    nodes = get_codes(f"node/{region}")
    nodes = list(map(str, nodes[nodes.index("World")].child))
    test_context.map_ISO_c = {region: nodes[0]}
    # Commit the scenario and set it in context
    s.commit(comment="test scenario setup")
    test_context.set_scenario(s)
    test_context["water build info"] = ScenarioInfo(s)
    test_context.RCP = "2p6"
    test_context.REL = "low"
    return test_context


@pytest.fixture(params=["R12"])
def water_context_global(test_context, request):
    """Setup test context for global-level regions."""
    region = request.param

    sets = {"year": [2020, 2030, 2040]}
    test_context["water build info"] = ScenarioInfo(y0=2020, set=sets)
    test_context.type_reg = "global"
    test_context.regions = region
    test_context.RCP = "2p6"
    test_context.REL = "low"
    return test_context


@pytest.fixture
def sample_data():
    """Sample data for testing core functions."""
    return {
        "basin": pd.DataFrame(
            {
                "BCU_name": ["Basin1", "Basin1", "Basin2", "Basin2", "Basin3"],
                "STATUS": ["DEV", "IND", "DEV", "DEV", "IND"],
                "country": ["Country1", "Country2", "Country1", "Country3", "Country2"],
            }
        ),
        "demand": pd.DataFrame(
            {
                "node": ["Basin1", "Basin1", "Basin2", "Basin2"],
                "year": [2020, 2030, 2020, 2030],
                "value": [0.5, 0.6, 0.3, 0.4],
                "time": ["year", "year", "year", "year"],
            }
        ),
    }


def test_get_basin_sizes(sample_data):
    """Test basin size calculation with multiple scenarios."""
    basin_data = sample_data["basin"]

    # Mixed basin (1 DEV, 1 IND)
    dev_size, ind_size = get_basin_sizes(basin_data, "Basin1")
    assert dev_size == 1 and ind_size == 1

    # DEV majority basin
    dev_size, ind_size = get_basin_sizes(basin_data, "Basin2")
    assert dev_size == 2 and ind_size == 0

    # Non-existent basin
    dev_size, ind_size = get_basin_sizes(basin_data, "NonExistent")
    assert dev_size == 0 and ind_size == 0


def test_set_target_rate_strategies(sample_data):
    """Test all target rate strategies with error handling."""
    demand_data = sample_data["demand"].copy()
    basin_data = sample_data["basin"]

    # Direct strategy
    result = set_target_rate(demand_data, "direct", 0.8, node="Basin1", year=2030)
    assert isinstance(result, pd.DataFrame)

    # Connection strategy
    result = set_target_rate(demand_data, "connection", 0.9, basin=basin_data)
    assert isinstance(result, pd.DataFrame)

    # Invalid strategy
    with pytest.raises(ValueError, match="Unknown strategy"):
        set_target_rate(demand_data, "invalid", 0.8)

    # Missing parameters
    with pytest.raises(ValueError, match="'direct' strategy requires"):
        set_target_rate(demand_data, "direct", 0.8)


def test_empty_basin_data():
    """Test get_basin_sizes with empty data."""
    empty_basin = pd.DataFrame(columns=["BCU_name", "STATUS"])
    dev_size, ind_size = get_basin_sizes(empty_basin, "AnyNode")
    assert dev_size == 0 and ind_size == 0


@add_sectoral_demands.minimum_version
@pytest.mark.parametrize("SDG", ["baseline", "SDG"])
@pytest.mark.parametrize("time", ["year", "month"])
def test_sectoral_demands_pipeline(water_context_country, SDG, time):
    """Test complete sectoral demands pipeline."""
    water_context_country.SDG = SDG
    water_context_country.time = time

    result = add_sectoral_demands(context=water_context_country)

    # Structure validation
    expected_keys = {
        "demand",
        "historical_new_capacity",
        "historical_activity",
        "share_commodity_lo",
    }
    assert set(result.keys()) == expected_keys

    # Data validation
    demand_df = result["demand"]
    assert not demand_df.empty
    assert demand_df["value"].notna().all()
    assert demand_df["unit"].eq("MCM/year").all()

    # Expected commodities
    expected_commodities = {
        "urban_mw",
        "rural_mw",
        "industry_mw",
        "urban_disconnected",
        "rural_disconnected",
        "urban_collected_wst",
        "rural_collected_wst",
        "urban_uncollected_wst",
        "rural_uncollected_wst",
        "industry_uncollected_wst",
    }
    actual_commodities = set(demand_df["commodity"].unique())
    assert expected_commodities.issubset(actual_commodities)

    # Water balance check
    for (node, year), group in demand_df.groupby(["node", "year"]):
        withdrawals = group[group["value"] > 0]["value"].sum()
        returns = abs(group[group["value"] < 0]["value"].sum())
        if withdrawals > 0 and returns > 0:
            assert returns <= withdrawals * 1.01


@pytest.mark.parametrize("time", ["year", "month"])
def test_water_availability(water_context_global, time):
    """Test water availability returns valid structure and data."""
    water_context_global.time = time

    result = add_water_availability(context=water_context_global)

    # Structure check
    assert "demand" in result and "share_commodity_lo" in result

    demand_df = result["demand"]
    assert not demand_df.empty
    assert (demand_df["value"] <= 0).all()  # Supply, not demand

    # Both water types present
    commodities = set(demand_df["commodity"].unique())
    assert {"surfacewater_basin", "groundwater_basin"} == commodities


@add_sectoral_demands.minimum_version
@pytest.mark.parametrize("SDG", ["baseline", "SDG"])
@pytest.mark.parametrize("time", ["year", "month"])
def test_time_series_continuity(water_context_country, SDG, time):
    """Test that time series have no gaps in year coverage."""
    water_context_country.SDG = SDG
    water_context_country.time = time

    result = add_sectoral_demands(context=water_context_country)
    demand_df = result["demand"]

    # Get model years from context
    model_years = water_context_country["water build info"].Y

    # Check each commodity/node combination has all years
    for (commodity, node), group in demand_df.groupby(["commodity", "node"]):
        actual_years = sorted(group["year"].unique())
        expected_years = sorted([y for y in model_years if y >= 2020])

        assert actual_years == expected_years, (
            f"Missing years for {commodity} at {node}:"
            f"expected {expected_years}, got {actual_years}"
        )


@add_sectoral_demands.minimum_version
@pytest.mark.parametrize("SDG", ["baseline", "SDG"])
@pytest.mark.parametrize("time", ["year", "month"])
def test_commodity_conservation(water_context_country, SDG, time):
    """Test that commodity transformations preserve mass balance principles."""
    water_context_country.SDG = SDG
    water_context_country.time = time

    result = add_sectoral_demands(context=water_context_country)
    demand_df = result["demand"]

    # For each node and year, check conservation
    for (node, year), group in demand_df.groupby(["node", "year"]):
        withdrawals = group[group["commodity"].str.contains("mw|disconnected")]
        wastewaters = group[group["commodity"].str.contains("wst")]

        if len(withdrawals) > 0 and len(wastewaters) > 0:
            total_withdrawal = withdrawals["value"].abs().sum()
            total_wastewater = wastewaters["value"].abs().sum()

            # Wastewater should not exceed withdrawals
            assert total_wastewater <= total_withdrawal * 1.01, (
                f"Wastewater ({total_wastewater}) exceeds"
                f"withdrawals ({total_withdrawal}) at {node}, {year}"
            )


@add_sectoral_demands.minimum_version
@pytest.mark.parametrize("SDG", ["baseline", "SDG"])
@pytest.mark.parametrize("time", ["year", "month"])
def test_data_pipeline_preserves_structure(water_context_country, SDG, time):
    """Test that the data pipeline preserves essential structural relationships."""
    water_context_country.SDG = SDG
    water_context_country.time = time

    result = add_sectoral_demands(context=water_context_country)

    # Check consistent node naming across all outputs
    demand_nodes = set(result["demand"]["node"].unique())
    hist_act_nodes = set(result["historical_activity"]["node_loc"].unique())

    # All should use the same basin prefix convention
    assert all(node.startswith("B") for node in demand_nodes)
    assert all(node.startswith("B") for node in hist_act_nodes)

    # Historical data should be subset of demand data nodes
    assert hist_act_nodes.issubset(demand_nodes)


def test_add_irrigation_demand(water_context_country):
    """Test irrigation demand functionality."""
    # The scenario is already committed and set in the context by the fixture
    result = add_irrigation_demand(context=water_context_country)

    # Assert the results
    assert isinstance(result, dict)
    assert "land_input" in result
    assert all(
        col in result["land_input"].columns
        for col in ["value", "unit", "level", "commodity", "node", "time", "year"]
    )
