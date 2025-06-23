"""
Refactored test suite for the MESSAGEix-Nexus water model build process.

This script defines pytest fixtures to set up different configurations (regions,
scenario parameters) and then tests the water model data generation functions,
both individually and as a complete build process.
"""

import traceback
from typing import Dict, List, Tuple

import pandas as pd
import pytest

# Cache for built scenarios (simplified approach without diskcache)
_built_scenario_cache = {}
from message_ix import Scenario

from message_ix_models import ScenarioInfo
from message_ix_models.model.structure import get_codes
from message_ix_models.model.water.data.demands import (
    add_irrigation_demand,
    add_sectoral_demands,
    add_water_availability,
)
from message_ix_models.model.water.data.infrastructure import (
    add_desalination,
    add_infrastructure_techs,
)
from message_ix_models.model.water.data.irrigation import add_irr_structure
from message_ix_models.model.water.data.water_for_ppl import cool_tech, non_cooling_tec
from message_ix_models.model.water.data.water_supply import add_e_flow, add_water_supply
from message_ix_models.model.water.utils import read_config
from message_ix_models.util import add_par_data, package_data_path

# ==============================================================================
# Configuration
# ==============================================================================

# Groups all initial model sets for cleaner setup
MODEL_SETUP_CONFIG = {
    "commodity": [
        "electr",
        "d_heat",
        "gas",
        "coal",
        "uranium",
        "biomass",
        "oil",
        "lightoil",
        "fueloil",
        "surfacewater_basin",
        "salinewater_basin",
        "groundwater_basin",
        "freshwater_basin",
        "freshwater",
        "saline_ppl",
        "cl_fresh",
        "ot_fresh",
        "air",
        "ot_saline",
        "cl_saline",
        "urban_mw",
        "rural_mw",
        "industry_mw",
        "freshwater_supply",
        "saline_supply",
        "urban_collected_wst",
        "rural_collected_wst",
        "industry_collected_wst",
        "urban_uncollected_wst",
        "rural_uncollected_wst",
        "industry_uncollected_wst",
        "urban_treated",
        "rural_treated",
        "industry_treated",
        "urban_disconnected",
        "rural_disconnected",
        "industry_disconnected",
    ],
    "level": [
        "secondary",
        "primary",
        "final",
        "water_treat",
        "water_avail_basin",
        "water_supply_basin",
        "water_supply",
        "saline_supply",
        "share",
        "water_demand",
        "municipal_mw",
        "industry_mw",
        "irr_cereal",
        "irr_oilcrops",
        "irr_sugarcrops",
        "waste_management",
        "urban_discharge",
        "rural_discharge",
        "industry_discharge",
    ],
    "emission": ["fresh_return", "CO2", "water_consumption"],
    "technology": {
        "parent_power_techs": [
            "bio_hpl",
            "bio_istig",
            "bio_istig_ccs",
            "bio_ppl",
            "coal_adv",
            "coal_adv_ccs",
            "coal_ppl",
            "coal_ppl_u",
            "csp_sm1_ppl",
            "csp_sm3_ppl",
            "foil_hpl",
            "foil_ppl",
            "gas_cc",
            "gas_cc_ccs",
            "gas_ct",
            "gas_hpl",
            "gas_htfc",
            "gas_ppl",
            "geo_hpl",
            "geo_ppl",
            "hydro_1",
            "hydro_2",
            "hydro_3",
            "hydro_4",
            "hydro_5",
            "hydro_6",
            "hydro_7",
            "hydro_8",
            "hydro_hc",
            "hydro_lc",
            "igcc",
            "igcc_ccs",
            "loil_cc",
            "loil_ppl",
            "nuc_hc",
            "nuc_lc",
            "solar_res1",
            "solar_res2",
            "solar_res3",
            "solar_res4",
            "solar_res5",
            "solar_res6",
            "solar_res7",
            "solar_res8",
            "solar_res_hist_2000",
            "solar_res_hist_2005",
            "solar_res_hist_2010",
            "solar_res_hist_2015",
            "solar_res_hist_2020",
            "solar_res_hist_2025",
            "solar_resins",
            "wind_ref1",
            "wind_ref2",
            "wind_ref3",
            "wind_ref4",
            "wind_ref5",
            "wind_ref_hist_2000",
            "wind_ref_hist_2005",
            "wind_ref_hist_2010",
            "wind_ref_hist_2015",
            "wind_ref_hist_2020",
            "wind_ref_hist_2025",
            "wind_res1",
            "wind_res2",
            "wind_res3",
            "wind_res4",
            "wind_res_hist_2000",
            "wind_res_hist_2005",
            "wind_res_hist_2010",
            "wind_res_hist_2015",
            "wind_res_hist_2020",
            "wind_res_hist_2025",
            "csp_sm1_res",
            "csp_sm1_res1",
            "csp_sm1_res2",
            "csp_sm1_res3",
            "csp_sm1_res4",
            "csp_sm1_res5",
            "csp_sm1_res6",
            "csp_sm1_res7",
            "csp_sm1_res_hist_2010",
            "csp_sm1_res_hist_2015",
            "csp_sm1_res_hist_2020",
            "csp_sm3_res",
            "csp_sm3_res1",
            "csp_sm3_res2",
            "csp_sm3_res3",
            "csp_sm3_res4",
            "csp_sm3_res5",
            "csp_sm3_res6",
            "csp_sm3_res7",
            "solar_th_ppl",
        ],
        "water_supply": [
            "return_flow",
            "gw_recharge",
            "basin_to_reg",
            "extract_surfacewater",
            "extract_groundwater",
            "extract_gw_fossil",
            "extract_salinewater_cool",
            "extract_salinewater_basin",
        ],
        "infrastructure": [
            "urban_t_d",
            "rural_t_d",
            "industry_unconnected",
            "industry_untreated",
            "urban_unconnected",
            "rural_unconnected",
            "urban_sewerage",
            "urban_untreated",
            "urban_discharge",
            "urban_recycle",
            "rural_discharge",
            "rural_untreated",
            "rural_recycle",
            "rural_sewerage",
        ],
        "desalination": ["membrane", "distillation", "desal_t_d", "saline_ppl_t_d"],
        "cooling_types": [
            "__cl_fresh",
            "__ot_fresh",
            "__air",
            "__ot_saline",
            "__cl_saline",
        ],
        "irrigation": [
            "irrigation_oilcrops",
            "irrigation_sugarcrops",
            "irrigation_cereal",
        ],
    },
}

# Configuration for skipping data functions not supported by certain regions
SKIPPED_FUNCS_FOR_REGION = {
    "ZMB": [
        cool_tech,
        add_infrastructure_techs,
        add_desalination,
        add_sectoral_demands,
        add_water_availability,
        add_irr_structure,
    ]
}

# ==============================================================================
# Helper Functions
# ==============================================================================


def _add_filtered_par_data(scenario: Scenario, data: Dict[str, pd.DataFrame]):
    """Robustly add parameter data, filtering rows with null/NaN years."""
    for par_name, df in data.items():
        if not isinstance(df, pd.DataFrame):
            continue

        # Ensure node columns are strings for compatibility with message_ix
        for col in [c for c in df.columns if "node" in c]:
            df[col] = df[col].astype(str)

        # Filter out rows with invalid year entries
        for year_col in ["year_vtg", "year_act", "year"]:
            if year_col in df.columns and df[year_col].isnull().any():
                original_rows = len(df)
                df.dropna(subset=[year_col], inplace=True)
                filtered_rows = original_rows - len(df)
                if filtered_rows > 0:
                    print(
                        f"  - INFO: Filtered {filtered_rows} rows with invalid "
                        f"years from '{par_name}'."
                    )

    add_par_data(scenario, data, dry_run=False)


def _get_scenario_size_info(scenario: Scenario) -> Dict[str, int]:
    """Get scenario size information, including parameter counts and totals."""
    size_info = {"_total_all": 0, "_total_parameters": 0, "_total_sets": 0}

    for par in scenario.par_list():
        count = len(scenario.par(par))
        size_info[par] = count
        size_info["_total_parameters"] += count

    for set_name in scenario.set_list():
        try:
            size_info["_total_sets"] += len(scenario.set(set_name))
        except TypeError:
            pass  # Some sets might not have a length

    size_info["_total_all"] = size_info["_total_parameters"] + size_info["_total_sets"]
    return size_info


def _apply_and_commit_data_func(
    scenario: Scenario, context, data_func, comment: str, **kwargs
):
    """Apply a data function within a transaction and log the impact."""
    func_name = data_func.__name__
    print(f"\nApplying `{func_name}`...")

    size_before = _get_scenario_size_info(scenario)
    try:
        data = data_func(context, scenario=scenario, **kwargs)
        if not data:
            print(f"  - WARNING: `{func_name}` returned no data, skipping.")
            return

        with scenario.transact(comment):
            _add_filtered_par_data(scenario, data)

        size_after = _get_scenario_size_info(scenario)
        increase = size_after["_total_all"] - size_before["_total_all"]
        print(f"  - SUCCESS: `{func_name}` applied.")
        print(
            f"  - Scenario size: {size_before['_total_all']:,} -> "
            f"{size_after['_total_all']:,} elements (+{increase:,})"
        )

    except Exception as e:
        print(f"  - FAILED: `{func_name}` failed with error: {e}")
        traceback.print_exc()
        pytest.fail(f"`{func_name}` execution failed.")


def _add_items_to_set(scenario: Scenario, set_name: str, items: List[str]):
    """Add a list of items to a scenario set, avoiding duplicates."""
    existing_items = set(scenario.set(set_name))
    items_to_add = [item for item in items if item not in existing_items]
    if items_to_add:
        scenario.add_set(set_name, items_to_add)


# ==============================================================================
# Pytest Fixtures
# ==============================================================================


@pytest.fixture(scope="function", params=["ZMB", "R12"])
def water_context(test_context, request):
    """Fixture to configure the context for different regions."""
    region = request.param
    ctx = test_context
    ctx.SDG = "baseline"
    ctx.time = ["year"]
    ctx.type_reg = "global" if region == "R12" else "country"
    ctx.regions = region
    ctx.RCP = "7p0"
    ctx.REL = "low"
    ctx.nexus_set = "nexus"
    ctx.ssp = "SSP2"

    nodes = get_codes(f"node/{region}")
    nodes = list(map(str, nodes[nodes.index("World")].child))
    if ctx.type_reg == "country":
        ctx.map_ISO_c = {region: nodes[0]}

    read_config(ctx)
    return ctx


@pytest.fixture(scope="function")
def scenario_base_info(request) -> Dict:
    """Returns unique model/scenario names for test isolation."""
    node_name_slug = request.node.name.replace("[", "_").replace("]", "")[:40]
    return {
        "model": f"test_{node_name_slug}",
        "scenario": f"test_{node_name_slug}",
        "version": "new",
    }


@pytest.fixture(scope="function")
def water_basin_nodes(water_context) -> Tuple[List[str], List[str]]:
    """Generates basin node and mode names from data files."""
    region = water_context.regions
    basin_file = package_data_path(
        "water", "delineation", f"basins_by_region_simpl_{region}.csv"
    )
    basin_df = pd.read_csv(basin_file)
    basin_nodes = [f"B{bcu}" for bcu in basin_df["BCU_name"]]
    basin_modes = [f"M{bcu}" for bcu in basin_df["BCU_name"]]
    return basin_nodes, basin_modes


def _add_water_yaml_sets(scenario, water_context):
    """Simple function to add water sets from YAML without complex dependencies."""
    from message_ix_models.model.water.utils import read_config

    # Load the water configuration YAML files
    config = read_config(water_context)

    # Extract and add the cooling sets we need
    cooling_sets = config["water set"]["cooling"]

    # Add type_addon set if it exists - this is what we specifically need for the cooling technology issue
    if "type_addon" in cooling_sets and "add" in cooling_sets["type_addon"]:
        type_addon_elements = cooling_sets["type_addon"]["add"]
        _add_items_to_set(scenario, "type_addon", type_addon_elements)
        print(f"Added {len(type_addon_elements)} elements to type_addon set")

    # Add shares from cooling section
    if "shares" in cooling_sets and "add" in cooling_sets["shares"]:
        shares_elements = cooling_sets["shares"]["add"]
        _add_items_to_set(scenario, "shares", shares_elements)
        print(f"Added {len(shares_elements)} cooling shares elements")

    # Add shares from nexus section if it exists
    if water_context.nexus_set == "nexus":
        nexus_sets = config["water set"]["nexus"]
        if "shares" in nexus_sets and "add" in nexus_sets["shares"]:
            nexus_shares_elements = nexus_sets["shares"]["add"]
            _add_items_to_set(scenario, "shares", nexus_shares_elements)
            print(f"Added {len(nexus_shares_elements)} nexus shares elements")


@pytest.fixture(scope="function")
def prepared_scenario(water_context, scenario_base_info, water_basin_nodes):
    """
    Creates and prepares a base scenario for water model tests.

    This fixture handles:
    - Initializing the Scenario object.
    - Adding required units, sets, and years.
    - Setting up basin and regional nodes.
    - Adding all necessary water and power technologies using proper YAML approach.
    - Adding dummy input/output for parent power techs.
    - Attaching a ScenarioInfo object to the context.
    """
    mp = water_context.get_platform()
    s = Scenario(mp=mp, **scenario_base_info, annotation="Initial setup")

    # 1. Add units
    water_units = [
        "MCM",
        "MCM/year",
        "MCM/GWa",
        "USD/MCM",
        "GWh/MCM",
        "km3",
        "-",
        "USD/kW",
        "MUSD/GW",
        "USD/kWa",
        "GWa",
        "year",  # Added for technical_lifetime
        "%",  # Added for share parameters
    ]
    for unit in water_units:
        try:
            mp.add_unit(unit)
        except Exception:
            pass  # Unit may already exist

    # 2. Set up horizon and year categories
    s.add_horizon(year=list(range(1950, 2115, 5)))
    s.add_cat("year", "firstmodelyear", 2020)

    # 3. Add manual setup first (commodities, levels, technologies)
    for set_name, items in MODEL_SETUP_CONFIG.items():
        if isinstance(items, list):
            _add_items_to_set(s, set_name, items)
        elif isinstance(items, dict):
            all_items = []
            for group in items.values():
                all_items.extend(group)
            _add_items_to_set(s, set_name, all_items)

    # 4. Add water-specific sets from YAML (including type_addon)
    _add_water_yaml_sets(s, water_context)

    # 5. Add regional and basin nodes
    region_codes = get_codes(f"node/{water_context.regions}")
    regional_nodes = list(map(str, region_codes[region_codes.index("World")].child))
    _add_items_to_set(s, "node", regional_nodes)
    basin_nodes, basin_modes = water_basin_nodes
    _add_items_to_set(s, "node", basin_nodes)
    _add_items_to_set(
        s, "mode", ["M1", "Mf"] + basin_modes
    )  # Add M1 for basic technology parameters

    # 6. Add dummy parent technology data
    with s.transact("Add dummy parameters for parent power technologies"):
        node = regional_nodes[0]
        for tech in MODEL_SETUP_CONFIG["technology"]["parent_power_techs"]:
            fuel = "electr"
            if any(x in tech for x in ["coal", "igcc"]):
                fuel = "coal"
            elif "gas" in tech:
                fuel = "gas"
            elif "nuc" in tech:
                fuel = "uranium"
            elif "bio" in tech:
                fuel = "biomass"
            elif "foil" in tech:
                fuel = "fueloil"
            elif "loil" in tech:
                fuel = "lightoil"

            s.add_par(
                "input",
                pd.DataFrame(
                    [
                        {
                            "node_loc": node,
                            "technology": tech,
                            "year_vtg": 2020,
                            "year_act": 2020,
                            "mode": "M1",
                            "node_origin": node,
                            "commodity": fuel,
                            "level": "primary",
                            "time": "year",
                            "time_origin": "year",
                            "value": 1.0,
                            "unit": "GWa",
                        }
                    ]
                ),
            )
            s.add_par(
                "output",
                pd.DataFrame(
                    [
                        {
                            "node_loc": node,
                            "technology": tech,
                            "year_vtg": 2020,
                            "year_act": 2020,
                            "mode": "M1",
                            "node_dest": node,
                            "commodity": "electr",
                            "level": "secondary",
                            "time": "year",
                            "time_dest": "year",
                            "value": 1.0,
                            "unit": "GWa",
                        }
                    ]
                ),
            )

    # 7. Prepare context for data functions
    info = ScenarioInfo(s)
    info.y0 = 2020
    water_context["water build info"] = info
    water_context.set_scenario(s)

    return water_context, s


# ==============================================================================
# Main Test Functions
# ==============================================================================


@pytest.mark.usefixtures("ssp_user_data")
def test_full_water_build(prepared_scenario):
    """Test the complete water build process by calling functions sequentially."""
    # Check cache first
    context, scenario = prepared_scenario
    cache_key = f"{context.regions}_{context.ssp}_{context.nexus_set}"

    if cache_key in _built_scenario_cache:
        print(f"\n🚀 Using cached full water build for {cache_key}")
        return _built_scenario_cache[cache_key]

    print(f"\n=== Testing Full Water Build for Region: {context.regions} ===")

    # Add derived cooling technology names to the set
    with scenario.transact("Add cooling technology names to set"):
        cooling_techs = [
            f"{ptech}{ctype}"
            for ptech in MODEL_SETUP_CONFIG["technology"]["parent_power_techs"]
            for ctype in MODEL_SETUP_CONFIG["technology"]["cooling_types"]
        ]
        _add_items_to_set(scenario, "technology", cooling_techs)

    # List of all data functions to run in sequence
    data_functions = [
        (add_water_supply, "Applied water supply"),
        (add_e_flow, "Applied environmental flow"),
        (cool_tech, "Applied cooling tech data"),
        (add_infrastructure_techs, "Applied infrastructure data"),
        (add_desalination, "Applied desalination data"),
        (non_cooling_tec, "Applied non-cooling nexus data"),
        (add_sectoral_demands, "Applied sectoral water demands"),
        (add_water_availability, "Applied water availability constraints"),
        (add_irrigation_demand, "Applied irrigation demands"),
        (add_irr_structure, "Applied irrigation structure"),
    ]

    # Get list of functions to skip for the current region
    skipped_funcs = SKIPPED_FUNCS_FOR_REGION.get(context.regions, [])

    for func, comment in data_functions:
        if func in skipped_funcs:
            print(
                f"\n--- SKIPPING `{func.__name__}` for {context.regions} (not supported)."
            )
            continue
        _apply_and_commit_data_func(scenario, context, func, comment)

    print("\n✅ Full water build process completed.")
    print("--- Validating Results ---")

    input_df = scenario.par("input", {"technology": "extract_surfacewater"})
    assert not input_df.empty, "'extract_surfacewater' should have 'input' data"

    if context.regions == "R12":
        demand_df = scenario.par("demand", {"commodity": "urban_mw"})
        assert not demand_df.empty, "'urban_mw' commodity should have demand"

    # Cache the result, platform as well to avoid weakly referenced object issues.
    result = (context, scenario, scenario.platform)
    _built_scenario_cache[cache_key] = result
    print(f"💾 Cached full water build for {cache_key}")

    return result


@pytest.mark.usefixtures("ssp_user_data")
def test_water_system_mass_balance(prepared_scenario):
    """Test system-wide water balance: sources vs. sinks and transformation tech balance."""
    # --- 2. Check balance of individual transformation technologies ---
    _, scenario, _ = test_full_water_build(prepared_scenario)
    transformation_techs = {
        "membrane": ("saline_supply", "freshwater_supply"),
        "distillation": ("saline_supply", "freshwater_supply"),
        "urban_t_d": ("freshwater_supply", "urban_mw"),
        "rural_t_d": ("freshwater_supply", "rural_mw"),
    }

    imbalance_issues = []
    for tech, (in_comm, out_comm) in transformation_techs.items():
        if tech not in scenario.set("technology"):
            continue

        inputs = scenario.par("input", {"technology": tech, "commodity": in_comm})[
            "value"
        ].sum()
        outputs = scenario.par("output", {"technology": tech, "commodity": out_comm})[
            "value"
        ].sum()

        if inputs > 0:
            imbalance_pct = abs(inputs - outputs) / inputs * 100
            if imbalance_pct > 1:  # Allow 1% tolerance
                imbalance_issues.append(
                    f"'{tech}': Input={inputs:.2f}, Output={outputs:.2f} ({imbalance_pct:.1f}% imbalance)"
                )

    if imbalance_issues:
        pytest.fail(
            "Transformation technologies have significant imbalances:\n"
            + "\n".join(imbalance_issues)
        )
    else:
        print("✅ All transformation technologies have balanced input/output flows.")
