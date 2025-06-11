"""Prepare data for adding techs related to water distribution,
treatment in urban & rural"""

from collections import defaultdict
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import pandas as pd
from iam_units import registry
from message_ix import Scenario, make_df

from message_ix_models.util import (
    broadcast,
    make_matched_dfs,
    package_data_path,
    same_node,
    same_time,
)

if TYPE_CHECKING:
    from message_ix_models import Context

# Constants for water infrastructure processing
DISTRIBUTION_TECHNOLOGIES = [
    "urban_t_d",
    "urban_unconnected",
    "industry_unconnected",
    "rural_t_d",
    "rural_unconnected",
]

# Unit conversions using iam_units registry
# Convert USD/(m³/day) to USD/MCM: m³/day * 365 days/year / 1e6 m³/MCM
USD_M3DAY_TO_USD_MCM = (registry("m^3/day").to("m^3/year").magnitude) / 1e6
ANNUAL_CAPACITY_FACTOR = 5  # Convert 5-year capacity to annual
# Convert km³ to MCM: 1 km³ = 1e9 m³, 1 MCM = 1e6 m³, so factor = 1000
KM3_TO_MCM = registry("1 km^3").to("meter^3").magnitude / 1e6  # km³ to MCM conversion


def _load_basin_data(context: "Context") -> pd.DataFrame:
    """Load and prepare basin delineation data.

    Parameters
    ----------
    context : Context
        Context object with scenario information

    Returns
    -------
    pd.DataFrame
        Basin data with proper nomenclature
    """
    FILE2 = f"basins_by_region_simpl_{context.regions}.csv"
    PATH = package_data_path("water", "delineation", FILE2)

    df_node = pd.read_csv(PATH)
    # Assigning proper nomenclature
    df_node["node"] = "B" + df_node["BCU_name"].astype(str)
    df_node["mode"] = "M" + df_node["BCU_name"].astype(str)
    df_node["region"] = (
        context.map_ISO_c[context.regions]
        if context.type_reg == "country"
        else f"{context.regions}_" + df_node["REGION"].astype(str)
    )

    return df_node


def _load_infrastructure_data() -> pd.DataFrame:
    """Load water distribution mapping data from Excel file.

    Returns
    -------
    pd.DataFrame
        Infrastructure technology data
    """
    path = package_data_path("water", "infrastructure", "water_distribution.xlsx")
    return pd.read_excel(path)


def _load_desalination_data(
    context: "Context",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load desalination-related data files.

    Parameters
    ----------
    context : Context
        Context object with scenario information

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Desalination tech data, historical capacity data, projected capacity data
    """
    # Load technology data
    path = package_data_path("water", "infrastructure", "desalination.xlsx")
    df_desal = pd.read_excel(path)

    # Load historical capacity data
    path2 = package_data_path(
        "water",
        "infrastructure",
        f"historical_capacity_desalination_km3_year_{context.regions}.csv",
    )
    df_hist = pd.read_csv(path2)

    # Load projected capacity data
    path3 = package_data_path(
        "water",
        "infrastructure",
        f"projected_desalination_potential_km3_year_{context.regions}.csv",
    )
    df_proj = pd.read_csv(path3)
    df_proj = df_proj[df_proj["rcp"] == f"{context.RCP}"]
    df_proj = df_proj[~(df_proj["year"] == 2065) & ~(df_proj["year"] == 2075)]
    df_proj.reset_index(inplace=True, drop=True)

    return df_desal, df_hist, df_proj


def _process_input_parameters(
    context: "Context",
    df_node: pd.DataFrame,
    df_non_elec: pd.DataFrame,
    df_dist: pd.DataFrame,
    df_elec: pd.DataFrame,
    scenario: Scenario,
    sub_time: list,
) -> pd.DataFrame:
    """Process all input parameters for infrastructure technologies.

    Parameters
    ----------
    context : Context
        Context object with scenario information
    df_node : pd.DataFrame
        Basin data
    df_non_elec : pd.DataFrame
        Non-electric technology data
    df_dist : pd.DataFrame
        Distribution technology data
    df_elec : pd.DataFrame
        Electric technology data
    scenario : Scenario
        MESSAGEix scenario
    sub_time : list
        Time dimension data

    Returns
    -------
    pd.DataFrame
        Combined input parameter data
    """
    # Process non-electric inputs
    inp_df = start_creating_input_dataframe(
        sdg=context.SDG,
        df_node=df_node,
        df_non_elec=df_non_elec,
        df_dist=df_dist,
        scenario=scenario,
        sub_time=sub_time,
    )

    # Process electric inputs
    result_dc = prepare_input_dataframe(
        context=context,
        sub_time=sub_time,
        df_node=df_node,
        techs=DISTRIBUTION_TECHNOLOGIES,
        df_elec=df_elec,
        scenario=scenario,
    )

    if result_dc and "input" in result_dc:
        results_new = {
            par_name: pd.concat(dfs, ignore_index=True)
            for par_name, dfs in result_dc.items()
        }
        inp_df = pd.concat([inp_df, results_new["input"]], ignore_index=True)

    # Remove duplicates from input data
    if not inp_df.empty:
        inp_df = inp_df.drop_duplicates().reset_index(drop=True)

    return inp_df


def _process_output_parameters(
    context: "Context",
    df: pd.DataFrame,
    df_node: pd.DataFrame,
    scenario: Scenario,
    sub_time: list,
) -> pd.DataFrame:
    """Process output parameters for infrastructure technologies.

    Parameters
    ----------
    context : Context
        Context object
    df : pd.DataFrame
        Technology data
    df_node : pd.DataFrame
        Basin data
    scenario : Scenario
        MESSAGEix scenario
    sub_time : list
        Time dimension data

    Returns
    -------
    pd.DataFrame
        Output parameter data
    """
    # Process outputs with basin filtering
    df_out = df[~df["outcmd"].isna()]
    df_out_dist = df_out[df_out["tec"].isin(DISTRIBUTION_TECHNOLOGIES)]
    df_out = df_out[~df_out["tec"].isin(DISTRIBUTION_TECHNOLOGIES)]

    out_dfs = []

    # Process non-distribution technologies
    for _, row in df_out.iterrows():
        relevant_basins = filter_basins_for_technology(row["tec"], df_node, context)

        if len(relevant_basins) > 0:
            # Get vintage-activity years
            first_basin = relevant_basins.iloc[0]["node"]
            yv_ya = get_vintage_and_active_years(scenario, first_basin, row["tec"])

            out_df = (
                make_df(
                    "output",
                    technology=row["tec"],
                    value=row["out_value_mid"],
                    unit="-",
                    level=row["outlvl"],
                    commodity=row["outcmd"],
                    mode="M1",
                )
                .pipe(
                    broadcast,
                    labels=yv_ya,
                    node_loc=relevant_basins["node"],
                    time=sub_time,
                )
                .pipe(same_node)
                .pipe(same_time)
            )
            out_dfs.append(out_df)

    # Process distribution technologies
    mode = "Mf" if context.SDG != "baseline" else "M1"

    for _, row in df_out_dist.iterrows():
        relevant_basins = filter_basins_for_technology(row["tec"], df_node, context)

        if len(relevant_basins) > 0:
            first_basin = relevant_basins.iloc[0]["node"]
            yv_ya = get_vintage_and_active_years(scenario, first_basin, row["tec"])

            out_df = (
                make_df(
                    "output",
                    technology=row["tec"],
                    value=row["out_value_mid"],
                    unit="-",
                    level=row["outlvl"],
                    commodity=row["outcmd"],
                    mode=mode,
                )
                .pipe(
                    broadcast,
                    labels=yv_ya,
                    node_loc=relevant_basins["node"],
                    time=sub_time,
                )
                .pipe(same_node)
                .pipe(same_time)
            )
            out_dfs.append(out_df)

    if out_dfs:
        combined_out = pd.concat(out_dfs, ignore_index=True)
        return combined_out.drop_duplicates().reset_index(drop=True)
    else:
        return pd.DataFrame()


def _process_investment_costs(
    df: pd.DataFrame, df_node: pd.DataFrame, context: "Context", year_wat: tuple
) -> pd.DataFrame:
    """Process investment cost parameters."""
    df_inv = df.dropna(subset=["investment_mid"])
    inv_dfs = []

    for _, row in df_inv.iterrows():
        if row["tec"] not in DISTRIBUTION_TECHNOLOGIES:
            relevant_basins = filter_basins_for_technology(row["tec"], df_node, context)

            if len(relevant_basins) > 0:
                inv_cost = make_df(
                    "inv_cost",
                    technology=row["tec"],
                    value=row["investment_mid"] * USD_M3DAY_TO_USD_MCM,
                    unit="USD/MCM",
                ).pipe(broadcast, year_vtg=year_wat, node_loc=relevant_basins["node"])
                inv_dfs.append(inv_cost)

    if inv_dfs:
        combined_inv = pd.concat(inv_dfs, ignore_index=True)
        return combined_inv.drop_duplicates().reset_index(drop=True)
    else:
        return pd.DataFrame()


def _process_fixed_costs(
    df: pd.DataFrame,
    df_node: pd.DataFrame,
    context: "Context",
    scenario: Scenario,
) -> pd.DataFrame:
    """Process fixed cost parameters for all technologies."""
    df_inv = df.dropna(subset=["investment_mid"])
    fix_dfs = []

    for _, row in df_inv.iterrows():
        # Skip if no fixed cost data
        if "fix_cost_mid" not in row or pd.isna(row["fix_cost_mid"]):
            continue

        relevant_basins = filter_basins_for_technology(row["tec"], df_node, context)

        if len(relevant_basins) > 0:
            first_basin = relevant_basins.iloc[0]["node"]
            yv_ya = get_vintage_and_active_years(scenario, first_basin, row["tec"])

            fix_cost = make_df(
                "fix_cost",
                technology=row["tec"],
                value=row["fix_cost_mid"] * USD_M3DAY_TO_USD_MCM,
                unit="USD/MCM",
            ).pipe(broadcast, labels=yv_ya, node_loc=relevant_basins["node"])
            fix_dfs.append(fix_cost)

    if fix_dfs:
        combined_fix = pd.concat(fix_dfs, ignore_index=True)
        return combined_fix.drop_duplicates().reset_index(drop=True)
    else:
        return pd.DataFrame()


def _process_variable_costs(
    df: pd.DataFrame,
    df_node: pd.DataFrame,
    context: "Context",
    scenario: Scenario,
    sub_time: list,
) -> pd.DataFrame:
    """Process variable cost parameters for all technologies."""
    df_inv = df.dropna(subset=["investment_mid"])
    var_dfs = []

    for _, row in df_inv.iterrows():
        relevant_basins = filter_basins_for_technology(row["tec"], df_node, context)

        if len(relevant_basins) > 0:
            first_basin = relevant_basins.iloc[0]["node"]
            yv_ya = get_vintage_and_active_years(scenario, first_basin, row["tec"])

            # Determine value column and mode based on technology type and SDG scenario
            if row["tec"] in DISTRIBUTION_TECHNOLOGIES:
                value_col = (
                    "var_cost_high" if context.SDG != "baseline" else "var_cost_mid"
                )
                mode = "Mf" if context.SDG != "baseline" else "M1"
            else:
                value_col = "var_cost_mid"
                mode = "M1"

            var_cost = make_df(
                "var_cost",
                technology=row["tec"],
                value=row[value_col] * USD_M3DAY_TO_USD_MCM,
                unit="USD/MCM",
                mode=mode,
            ).pipe(
                broadcast, labels=yv_ya, node_loc=relevant_basins["node"], time=sub_time
            )
            var_dfs.append(var_cost)

    if var_dfs:
        combined_var = pd.concat(var_dfs, ignore_index=True)
        return combined_var.drop_duplicates().reset_index(drop=True)
    else:
        return pd.DataFrame()


def _process_setup_and_data_loading(
    context: "Context",
) -> tuple[pd.DataFrame, pd.DataFrame, list, tuple]:
    """Setup and load all necessary data for infrastructure processing."""
    # Reference to the water configuration
    info = context["water build info"]
    sub_time = context.time

    # Validate that context.time is a list to prevent character iteration issues
    if not isinstance(sub_time, list):
        raise TypeError(
            f"context.time must be a list, got {type(sub_time)}: {sub_time}"
        )

    year_wat = (*range(2010, info.Y[0] + 1, 5), *info.Y)

    # Load basin data
    df_node = _load_basin_data(context)

    # Load infrastructure data
    water_distribution = _load_infrastructure_data()

    return df_node, water_distribution, sub_time, year_wat


def _process_capacity_factor_parameters(
    df: pd.DataFrame,
    df_node: pd.DataFrame,
    context: "Context",
    scenario: Scenario,
    sub_time: list,
) -> pd.DataFrame:
    """Process capacity factor parameters for all technologies."""
    df_cap = df.dropna(subset=["capacity_factor_mid"])
    cap_dfs = []

    for _, row in df_cap.iterrows():
        relevant_basins = filter_basins_for_technology(row["tec"], df_node, context)

        if len(relevant_basins) > 0:
            first_basin = relevant_basins.iloc[0]["node"]
            yv_ya = get_vintage_and_active_years(scenario, first_basin, row["tec"])

            cap_df = (
                make_df(
                    "capacity_factor",
                    technology=row["tec"],
                    value=row["capacity_factor_mid"],
                    unit="%",
                )
                .pipe(
                    broadcast,
                    labels=yv_ya,
                    node_loc=relevant_basins["node"],
                    time=sub_time,
                )
                .pipe(same_node)
            )
            cap_dfs.append(cap_df)

    if cap_dfs:
        combined_cap = pd.concat(cap_dfs, ignore_index=True)
        return combined_cap.drop_duplicates().reset_index(drop=True)
    else:
        return pd.DataFrame()


def _process_technical_lifetime_parameters(
    df: pd.DataFrame,
    df_node: pd.DataFrame,
    context: "Context",
    year_wat: tuple,
) -> dict[str, pd.DataFrame]:
    """Process technical lifetime and construction time parameters."""
    df_tl = df.dropna(subset=["technical_lifetime_mid"])
    tl_dfs = []

    for _, row in df_tl.iterrows():
        relevant_basins = filter_basins_for_technology(row["tec"], df_node, context)

        if len(relevant_basins) > 0:
            tl = (
                make_df(
                    "technical_lifetime",
                    technology=row["tec"],
                    value=row["technical_lifetime_mid"],
                    unit="y",
                )
                .pipe(broadcast, year_vtg=year_wat, node_loc=relevant_basins["node"])
                .pipe(same_node)
            )
            tl_dfs.append(tl)

    results = {}
    if tl_dfs:
        tl_combined = pd.concat(tl_dfs, ignore_index=True)
        tl_combined = tl_combined.drop_duplicates().reset_index(drop=True)
        results["technical_lifetime"] = tl_combined

        cons_time = make_matched_dfs(tl_combined, construction_time=1)
        results["construction_time"] = cons_time["construction_time"]
    else:
        results["technical_lifetime"] = pd.DataFrame()
        results["construction_time"] = pd.DataFrame()

    return results


def _process_cost_parameters(
    context: "Context",
    df: pd.DataFrame,
    df_node: pd.DataFrame,
    scenario: Scenario,
    sub_time: list,
    year_wat: tuple,
) -> dict[str, pd.DataFrame]:
    """Process cost parameters for infrastructure technologies."""
    results = {}

    # Process each cost type separately
    results["inv_cost"] = _process_investment_costs(df, df_node, context, year_wat)
    results["fix_cost"] = _process_fixed_costs(df, df_node, context, scenario)
    results["var_cost"] = _process_variable_costs(
        df, df_node, context, scenario, sub_time
    )

    return results


# Cache for vintage and active years to avoid repeated computation
@lru_cache(maxsize=256)
def get_vintage_and_active_years(
    scenario: Scenario, node_loc: str, technology: str
) -> pd.DataFrame:
    """Get vintage and active years using standard MESSAGEix approach.

    This replaces the problematic map_yv_ya_lt function that was generating
    excessive combinations and allowing historical vintages in model periods.
    """
    try:
        yv_ya = scenario.vintage_and_active_years((node_loc, technology))

        # The MESSAGEix method returns a DataFrame directly
        if not yv_ya.empty:
            return yv_ya
    except (ValueError, KeyError):
        # No technical lifetime data available for this technology/node combination
        pass

    # Fallback: create minimal valid combinations using only model years
    else:
        # Return empty DataFrame with correct columns if no valid years
        return pd.DataFrame(columns=["year_vtg", "year_act"])


def filter_basins_for_technology(
    technology: str, all_basins: pd.DataFrame, context: "Context"
) -> pd.DataFrame:
    """Filter basins relevant for a specific technology.

    This prevents inappropriate technology-basin combinations like
    desalination in landlocked areas.

    Parameters
    ----------
    technology : str
        Technology name
    all_basins : pd.DataFrame
        DataFrame with all basin information
    context : Context
        Context object with scenario information

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only relevant basins
    """
    # FIXME: Implement detailed basin-technology mapping based on:
    # - Basin characteristics (coastal vs inland)
    # - Technology applicability (e.g., desalination only in coastal)
    # - Economic feasibility constraints
    # - Regional technology availability

    # For now, return all basins but log warning
    # Handle case where technology might be NaN or not a string
    if pd.isna(technology) or not isinstance(technology, str):
        return pd.DataFrame()  # Return empty DataFrame for invalid technologies

    if "desal" in technology.lower():
        # FIXME: Filter to only coastal basins once basin classification data available
        pass
    elif "urban" in technology.lower():
        # FIXME: Filter to only urban/mixed basins
        pass
    elif "rural" in technology.lower():
        # FIXME: Filter to only rural/mixed basins
        pass

    # Placeholder: return all basins for now
    return all_basins


def create_input_data_for_technology(
    technology_row: pd.Series,
    relevant_basins: pd.DataFrame,
    scenario: Scenario,
    sub_time: str,
    mode: str = "M1",
) -> pd.DataFrame:
    """Create input DataFrame for a single technology.

    This consolidates the logic for creating input data and ensures
    proper vintage-activity year combinations.
    """
    # Get vintage-activity years for first basin (they should be same for all)
    if len(relevant_basins) > 0:
        first_basin = relevant_basins.iloc[0]["node"]
        yv_ya = get_vintage_and_active_years(
            scenario, first_basin, technology_row["tec"]
        )
    else:
        return pd.DataFrame()

    # Create base DataFrame
    base_df = make_df(
        "input",
        technology=technology_row["tec"],
        value=technology_row.get("value_mid", technology_row.get("value_high", 0)),
        unit=technology_row.get("unit", "-"),
        level=technology_row["inlvl"],
        commodity=technology_row["incmd"],
        mode=mode,
        node_loc=relevant_basins["node"],
    )

    # Broadcast with proper vintage-activity years
    # Use labels parameter for vintage-activity combinations, then keyword args
    return (
        base_df.pipe(broadcast, labels=yv_ya, time=sub_time)
        .pipe(same_node)
        .pipe(same_time)
    )


def start_creating_input_dataframe(
    sdg: str,
    df_node: pd.DataFrame,
    df_non_elec: pd.DataFrame,
    df_dist: pd.DataFrame,
    scenario: Scenario,
    sub_time,
) -> pd.DataFrame:
    """Creates an input pd.DataFrame with proper basin filtering."""
    inp_dfs = []

    # Process non-electric commodities
    for _, row in df_non_elec.iterrows():
        # Filter basins for this technology
        relevant_basins = filter_basins_for_technology(
            row["tec"], df_node, scenario.platform
        )

        if len(relevant_basins) > 0:
            inp_df = create_input_data_for_technology(
                row, relevant_basins, scenario, sub_time
            )
            inp_dfs.append(inp_df)

    # Process distribution technologies based on SDG scenario
    mode = "Mf" if sdg != "baseline" else "M1"
    value_col = "value_high" if sdg != "baseline" else "value_mid"

    for _, row in df_dist.iterrows():
        # Update row with appropriate value
        row_copy = row.copy()
        row_copy["value_mid"] = row[value_col]

        relevant_basins = filter_basins_for_technology(
            row["tec"], df_node, scenario.platform
        )

        if len(relevant_basins) > 0:
            inp_df = create_input_data_for_technology(
                row_copy, relevant_basins, scenario, sub_time, mode=mode
            )
            inp_dfs.append(inp_df)

    # Combine all DataFrames and remove duplicates
    if inp_dfs:
        combined = pd.concat(inp_dfs, ignore_index=True)
        return combined.drop_duplicates().reset_index(drop=True)
    else:
        return pd.DataFrame()


def add_infrastructure_techs(context: "Context") -> dict[str, pd.DataFrame]:
    """Process water distribution data for a scenario instance.

    Parameters
    ----------
    context : .Context

    Returns
    -------
    data : dict of (str -> pandas.DataFrame)
        Keys are MESSAGE parameter names such as 'input', 'fix_cost'.
        Values are data frames ready for :meth:`~.Scenario.add_par`.
        Years in the data include the model horizon indicated by
        ``context["water build info"]``, plus the additional year 2010.
    """
    # Step 1: Setup and load data
    df_node, df, sub_time, year_wat = _process_setup_and_data_loading(context)
    scen = context.get_scenario()
    results = {}

    # Step 2: Prepare data splits and process input parameters
    df_non_elec = df[df["incmd"] != "electr"].reset_index()
    df_dist = df_non_elec[df_non_elec["tec"].isin(DISTRIBUTION_TECHNOLOGIES)]
    df_non_elec = df_non_elec[~df_non_elec["tec"].isin(DISTRIBUTION_TECHNOLOGIES)]
    df_elec = df[df["incmd"] == "electr"].reset_index()

    results["input"] = _process_input_parameters(
        context, df_node, df_non_elec, df_dist, df_elec, scen, sub_time
    )

    # Step 3: Process output parameters
    results["output"] = _process_output_parameters(context, df, df_node, scen, sub_time)

    # Step 4: Process capacity factor parameters
    results["capacity_factor"] = _process_capacity_factor_parameters(
        df, df_node, context, scen, sub_time
    )

    # Step 5: Process technical lifetime and construction time
    lifetime_results = _process_technical_lifetime_parameters(
        df, df_node, context, year_wat
    )
    results.update(lifetime_results)

    # Step 6: Process cost parameters
    cost_results = _process_cost_parameters(
        context, df, df_node, scen, sub_time, year_wat
    )
    results.update(cost_results)

    return results


def prepare_input_dataframe(
    context: "Context",
    sub_time,
    df_node: pd.DataFrame,
    techs: list[str],
    df_elec: pd.DataFrame,
    scenario: Scenario,
) -> defaultdict[Any, list]:
    """Prepare electricity input data with proper basin filtering."""
    result_dc = defaultdict(list)

    for _, row in df_elec.iterrows():
        # Filter basins for this technology
        relevant_basins = filter_basins_for_technology(row["tec"], df_node, context)

        if len(relevant_basins) == 0:
            continue

        # Get vintage-activity years
        first_basin = relevant_basins.iloc[0]["node"]
        yv_ya = get_vintage_and_active_years(scenario, first_basin, row["tec"])

        if row["tec"] in techs:
            # Distribution technologies
            if context.SDG != "baseline":
                # SDG scenario - only Mf mode with high value
                inp = make_df(
                    "input",
                    technology=row["tec"],
                    value=row["value_high"],
                    unit="GWh/MCM",
                    level="final",
                    commodity="electr",
                    mode="Mf",
                    time_origin="year",
                    node_loc=relevant_basins["node"],
                    node_origin=relevant_basins["region"],
                ).pipe(broadcast, labels=yv_ya, time=sub_time)

                result_dc["input"].append(inp)
            else:
                # Baseline scenario - both M1 and Mf modes
                # M1 mode with mid value
                inp_m1 = make_df(
                    "input",
                    technology=row["tec"],
                    value=row["value_mid"],
                    unit="GWh/MCM",
                    level="final",
                    commodity="electr",
                    mode="M1",
                    time_origin="year",
                    node_loc=relevant_basins["node"],
                    node_origin=relevant_basins["region"],
                ).pipe(broadcast, labels=yv_ya, time=sub_time)

                # Mf mode with high value
                inp_mf = make_df(
                    "input",
                    technology=row["tec"],
                    value=row["value_high"],
                    unit="GWh/MCM",
                    level="final",
                    commodity="electr",
                    mode="Mf",
                    time_origin="year",
                    node_loc=relevant_basins["node"],
                    node_origin=relevant_basins["region"],
                ).pipe(broadcast, labels=yv_ya, time=sub_time)

                result_dc["input"].extend([inp_m1, inp_mf])
        else:
            # Non-distribution technologies
            inp = make_df(
                "input",
                technology=row["tec"],
                value=row["value_mid"],
                unit="GWh/MCM",
                level="final",
                commodity="electr",
                mode="M1",
                time_origin="year",
                node_loc=relevant_basins["node"],
                node_origin=relevant_basins["region"],
            ).pipe(broadcast, labels=yv_ya, time=sub_time)

            result_dc["input"].append(inp)

    return result_dc


def _process_desalination_setup_and_data_loading(
    context: "Context",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list, tuple]:
    """Setup and load desalination data and configurations."""
    info = context["water build info"]
    sub_time = context.time

    # Validate that context.time is a list to prevent character iteration issues
    if not isinstance(sub_time, list):
        raise TypeError(
            f"context.time must be a list, got {type(sub_time)}: {sub_time}"
        )

    year_wat = (*range(2010, info.Y[0] + 1, 5), *info.Y)

    # Load desalination data
    df_desal, df_hist, df_proj = _load_desalination_data(context)
    df_proj = df_proj[df_proj["year"].isin(info.Y)]

    # Load basin data
    df_node = _load_basin_data(context)

    return df_desal, df_hist, df_proj, df_node, sub_time, year_wat


def _process_saline_water_extraction(
    desal_basins: pd.DataFrame,
    scenario: "Scenario",
    sub_time: list,
    year_wat: tuple,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process saline water extraction output and technical lifetime."""
    if len(desal_basins) > 0:
        first_basin = desal_basins.iloc[0]["node"]
        yv_ya = get_vintage_and_active_years(
            scenario, first_basin, "extract_salinewater_basin"
        )

        out_df = (
            make_df(
                "output",
                technology="extract_salinewater_basin",
                value=1,
                unit="MCM/year",
                level="water_avail_basin",
                commodity="salinewater_basin",
                mode="M1",
            )
            .pipe(broadcast, labels=yv_ya, node_loc=desal_basins["node"], time=sub_time)
            .pipe(same_node)
            .pipe(same_time)
        )

        # Technical lifetime for extraction
        tl = (
            make_df(
                "technical_lifetime",
                technology="extract_salinewater_basin",
                value=20,
                unit="y",
            )
            .pipe(broadcast, year_vtg=year_wat, node_loc=desal_basins["node"])
            .pipe(same_node)
        )
        return out_df, tl
    else:
        return pd.DataFrame(), pd.DataFrame()


def _process_historical_capacity_data(
    df_hist: pd.DataFrame, desal_basins: pd.DataFrame
) -> pd.DataFrame:
    """Process historical desalination capacity data."""
    if not df_hist.empty:
        # Filter to only basins in desal_basins
        hist_basins = "B" + df_hist["BCU_name"]
        valid_hist = df_hist[hist_basins.isin(desal_basins["node"])]

        if not valid_hist.empty:
            return make_df(
                "historical_new_capacity",
                node_loc="B" + valid_hist["BCU_name"],
                technology=valid_hist["tec_type"],
                year_vtg=valid_hist["year"],
                value=valid_hist["cap_km3_year"] * KM3_TO_MCM / ANNUAL_CAPACITY_FACTOR,
                unit="MCM/year",
            )
    return pd.DataFrame()


def _process_projected_capacity_bounds(
    df_proj: pd.DataFrame, desal_basins: pd.DataFrame
) -> pd.DataFrame:
    """Process projected desalination capacity bounds."""
    if not df_proj.empty:
        proj_basins = "B" + df_proj["BCU_name"]
        valid_proj = df_proj[proj_basins.isin(desal_basins["node"])]

        if not valid_proj.empty:
            bound_up = make_df(
                "bound_total_capacity_up",
                node_loc="B" + valid_proj["BCU_name"],
                technology="extract_salinewater_basin",
                year_act=valid_proj["year"],
                value=valid_proj["cap_km3_year"] * KM3_TO_MCM,
                unit="MCM/year",
            )
            bound_up["value"] = bound_up["value"].clip(lower=0)
            bound_up = bound_up[bound_up["year_act"] > 2020]
            return bound_up
    return pd.DataFrame()


def _process_desalination_technology_parameters(
    df_desal: pd.DataFrame,
    desal_basins: pd.DataFrame,
    context: "Context",
    scenario: "Scenario",
    sub_time: list,
    year_wat: tuple,
) -> dict[str, list]:
    """Process all parameters for desalination technologies."""
    inv_dfs = []
    fix_dfs = []
    var_dfs = []
    inp_dfs = []
    out_dfs = []
    tl_dfs = []

    for _, row in df_desal.iterrows():
        # Get relevant basins for this desalination technology
        tech_basins = filter_basins_for_technology(row["tec"], desal_basins, context)

        if len(tech_basins) == 0:
            continue

        # Get vintage-activity years
        first_basin = tech_basins.iloc[0]["node"]
        yv_ya = get_vintage_and_active_years(scenario, first_basin, row["tec"])

        # Investment cost
        inv_cost = make_df(
            "inv_cost",
            technology=row["tec"],
            value=row["inv_cost_mid"] * USD_M3DAY_TO_USD_MCM,
            unit="USD/MCM",
        ).pipe(broadcast, year_vtg=year_wat, node_loc=tech_basins["node"])
        inv_dfs.append(inv_cost)

        # Fixed cost
        fix_cost = make_df(
            "fix_cost",
            technology=row["tec"],
            value=row["fix_cost_mid"] * USD_M3DAY_TO_USD_MCM,
            unit="USD/MCM",
        ).pipe(broadcast, labels=yv_ya, node_loc=tech_basins["node"])
        fix_dfs.append(fix_cost)

        # Variable cost
        var_cost = make_df(
            "var_cost",
            technology=row["tec"],
            value=row["var_cost_mid"] * USD_M3DAY_TO_USD_MCM,
            unit="USD/MCM",
            mode="M1",
        ).pipe(broadcast, labels=yv_ya, node_loc=tech_basins["node"], time=sub_time)
        var_dfs.append(var_cost)

        # Electricity input
        if row["electricity_input_mid"] > 0:
            inp_elec = make_df(
                "input",
                technology=row["tec"],
                value=row["electricity_input_mid"],
                unit="-",
                level="final",
                commodity="electr",
                mode="M1",
                time_origin="year",
                node_loc=tech_basins["node"],
                node_origin=tech_basins["region"],
            ).pipe(broadcast, labels=yv_ya, time=sub_time)
            inp_dfs.append(inp_elec)

        # Heat input (if applicable)
        if "heat_input_mid" in row and row["heat_input_mid"] > 0:
            inp_heat = make_df(
                "input",
                technology=row["tec"],
                value=row["heat_input_mid"],
                unit="-",
                level="final",
                commodity="d_heat",
                mode="M1",
                time_origin="year",
                node_loc=tech_basins["node"],
                node_origin=tech_basins["region"],
            ).pipe(broadcast, labels=yv_ya, time=sub_time)
            inp_dfs.append(inp_heat)

        # Main input (saline water)
        inp_main = (
            make_df(
                "input",
                technology=row["tec"],
                value=1,
                unit="-",
                level=row["inlvl"],
                commodity=row["incmd"],
                mode="M1",
            )
            .pipe(broadcast, labels=yv_ya, node_loc=tech_basins["node"], time=sub_time)
            .pipe(same_node)
            .pipe(same_time)
        )
        inp_dfs.append(inp_main)

        # Output
        out = (
            make_df(
                "output",
                technology=row["tec"],
                value=1,
                unit="-",
                level=row["outlvl"],
                commodity=row["outcmd"],
                mode="M1",
            )
            .pipe(broadcast, labels=yv_ya, node_loc=tech_basins["node"], time=sub_time)
            .pipe(same_node)
            .pipe(same_time)
        )
        out_dfs.append(out)

        # Technical lifetime
        tl_tech = (
            make_df(
                "technical_lifetime",
                technology=row["tec"],
                value=row["lifetime_mid"],
                unit="y",
            )
            .pipe(broadcast, year_vtg=year_wat, node_loc=tech_basins["node"])
            .pipe(same_node)
        )
        tl_dfs.append(tl_tech)

    return {
        "inv_dfs": inv_dfs,
        "fix_dfs": fix_dfs,
        "var_dfs": var_dfs,
        "inp_dfs": inp_dfs,
        "out_dfs": out_dfs,
        "tl_dfs": tl_dfs,
    }


def _process_lower_bounds(
    df_hist: pd.DataFrame,
    desal_basins: pd.DataFrame,
    year_wat: tuple,
    sub_time: list,
) -> pd.DataFrame:
    """Process lower bounds based on historical capacity."""
    if not df_hist.empty:
        df_bound = df_hist[df_hist["year"] == 2025]
        bound_basins = "B" + df_bound["BCU_name"]
        valid_bound = df_bound[bound_basins.isin(desal_basins["node"])]

        if not valid_bound.empty:
            bound_lo = make_df(
                "bound_activity_lo",
                node_loc="B" + valid_bound["BCU_name"],
                technology=valid_bound["tec_type"],
                mode="M1",
                value=valid_bound["cap_km3_year"] * KM3_TO_MCM / ANNUAL_CAPACITY_FACTOR,
                unit="MCM/year",
            ).pipe(broadcast, year_act=year_wat, time=sub_time)

            bound_lo = bound_lo[bound_lo["year_act"] <= 2040]
            return bound_lo
    return pd.DataFrame()


def add_desalination(context: "Context") -> dict[str, pd.DataFrame]:
    """Add desalination infrastructure with proper basin filtering.

    Two types of desalination are considered:
    1. Membrane
    2. Distillation

    Parameters
    ----------
    context : .Context

    Returns
    -------
    data : dict of (str -> pandas.DataFrame)
        Keys are MESSAGE parameter names such as 'input', 'fix_cost'.
        Values are data frames ready for :meth:`~.Scenario.add_par`.
    """
    # Step 1: Setup and load data
    df_desal, df_hist, df_proj, df_node, sub_time, year_wat = (
        _process_desalination_setup_and_data_loading(context)
    )
    scen = context.get_scenario()
    results = {}

    # Step 2: Filter basins for desalination (coastal only)
    # FIXME: For now, using all basins but this should be restricted
    desal_basins = filter_basins_for_technology("desalination", df_node, context)

    # Step 3: Process saline water extraction
    out_df, tl = _process_saline_water_extraction(
        desal_basins, scen, sub_time, year_wat
    )
    results["output"] = out_df

    # Step 4: Process historical capacity data
    hist_cap = _process_historical_capacity_data(df_hist, desal_basins)
    if not hist_cap.empty:
        results["historical_new_capacity"] = hist_cap

    # Step 5: Process projected capacity bounds
    bound_up = _process_projected_capacity_bounds(df_proj, desal_basins)
    if not bound_up.empty:
        results["bound_total_capacity_up"] = bound_up

    # Step 6: Process desalination technology parameters
    tech_params = _process_desalination_technology_parameters(
        df_desal, desal_basins, context, scen, sub_time, year_wat
    )

    # Step 7: Combine technology parameter results
    if tech_params["inv_dfs"]:
        results["inv_cost"] = pd.concat(tech_params["inv_dfs"], ignore_index=True)
    if tech_params["fix_dfs"]:
        results["fix_cost"] = pd.concat(tech_params["fix_dfs"], ignore_index=True)
    if tech_params["var_dfs"]:
        results["var_cost"] = pd.concat(tech_params["var_dfs"], ignore_index=True)
    if tech_params["inp_dfs"]:
        results["input"] = pd.concat(tech_params["inp_dfs"], ignore_index=True)
    if tech_params["out_dfs"]:
        if "output" in results:
            results["output"] = pd.concat(
                [results["output"]] + tech_params["out_dfs"], ignore_index=True
            )
        else:
            results["output"] = pd.concat(tech_params["out_dfs"], ignore_index=True)

    # Step 8: Process technical lifetime and construction time
    if tech_params["tl_dfs"] or not tl.empty:
        all_tl = [tl] if not tl.empty else []
        all_tl.extend(tech_params["tl_dfs"])
        tl_combined = pd.concat(all_tl, ignore_index=True)
        results["technical_lifetime"] = tl_combined

        cons_time = make_matched_dfs(tl_combined, construction_time=3)
        results["construction_time"] = cons_time["construction_time"]

    # Step 9: Process lower bounds
    bound_lo = _process_lower_bounds(df_hist, desal_basins, year_wat, sub_time)
    if not bound_lo.empty:
        results["bound_activity_lo"] = bound_lo

    return results
