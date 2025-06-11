"""Prepare data for adding demands"""

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Union

import numpy as np
import pandas as pd
import xarray as xr
from iam_units import registry
from message_ix import make_df

from message_ix_models.util import broadcast, minimum_version, package_data_path

if TYPE_CHECKING:
    from message_ix_models import Context

# Constants for water demands processing
SDG_RURAL_SANITATION_TARGET = 0.8
SDG_URBAN_SANITATION_TARGET = 0.95
SDG_URBAN_CONNECTION_TARGET = 0.99
SDG_RURAL_CONNECTION_TARGET = 0.8

# Time periods for interpolation
INTERPOLATION_YEARS = [2015, 2025, 2035, 2045, 2055]

# Unit conversions
MONTHLY_CONVERSION = (
    (30 * registry.day / registry.month).to_base_units().magnitude
)  # MCM/day to MCM/month
KM3_2_MCM = (
    registry("1 km^3").to("meter^3").magnitude / 1e6
    # Convert km³/year to MCM/year (1 km³ = 1e9 m³, 1 MCM = 1e6 m³, so factor = 1000)
)


def get_basin_sizes(
    basin: pd.DataFrame, node: str
) -> Sequence[Union[pd.Series, Literal[0]]]:
    """Returns the sizes of developing and developed basins for a given node"""
    temp = basin[basin["BCU_name"] == node]
    sizes = temp.pivot_table(index=["STATUS"], aggfunc="size")
    # sizes_### = sizes["###"] if "###" in sizes.index else 0
    sizes_dev = sizes["DEV"] if "DEV" in sizes.index else 0
    sizes_ind = sizes["IND"] if "IND" in sizes.index else 0
    return_tuple: tuple[Union[pd.Series, Literal[0]], Union[pd.Series, Literal[0]]] = (
        sizes_dev,
        sizes_ind,
    )  # type: ignore # Somehow, mypy is unable to recognize the proper type without forcing it
    return return_tuple


@minimum_version("python 3.10")
def set_target_rate(
    df: pd.DataFrame,
    strategy: str,
    target: float,
    basin: pd.DataFrame | None = None,
    node: str | None = None,
    year: int | None = None,
) -> pd.DataFrame:
    """
    Unified function for setting target rates using different strategies.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with water demand data
    strategy : str
        Strategy to use: "direct", "connection", or "treatment"
    target : float
        Target value to set
    basin : pd.DataFrame, optional
        Basin classification data (required for "connection" and "treatment")
    node : str, optional
        Specific node (required for "direct")
    year : int, optional
        Specific year (required for "direct")

    Returns
    -------
    pd.DataFrame
        Modified DataFrame with updated target rates
    """
    match strategy:
        case "direct":
            # Direct target setting for specific node and year
            if node is None or year is None:
                raise ValueError(
                    "'direct' strategy requires both 'node' and 'year' parameters"
                )
            _set_direct_target(df, node, year, target)
            return df

        case "connection":
            # Connection rates based on basin development classification
            if basin is None:
                raise ValueError("'connection' strategy requires 'basin' parameter")
            _set_connection_targets(df, basin, target)
            return df

        case "treatment":
            # Treatment rates with wastewater halving logic
            if basin is None:
                raise ValueError("'treatment' strategy requires 'basin' parameter")
            return _set_treatment_targets(df, basin)

        case _:
            raise ValueError(
                f"Unknown strategy: {
                    strategy
                }. Use 'direct', 'connection', or 'treatment'"
            )


def _set_direct_target(df: pd.DataFrame, node: str, year: int, target: float) -> None:
    """Set target value for a specific node and year
    (only if current value is lower)."""
    mask = (df["node"] == node) & (df["year"] == year)
    indices = df[mask].index

    for index in indices:
        if df.at[index, "value"] < target:
            df.at[index, "value"] = target


def _set_connection_targets(
    df: pd.DataFrame, basin: pd.DataFrame, target: float
) -> None:
    """Set connection rate targets based on basin development classification."""
    for node in df.node.unique():
        dev_size, ind_size = get_basin_sizes(basin, node)

        match dev_size >= ind_size:
            case True:
                # Developed basin: set target for 2030
                _set_direct_target(df, node, 2030, target)
            case False:
                # Developing basin: set graduated targets for 2035 and 2040
                # First get 2030 value
                mask_2030 = (df["node"] == node) & (df["year"] == 2030)
                if mask_2030.any():
                    value_2030 = df[mask_2030]["value"].iloc[0]
                    # Set intermediate target for 2035
                    # (average of 2030 and final target)
                    _set_direct_target(df, node, 2035, (value_2030 + target) / 2)
                    # Set final target for 2040
                    _set_direct_target(df, node, 2040, target)


def _set_treatment_targets(df: pd.DataFrame, basin: pd.DataFrame) -> pd.DataFrame:
    """
    Set treatment rate targets to halve untreated wastewater.

    For developed regions: halve untreated wastewater from 2030 onwards
    For developing regions: halve untreated wastewater from 2040 onwards
    """
    df = df.copy()
    updates = []

    for node in df.node.unique():
        temp = basin[basin["BCU_name"] == node]

        if temp.empty:
            continue

        sizes = temp.pivot_table(index=["STATUS"], aggfunc="size")

        # Determine if basin is primarily developing
        is_developing = _is_developing_basin(sizes)

        # Set year threshold based on development status
        year_threshold = 2040 if is_developing else 2030

        # Apply treatment rate improvements
        node_mask = df["node"] == node
        year_mask = df["year"] >= year_threshold
        target_rows = df[node_mask & year_mask]

        for index in target_rows.index:
            current_value = df.at[index, "value"]
            # Halve untreated wastewater: new_rate = current + (1 - current) / 2
            new_value = current_value + (1 - current_value) / 2
            updates.append((index, new_value))

    # Apply all updates
    for index, new_value in updates:
        df.at[index, "value"] = np.float64(new_value)

    return df


def _is_developing_basin(sizes: pd.Series) -> bool:
    """Determine if a basin is primarily developing based on country classification."""
    if len(sizes) > 1:
        # Mixed basin: developing if DEV >= IND
        return sizes.get("DEV", 0) >= sizes.get("IND", 0)
    else:
        # Single classification basin
        return sizes.index[0] == "DEV" if len(sizes) > 0 else True


def _load_and_process_demand_data(context: "Context") -> pd.DataFrame:
    """Load and process water demand data from CSV files with interpolation."""
    region = f"{context.regions}"
    path = package_data_path("water", "demands", "harmonized", region, ".")

    # Load all CSV files matching the pattern
    list_of_csvs = list(path.glob("ssp2_regional_*.csv"))
    if not list_of_csvs:
        raise FileNotFoundError(f"No demand data files found in {path}")

    # Extract variable names from filenames
    fns = [os.path.splitext(os.path.basename(x))[0] for x in list_of_csvs]
    fns = " ".join(fns).replace("ssp2_regional_", "").split()

    # Read CSV files into dictionary
    d: dict[str, pd.DataFrame] = {}
    for i, fn in enumerate(fns):
        d[fn] = pd.read_csv(list_of_csvs[i])

    # Process dataframes: standardize year column and set index
    dfs = {}
    for key, df in d.items():
        df.rename(columns={"Unnamed: 0": "year"}, inplace=True)
        df.set_index("year", inplace=True)
        dfs[key] = df

    # Convert to xarray and interpolate missing years
    df_x = xr.Dataset(dfs).to_array()
    df_x_interp = df_x.interp(year=INTERPOLATION_YEARS)
    df_x_c = df_x.combine_first(df_x_interp)

    # Convert back to pandas and format for MESSAGE
    df_f = df_x_c.to_dataframe("").unstack()
    df_dmds = df_f.stack(future_stack=True).reset_index(level=0).reset_index()
    df_dmds.columns = ["year", "node", "variable", "value"]
    df_dmds.sort_values(["year", "node", "variable", "value"], inplace=True)
    df_dmds["time"] = "year"

    # Handle sub-annual timesteps (monthly data)
    if "year" not in context.time:
        df_dmds = _process_monthly_data(df_dmds, region)

    return df_dmds


def _process_monthly_data(df_dmds: pd.DataFrame, region: str) -> pd.DataFrame:
    """Process monthly water demand data for sub-annual timesteps."""
    PATH = package_data_path(
        "water", "demands", "harmonized", region, "ssp2_m_water_demands.csv"
    )
    df_m = pd.read_csv(PATH)

    # Convert from MCM/day to MCM/month using pint
    df_m.value *= MONTHLY_CONVERSION

    # Standardize sector naming
    df_m.loc[df_m["sector"] == "industry", "sector"] = "manufacturing"
    df_m["variable"] = df_m["sector"] + "_" + df_m["type"] + "_baseline"

    # Handle urban variable naming inconsistency
    df_m.loc[df_m["variable"] == "urban_withdrawal_baseline", "variable"] = (
        "urban_withdrawal2_baseline"
    )
    df_m.loc[df_m["variable"] == "urban_return_baseline", "variable"] = (
        "urban_return2_baseline"
    )

    # Format for concatenation
    df_m = df_m[["year", "pid", "variable", "value", "month"]]
    df_m.columns = pd.Index(["year", "node", "variable", "value", "time"])

    # Remove yearly data that will be replaced with monthly
    monthly_variables = [
        "urban_withdrawal2_baseline",
        "rural_withdrawal_baseline",
        "manufacturing_withdrawal_baseline",
        "manufacturing_return_baseline",
        "urban_return2_baseline",
        "rural_return_baseline",
    ]
    df_dmds = df_dmds[~df_dmds["variable"].isin(monthly_variables)]

    return pd.concat([df_dmds, df_m])


def _extract_variable_dataframes(df_dmds: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Extract and format individual variable dataframes from the main dataset."""
    variables = {
        "urban_withdrawal": "urban_withdrawal2_baseline",
        "rural_withdrawal": "rural_withdrawal_baseline",
        "industrial_withdrawal": "manufacturing_withdrawal_baseline",
        "industrial_return": "manufacturing_return_baseline",
        "urban_return": "urban_return2_baseline",
        "rural_return": "rural_return_baseline",
        "urban_connection_rate": "urban_connection_rate_baseline",
        "rural_connection_rate": "rural_connection_rate_baseline",
        "urban_treatment_rate": "urban_treatment_rate_baseline",
        "rural_treatment_rate": "rural_treatment_rate_baseline",
        "recycling": "urban_recycling_rate_baseline",
    }

    result = {}
    for key, variable in variables.items():
        df = df_dmds[df_dmds["variable"] == variable].reset_index(drop=True)
        result[key] = df

    return result


def _apply_policy_scenarios(
    variables: dict[str, pd.DataFrame], df_dmds: pd.DataFrame, context: "Context"
) -> dict[str, pd.DataFrame]:
    """Apply SDG targets or other policy scenarios to rate variables."""
    if context.SDG == "baseline":
        return variables

    if context.SDG == "SDG":
        # Apply SDG targets using basin classification
        FILE2 = f"basins_country_{context.regions}.csv"
        PATH = package_data_path("water", "delineation", FILE2)
        df_basin = pd.read_csv(PATH)

        # Apply SDG targets
        variables["rural_treatment_rate"] = set_target_rate(
            variables["rural_treatment_rate"],
            "connection",
            SDG_RURAL_SANITATION_TARGET,
            basin=df_basin,
        )
        variables["urban_treatment_rate"] = set_target_rate(
            variables["urban_treatment_rate"],
            "connection",
            SDG_URBAN_SANITATION_TARGET,
            basin=df_basin,
        )
        variables["urban_connection_rate"] = set_target_rate(
            variables["urban_connection_rate"],
            "connection",
            SDG_URBAN_CONNECTION_TARGET,
            basin=df_basin,
        )
        variables["rural_connection_rate"] = set_target_rate(
            variables["rural_connection_rate"],
            "connection",
            SDG_RURAL_CONNECTION_TARGET,
            basin=df_basin,
        )
        variables["recycling"] = set_target_rate(
            variables["recycling"], "treatment", 0.0, basin=df_basin
        )

        # Save all rates for reporting
        _save_policy_rates(variables, context, "SDG")

    else:
        # Handle other policy scenarios
        variables = _apply_custom_policy(variables, df_dmds, context)

    return variables


def _apply_custom_policy(
    variables: dict[str, pd.DataFrame], df_dmds: pd.DataFrame, context: "Context"
) -> dict[str, pd.DataFrame]:
    """Apply custom policy scenario data."""
    pol_scen = context.SDG

    # Check if policy data exists
    check_dm = df_dmds[df_dmds["variable"] == f"urban_connection_rate_{pol_scen}"]
    if check_dm.empty:
        raise ValueError(f"Policy data is missing for the {pol_scen} scenario.")

    # Extract policy-specific variables
    policy_variables = [
        "urban_connection_rate",
        "rural_connection_rate",
        "urban_treatment_rate",
        "rural_treatment_rate",
        "recycling",
    ]

    for var in policy_variables:
        policy_var_name = f"{var}_{pol_scen}".replace(
            "recycling", "urban_recycling_rate"
        )
        variables[var] = df_dmds[df_dmds["variable"] == policy_var_name].reset_index(
            drop=True
        )

    # Save policy rates for reporting
    _save_policy_rates(variables, context, pol_scen)

    return variables


def _save_policy_rates(
    variables: dict[str, pd.DataFrame], context: "Context", policy_name: str
) -> None:
    """Save policy rates for reporting purposes."""
    all_rates_base = pd.concat(
        [
            variables["urban_connection_rate"],
            variables["rural_connection_rate"],
            variables["urban_treatment_rate"],
            variables["rural_treatment_rate"],
            variables["recycling"],
        ]
    )

    save_path = package_data_path("water", "demands", "harmonized", context.regions)
    all_rates_base.to_csv(save_path / "all_rates_SSP2.csv", index=False)


def _process_sectoral_demands(variables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Process urban, rural, and industrial water demands into MESSAGE format."""
    dmd_df = pd.DataFrame()

    # Urban connected water demand
    urban_mw = variables["urban_withdrawal"].reset_index(drop=True)
    urban_mw = urban_mw.merge(
        variables["urban_connection_rate"]
        .drop(columns=["variable", "time"])
        .rename(columns={"value": "rate"})
    )
    urban_mw["value"] = urban_mw["value"] * urban_mw["rate"]

    dmd_df = pd.concat(
        [
            dmd_df,
            make_df(
                "demand",
                node="B" + urban_mw["node"],
                commodity="urban_mw",
                level="final",
                year=urban_mw["year"],
                time=urban_mw["time"],
                value=urban_mw["value"],
                unit="MCM/year",
            ),
        ]
    )

    # Urban disconnected water demand
    urban_dis = variables["urban_withdrawal"].reset_index(drop=True)
    urban_dis = urban_dis.merge(
        variables["urban_connection_rate"]
        .drop(columns=["variable", "time"])
        .rename(columns={"value": "rate"})
    )
    urban_dis["value"] = urban_dis["value"] * (1 - urban_dis["rate"])

    dmd_df = pd.concat(
        [
            dmd_df,
            make_df(
                "demand",
                node="B" + urban_dis["node"],
                commodity="urban_disconnected",
                level="final",
                year=urban_dis["year"],
                time=urban_dis["time"],
                value=urban_dis["value"],
                unit="MCM/year",
            ),
        ]
    )

    # Rural connected water demand
    rural_mw = variables["rural_withdrawal"].reset_index(drop=True)
    rural_mw = rural_mw.merge(
        variables["rural_connection_rate"]
        .drop(columns=["variable", "time"])
        .rename(columns={"value": "rate"})
    )
    rural_mw["value"] = rural_mw["value"] * rural_mw["rate"]

    dmd_df = pd.concat(
        [
            dmd_df,
            make_df(
                "demand",
                node="B" + rural_mw["node"],
                commodity="rural_mw",
                level="final",
                year=rural_mw["year"],
                time=rural_mw["time"],
                value=rural_mw["value"],
                unit="MCM/year",
            ),
        ]
    )

    # Rural disconnected water demand
    rural_dis = variables["rural_withdrawal"].reset_index(drop=True)
    rural_dis = rural_dis.merge(
        variables["rural_connection_rate"]
        .drop(columns=["variable", "time"])
        .rename(columns={"value": "rate"})
    )
    rural_dis["value"] = rural_dis["value"] * (1 - rural_dis["rate"])

    dmd_df = pd.concat(
        [
            dmd_df,
            make_df(
                "demand",
                node="B" + rural_dis["node"],
                commodity="rural_disconnected",
                level="final",
                year=rural_dis["year"],
                time=rural_dis["time"],
                value=rural_dis["value"],
                unit="MCM/year",
            ),
        ]
    )

    # Industrial water demand
    manuf_mw = variables["industrial_withdrawal"].reset_index(drop=True)

    dmd_df = pd.concat(
        [
            dmd_df,
            make_df(
                "demand",
                node="B" + manuf_mw["node"],
                commodity="industry_mw",
                level="final",
                year=manuf_mw["year"],
                time=manuf_mw["time"],
                value=manuf_mw["value"],
                unit="MCM/year",
            ),
        ]
    )

    # Industrial return flows (negative demand)
    manuf_return = variables["industrial_return"].reset_index(drop=True)

    dmd_df = pd.concat(
        [
            dmd_df,
            make_df(
                "demand",
                node="B" + manuf_return["node"],
                commodity="industry_uncollected_wst",
                level="final",
                year=manuf_return["year"],
                time=manuf_return["time"],
                value=-manuf_return["value"],
                unit="MCM/year",
            ),
        ]
    )

    return dmd_df


def _process_wastewater_streams(variables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Process collected and uncollected wastewater streams."""
    wst_df = pd.DataFrame()

    # Urban collected wastewater
    urban_collected_wst = variables["urban_return"].reset_index(drop=True)
    urban_collected_wst = urban_collected_wst.merge(
        variables["urban_treatment_rate"]
        .drop(columns=["variable", "time"])
        .rename(columns={"value": "rate"})
    )
    urban_collected_wst["value"] = (
        urban_collected_wst["value"] * urban_collected_wst["rate"]
    )

    wst_df = pd.concat(
        [
            wst_df,
            make_df(
                "demand",
                node="B" + urban_collected_wst["node"],
                commodity="urban_collected_wst",
                level="final",
                year=urban_collected_wst["year"],
                time=urban_collected_wst["time"],
                value=-urban_collected_wst["value"],
                unit="MCM/year",
            ),
        ]
    )

    # Rural collected wastewater
    rural_collected_wst = variables["rural_return"].reset_index(drop=True)
    rural_collected_wst = rural_collected_wst.merge(
        variables["rural_treatment_rate"]
        .drop(columns=["variable", "time"])
        .rename(columns={"value": "rate"})
    )
    rural_collected_wst["value"] = (
        rural_collected_wst["value"] * rural_collected_wst["rate"]
    )

    wst_df = pd.concat(
        [
            wst_df,
            make_df(
                "demand",
                node="B" + rural_collected_wst["node"],
                commodity="rural_collected_wst",
                level="final",
                year=rural_collected_wst["year"],
                time=rural_collected_wst["time"],
                value=-rural_collected_wst["value"],
                unit="MCM/year",
            ),
        ]
    )

    # Urban uncollected wastewater
    urban_uncollected_wst = variables["urban_return"].reset_index(drop=True)
    urban_uncollected_wst = urban_uncollected_wst.merge(
        variables["urban_treatment_rate"]
        .drop(columns=["variable", "time"])
        .rename(columns={"value": "rate"})
    )
    urban_uncollected_wst["value"] = urban_uncollected_wst["value"] * (
        1 - urban_uncollected_wst["rate"]
    )

    wst_df = pd.concat(
        [
            wst_df,
            make_df(
                "demand",
                node="B" + urban_uncollected_wst["node"],
                commodity="urban_uncollected_wst",
                level="final",
                year=urban_uncollected_wst["year"],
                time=urban_uncollected_wst["time"],
                value=-urban_uncollected_wst["value"],
                unit="MCM/year",
            ),
        ]
    )

    # Rural uncollected wastewater
    rural_uncollected_wst = variables["rural_return"].reset_index(drop=True)
    rural_uncollected_wst = rural_uncollected_wst.merge(
        variables["rural_treatment_rate"]
        .drop(columns=["variable", "time"])
        .rename(columns={"value": "rate"})
    )
    rural_uncollected_wst["value"] = rural_uncollected_wst["value"] * (
        1 - rural_uncollected_wst["rate"]
    )

    wst_df = pd.concat(
        [
            wst_df,
            make_df(
                "demand",
                node="B" + rural_uncollected_wst["node"],
                commodity="rural_uncollected_wst",
                level="final",
                year=rural_uncollected_wst["year"],
                time=rural_uncollected_wst["time"],
                value=-rural_uncollected_wst["value"],
                unit="MCM/year",
            ),
        ]
    )

    return wst_df


def _calculate_historical_data(
    dmd_df: pd.DataFrame, year_vtgs: tuple
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate historical activity and capacity data."""
    # Add historical activities to corresponding technologies
    h_act = dmd_df[dmd_df["year"].isin(year_vtgs)].copy()

    # Technology mapping for historical activities
    conditions = [
        (h_act["commodity"] == "urban_mw"),
        (h_act["commodity"] == "industry_mw"),
        (h_act["commodity"] == "rural_mw"),
        (h_act["commodity"] == "urban_disconnected"),
        (h_act["commodity"] == "rural_disconnected"),
        (h_act["commodity"] == "urban_collected_wst"),
        (h_act["commodity"] == "rural_collected_wst"),
        (h_act["commodity"] == "urban_uncollected_wst"),
        (h_act["commodity"] == "industry_uncollected_wst"),
        (h_act["commodity"] == "rural_uncollected_wst"),
    ]

    values = [
        "urban_t_d",
        "industry_unconnected",
        "rural_t_d",
        "urban_unconnected",
        "rural_unconnected",
        "urban_sewerage",
        "rural_sewerage",
        "urban_untreated",
        "industry_untreated",
        "rural_untreated",
    ]

    h_act["commodity"] = np.select(conditions, values, "Unknown commodity")
    h_act["value"] = h_act["value"].abs()

    hist_act = make_df(
        "historical_activity",
        node_loc=h_act["node"],
        technology=h_act["commodity"],
        year_act=h_act["year"],
        mode="M1",
        time=h_act["time"],
        value=h_act["value"] * KM3_2_MCM,
        unit="MCM/year",
    )

    # Historical capacity
    h_cap = h_act[h_act["year"] > year_vtgs[0]]
    h_cap = (
        h_cap.groupby(["node", "commodity", "level", "year", "unit"])["value"]
        .sum()
        .reset_index()
    )

    hist_cap = make_df(
        "historical_new_capacity",
        node_loc=h_cap["node"],
        technology=h_cap["commodity"],
        year_vtg=h_cap["year"],
        value=h_cap["value"] / 5 * KM3_2_MCM,
        unit="MCM/year",
    )

    return hist_act, hist_cap


def _create_recycling_constraints(
    variables: dict[str, pd.DataFrame], sub_time: str, info
) -> pd.DataFrame:
    """Create water recycling share constraints."""
    df_share_wat = make_df(
        "share_commodity_lo",
        shares="share_wat_recycle",
        node_share="B" + variables["recycling"]["node"],
        year_act=variables["recycling"]["year"],
        value=variables["recycling"]["value"],
        unit="-",
    ).pipe(
        broadcast,
        time=pd.Series(sub_time),
    )

    df_share_wat = df_share_wat[df_share_wat["year_act"].isin(info.Y)]
    return df_share_wat


@minimum_version("message_ix 3.7")
def add_sectoral_demands(context: "Context", scenario=None) -> dict[str, pd.DataFrame]:
    """
    Adds water sectoral demands using a refactored, modular approach.

    Parameters
    ----------
    context : .Context
    scenario : .Scenario, optional
        Scenario to use. If not provided, uses context.get_scenario().

    Returns
    -------
    data : dict of (str -> pandas.DataFrame)
        Keys are MESSAGE parameter names such as 'input', 'fix_cost'. Values
        are data frames ready for :meth:`~.Scenario.add_par`.
    """
    results = {}

    # Reference to the water configuration
    info = context["water build info"]
    year_vtgs = tuple(range(2010, info.Y[0], 5))
    sub_time = context.time

    # Step 1: Load and process demand data
    df_dmds = _load_and_process_demand_data(context)

    # Step 2: Extract individual variable dataframes
    variables = _extract_variable_dataframes(df_dmds)

    # Step 3: Apply policy scenarios (SDG targets or custom policies)
    variables = _apply_policy_scenarios(variables, df_dmds, context)

    # Step 4: Process sectoral demands (urban, rural, industrial)
    dmd_df = _process_sectoral_demands(variables)

    # Step 5: Process wastewater streams (collected/uncollected)
    wst_df = _process_wastewater_streams(variables)

    # Combine all demand dataframes
    all_dmd_df = pd.concat([dmd_df, wst_df])

    # Step 6: Calculate historical activity and capacity data
    # (before filtering to model years)
    hist_act, hist_cap = _calculate_historical_data(all_dmd_df, year_vtgs)

    # Filter to model years only
    all_dmd_df = all_dmd_df[all_dmd_df["year"].isin(info.Y)]
    results["demand"] = all_dmd_df
    results["historical_activity"] = hist_act
    results["historical_new_capacity"] = hist_cap

    # Step 7: Create recycling constraints
    df_share_wat = _create_recycling_constraints(variables, sub_time, info)
    results["share_commodity_lo"] = df_share_wat

    return results


def read_water_availability(context: "Context") -> Sequence[pd.DataFrame]:
    """
    Reads water availability data and bias correct
    it for the historical years and no climate
    scenario assumptions.

    Parameters
    ----------
    context : .Context

    Returns
    -------
    data : (pd.DataFrame, pd.DataFrame)
    """

    # Reference to the water configuration
    info = context["water build info"]
    # reading sample for assiging basins
    PATH = package_data_path(
        "water", "delineation", f"basins_by_region_simpl_{context.regions}.csv"
    )
    df_x = pd.read_csv(PATH)

    if "year" in context.time:
        # path for reading basin delineation file
        PATH = package_data_path(
            "water", "delineation", f"basins_by_region_simpl_{context.regions}.csv"
        )
        df_x = pd.read_csv(PATH)
        # Adding freshwater supply constraints
        # Reading data, the data is spatially and temprally aggregated from GHMs
        path_qtot = package_data_path(
            "water",
            "availability",
            f"qtot_5y_{context.RCP}_{context.REL}_{context.regions}.csv",
        )
        # Read rcp 2.6 data
        df_sw = pd.read_csv(path_qtot)
        df_sw.drop(["Unnamed: 0"], axis=1, inplace=True)

        df_sw.index = df_x["BCU_name"].index
        df_sw = df_sw.stack().reset_index()
        df_sw.columns = pd.Index(["Region", "years", "value"])
        df_sw.fillna(0, inplace=True)
        df_sw.reset_index(drop=True, inplace=True)
        df_sw["year"] = pd.DatetimeIndex(df_sw["years"]).year
        df_sw["time"] = "year"
        df_sw["Region"] = df_sw["Region"].map(df_x["BCU_name"])
        df_sw2210 = df_sw[df_sw["year"] == 2100].copy()
        df_sw2210["year"] = 2110
        df_sw = pd.concat([df_sw, df_sw2210])
        df_sw = df_sw[df_sw["year"].isin(info.Y)]

        # Adding groundwater supply constraints
        # Reading data, the data is spatially and temprally aggregated from GHMs
        path_qr = package_data_path(
            "water",
            "availability",
            f"qr_5y_{context.RCP}_{context.REL}_{context.regions}.csv",
        )

        # Read groundwater data
        df_gw = pd.read_csv(path_qr)
        df_gw.drop(["Unnamed: 0"], axis=1, inplace=True)
        df_gw.index = df_x["BCU_name"].index
        df_gw = df_gw.stack().reset_index()
        df_gw.columns = pd.Index(["Region", "years", "value"])
        df_gw.fillna(0, inplace=True)
        df_gw.reset_index(drop=True, inplace=True)
        df_gw["year"] = pd.DatetimeIndex(df_gw["years"]).year
        df_gw["time"] = "year"
        df_gw["Region"] = df_gw["Region"].map(df_x["BCU_name"])
        df_gw2210 = df_gw[df_gw["year"] == 2100].copy()
        df_gw2210["year"] = 2110
        df_gw = pd.concat([df_gw, df_gw2210])
        df_gw = df_gw[df_gw["year"].isin(info.Y)]

    else:
        # Adding freshwater supply constraints
        # Reading data, the data is spatially and temprally aggregated from GHMs
        path_qtot_month = package_data_path(
            "water",
            "availability",
            f"qtot_5y_m_{context.RCP}_{context.REL}_{context.regions}.csv",
        )
        df_sw = pd.read_csv(path_qtot_month)
        df_sw.drop(["Unnamed: 0"], axis=1, inplace=True)

        df_sw.index = df_x["BCU_name"].index
        df_sw = df_sw.stack().reset_index()
        df_sw.columns = pd.Index(["Region", "years", "value"])
        df_sw.sort_values(["Region", "years", "value"], inplace=True)
        df_sw.fillna(0, inplace=True)
        df_sw.reset_index(drop=True, inplace=True)
        df_sw["year"] = pd.DatetimeIndex(df_sw["years"]).year
        df_sw["time"] = pd.DatetimeIndex(df_sw["years"]).month
        df_sw["Region"] = df_sw["Region"].map(df_x["BCU_name"])
        df_sw2210 = df_sw[df_sw["year"] == 2100].copy()
        df_sw2210["year"] = 2110
        df_sw = pd.concat([df_sw, df_sw2210])
        df_sw = df_sw[df_sw["year"].isin(info.Y)]

        # Reading data, the data is spatially and temporally aggregated from GHMs
        path_qr_month = package_data_path(
            "water",
            "availability",
            f"qr_5y_m_{context.RCP}_{context.REL}_{context.regions}.csv",
        )
        df_gw = pd.read_csv(path_qr_month)
        df_gw.drop(["Unnamed: 0"], axis=1, inplace=True)

        df_gw.index = df_x["BCU_name"].index
        df_gw = df_gw.stack().reset_index()
        df_gw.columns = pd.Index(["Region", "years", "value"])
        df_gw.sort_values(["Region", "years", "value"], inplace=True)
        df_gw.fillna(0, inplace=True)
        df_gw.reset_index(drop=True, inplace=True)
        df_gw["year"] = pd.DatetimeIndex(df_gw["years"]).year
        df_gw["time"] = pd.DatetimeIndex(df_gw["years"]).month
        df_gw["Region"] = df_gw["Region"].map(df_x["BCU_name"])
        df_gw2210 = df_gw[df_gw["year"] == 2100].copy()
        df_gw2210["year"] = 2110
        df_gw = pd.concat([df_gw, df_gw2210])
        df_gw = df_gw[df_gw["year"].isin(info.Y)]

    return df_sw, df_gw


def add_water_availability(context: "Context", scenario=None) -> dict[str, pd.DataFrame]:
    """
    Adds water supply constraints

    Parameters
    ----------
    context : .Context
    scenario : .Scenario, optional
        Scenario to use. If not provided, uses context.get_scenario().

    Returns
    -------
    data : dict of (str -> pandas.DataFrame)
        Keys are MESSAGE parameter names such as 'input', 'fix_cost'. Values
        are data frames ready for :meth:`~.Scenario.add_par`.
    """

    # define an empty dictionary
    results = {}
    # Adding freshwater supply constraints
    # Reading data, the data is spatially and temprally aggregated from GHMs

    df_sw, df_gw = read_water_availability(context)

    dmd_df = make_df(
        "demand",
        node="B" + df_sw["Region"].astype(str),
        commodity="surfacewater_basin",
        level="water_avail_basin",
        year=df_sw["year"],
        time=df_sw["time"],
        value=df_sw["value"] * KM3_2_MCM * -1,
        unit="MCM/year",
    )

    dmd_df = pd.concat(
        [
            dmd_df,
            make_df(
                "demand",
                node="B" + df_gw["Region"].astype(str),
                commodity="groundwater_basin",
                level="water_avail_basin",
                year=df_gw["year"],
                time=df_gw["time"],
                value=df_gw["value"] * KM3_2_MCM * -1,
                unit="MCM/year",
            ),
        ]
    )

    dmd_df["value"] = dmd_df["value"].apply(lambda x: x if x <= 0 else 0)

    results["demand"] = dmd_df

    # share constraint lower bound on groundwater
    df_share = make_df(
        "share_commodity_lo",
        shares="share_low_lim_GWat",
        node_share="B" + df_gw["Region"].astype(str),
        year_act=df_gw["year"],
        time=df_gw["time"],
        value=df_gw["value"]
        / (df_sw["value"] + df_gw["value"])
        * 0.95,  # 0.95 buffer factor to avoid numerical error
        unit="-",
    )

    df_share["value"] = df_share["value"].fillna(0)

    results["share_commodity_lo"] = df_share

    return results


def add_irrigation_demand(context: "Context", scenario=None) -> dict[str, pd.DataFrame]:
    """
    Adds endogenous irrigation water demands from GLOBIOM emulator

    Parameters
    ----------
    context : .Context
    scenario : .Scenario, optional
        Scenario to use. If not provided, uses context.get_scenario().

    Returns
    -------
    data : dict of (str -> pandas.DataFrame)
        Keys are MESSAGE parameter names such as 'input', 'fix_cost'. Values
        are data frames ready for :meth:`~.Scenario.add_par`.
    """
    # define an empty dictionary
    results = {}

    scen = scenario if scenario is not None else context.get_scenario()
    # add water for irrigation from globiom
    land_out_1 = scen.par(
        "land_output", {"commodity": "Water|Withdrawal|Irrigation|Cereals"}
    )
    land_out_1["level"] = "irr_cereal"
    land_out_2 = scen.par(
        "land_output", {"commodity": "Water|Withdrawal|Irrigation|Oilcrops"}
    )
    land_out_2["level"] = "irr_oilcrops"
    land_out_3 = scen.par(
        "land_output", {"commodity": "Water|Withdrawal|Irrigation|Sugarcrops"}
    )
    land_out_3["level"] = "irr_sugarcrops"

    land_out = pd.concat([land_out_1, land_out_2, land_out_3])
    land_out["commodity"] = "freshwater"

    land_out["value"] = land_out["value"]

    # take land_out edited and add as a demand in  land_input
    results["land_input"] = land_out

    return results
