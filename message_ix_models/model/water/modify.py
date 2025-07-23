"""Efficient modification of existing water scenarios for RCP parameter changes."""

import logging
from typing import TYPE_CHECKING

from message_ix_models.util import add_par_data

if TYPE_CHECKING:
    from message_ix_models import Context

log = logging.getLogger(__name__)


def modify_rcp(scenario, context: "Context", new_rcp: str):
    """Efficiently modify scenario for new RCP parameter.

    This function updates an existing water scenario to use a different RCP
    (Representative Concentration Pathway) climate scenario by:
    1. Removing parameters that depend on the old RCP value
    2. Regenerating only the RCP-dependent data functions
    3. Adding the new parameters to the scenario

    This is much more efficient than rebuilding the entire scenario.

    Parameters
    ----------
    scenario : message_ix.Scenario
        The scenario to modify
    context : Context
        The context object containing configuration
    new_rcp : str
        The new RCP value (e.g., "2p6", "6p0", "7p0", "no_climate")

    Raises
    ------
    ValueError
        If new_rcp is not a valid RCP value
    """
    valid_rcps = ["no_climate", "6p0", "2p6", "7p0"]
    if new_rcp not in valid_rcps:
        raise ValueError(f"Invalid RCP value: {new_rcp}. Must be one of {valid_rcps}")

    old_rcp = context.RCP

    if old_rcp == new_rcp:
        log.info(f"RCP already set to {new_rcp}, no changes needed")
        return

    log.info(f"Modifying scenario RCP from {old_rcp} to {new_rcp}")

    # Update context with new RCP
    context.RCP = new_rcp

    # Remove old RCP-dependent parameters
    _remove_rcp_parameters(scenario, context)

    # Regenerate only RCP-dependent data
    _add_rcp_data(scenario, context)

    log.info(f"Successfully modified scenario RCP to {new_rcp}")


def _remove_rcp_parameters(scenario, context: "Context"):
    """Remove parameters that depend on the RCP value.

    This function identifies and removes all parameters from the scenario
    that were generated based on the RCP value, preparing for the
    new RCP data to be added.
    """
    log.info("Removing RCP-dependent parameters")

    # Get the mapping of data functions to parameters they generate
    function_params = _get_rcp_function_parameters()

    with scenario.transact("Remove old RCP parameters"):
        for func_name, param_names in function_params.items():
            log.debug(f"Removing parameters from {func_name}")
            for param_name in param_names:
                try:
                    # Check if parameter exists and has data
                    existing_data = scenario.par(param_name)
                    if not existing_data.empty:
                        # For water-specific parameters, we can be more selective
                        # Remove only water-related technologies/nodes
                        filtered_data = _filter_water_parameters(
                            existing_data, param_name, context
                        )
                        if not filtered_data.empty:
                            scenario.remove_par(param_name, filtered_data)
                            log.debug(
                                f"Removed {len(filtered_data)} rows from {param_name}"
                            )
                except Exception as e:
                    # Parameter might not exist, which is fine
                    log.debug(f"Could not remove {param_name}: {e}")
                    continue


def _add_rcp_data(scenario, context: "Context"):
    """Add new data for RCP-dependent functions.

    This function calls only the data functions that depend on RCP
    and adds their generated parameters to the scenario.
    """
    from .data.demands import add_water_availability
    from .data.infrastructure import add_desalination
    from .data.water_for_ppl import cool_tech
    from .data.water_supply import add_e_flow, add_water_supply

    log.info("Adding new RCP-dependent data")

    # List of data functions that depend on RCP parameter
    rcp_data_functions = [
        add_water_supply,  # Uses qtot_5y_{RCP}_{REL}_{regions}.csv
        add_water_availability,  # Uses qtot_5y_{RCP}_{REL}_{regions}.csv and qr_5y_{RCP}_{REL}_{regions}.csv
        cool_tech,  # Uses power_plant_cooling_impact_MESSAGE_{regions}_{RCP}.csv
        add_e_flow,  # Uses e-flow_{RCP}_{regions}.csv
        add_desalination,  # Filters data by RCP column
    ]

    with scenario.transact("Add new RCP parameters"):
        for func in rcp_data_functions:
            try:
                log.debug(f"Generating data from {func.__name__}()")
                data = func(context)
                add_par_data(scenario, data, dry_run=False)
                log.debug(f"Added data from {func.__name__}()")
            except Exception as e:
                log.error(f"Error in {func.__name__}(): {e}")
                raise


def _get_rcp_function_parameters():
    """Get mapping of RCP-dependent data functions to their generated parameters.

    Returns a dictionary mapping function names to the list of parameters
    they generate that depend on RCP values.

    Returns
    -------
    dict
        Mapping of function names to parameter lists
    """
    return {
        "add_water_supply": [
            "share_commodity_lo",
            "share_commodity_up",
        ],
        "add_water_availability": [
            "bound_activity_up",  # Water availability bounds
        ],
        "cool_tech": [
            "input",  # Cooling technology water inputs
            "capacity_factor",  # Climate impacts on cooling
        ],
        "add_e_flow": [
            "bound_activity_lo",  # Environmental flow constraints
        ],
        "add_desalination": [
            "inv_cost",  # Investment costs (RCP filtered)
            "fix_cost",  # Fixed costs (RCP filtered)
            "var_cost",  # Variable costs (RCP filtered)
        ],
    }


def _filter_water_parameters(data, param_name, context):
    """Filter parameter data to only include water-related entries.

    This function identifies which rows in a parameter DataFrame
    are related to water technologies/nodes and should be removed
    when changing RCP.

    Parameters
    ----------
    data : pandas.DataFrame
        The parameter data from the scenario
    param_name : str
        Name of the parameter
    context : Context
        Context object with water configuration

    Returns
    -------
    pandas.DataFrame
        Filtered data containing only water-related entries
    """
    # Start with all data - we'll filter based on water-specific patterns
    filtered = data.copy()

    # Filter based on water-specific technology patterns
    if "technology" in data.columns:
        # Look for cooling technologies (contain "__")
        cooling_mask = data["technology"].str.contains("__", na=False)
        # Look for water supply technologies
        water_tech_patterns = ["extract_", "treat_", "supply_", "desal_", "conv_"]
        water_mask = data["technology"].str.contains(
            "|".join(water_tech_patterns), na=False
        )
        tech_mask = cooling_mask | water_mask
        filtered = filtered[tech_mask]

    # Filter based on water-specific node patterns (basins start with "B")
    if "node" in data.columns:
        basin_mask = data["node"].str.startswith("B", na=False)
        if basin_mask.any():
            filtered = filtered[basin_mask]
    elif "node_loc" in data.columns:
        basin_mask = data["node_loc"].str.startswith("B", na=False)
        if basin_mask.any():
            filtered = filtered[basin_mask]

    # Filter based on water-specific commodities
    if "commodity" in data.columns:
        water_commodities = [
            "freshwater",
            "surfacewater",
            "groundwater",
            "urban_collected_wst",
            "rural_collected_wst",
            "surfacewater_basin",
            "groundwater_basin",
        ]
        commodity_mask = data["commodity"].isin(water_commodities)
        if commodity_mask.any():
            filtered = filtered[commodity_mask]

    return filtered

