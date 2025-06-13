import logging
from collections import defaultdict
from functools import lru_cache
from itertools import product
from typing import Optional

import pandas as pd
import xarray as xr
from sdmx.model.v21 import Code

from message_ix_models import Context
from message_ix_models.model.structure import get_codes
from message_ix_models.util import load_package_data

log = logging.getLogger(__name__)

# Configuration files
METADATA = [
    # Information about MESSAGE-water
    ("water", "config"),
    ("water", "set"),
    ("water", "technology"),
]


def read_config(context: Optional[Context] = None):
    """Read the water model configuration / metadata from file.

    Numerical values are converted to computation-ready data structures.

    Returns
    -------
    .Context
        The current Context, with the loaded configuration.
    """

    context = context or Context.get_instance(-1)

    # if context.nexus_set == 'nexus':
    if "water set" in context:
        # Already loaded
        return context

    # Load water configuration
    for parts in METADATA:
        # Key for storing in the context
        key = " ".join(parts)

        # Actual filename parts; ends with YAML
        _parts = list(parts)
        _parts[-1] += ".yaml"

        context[key] = load_package_data(*_parts)

    return context


@lru_cache()
def map_add_on(rtype=Code):
    """Map addon & type_addon in ``sets.yaml``."""
    dims = ["add_on", "type_addon"]

    # Retrieve configuration
    context = read_config()

    # Assemble group information
    result = defaultdict(list)

    for indices in product(*[context["water set"][d]["add"] for d in dims]):
        # Create a new code by combining two
        result["code"].append(
            Code(
                id="".join(str(c.id) for c in indices),
                name=", ".join(str(c.name) for c in indices),
            )
        )

        # Tuple of the values along each dimension
        result["index"].append(tuple(c.id for c in indices))

    if rtype == "indexers":
        # Three tuples of members along each dimension
        indexers = zip(*result["index"])
        indexers = {
            d: xr.DataArray(list(i), dims="consumer_group")
            for d, i in zip(dims, indexers)
        }
        indexers["consumer_group"] = xr.DataArray(
            [c.id for c in result["code"]],
            dims="consumer_group",
        )
        return indexers
    elif rtype is Code:
        return sorted(result["code"], key=str)
    else:
        raise ValueError(rtype)


def add_commodity_and_level(df: pd.DataFrame, default_level=None):
    # Add input commodity and level
    t_info: list = Context.get_instance()["water set"]["technology"]["add"]
    c_info: list = get_codes("commodity")

    @lru_cache()
    def t_cl(t):
        input = t_info[t_info.index(t)].annotations["input"]
        # Commodity must be specified
        commodity = input["commodity"]
        # Use the default level for the commodity in the RES (per
        # commodity.yaml)
        level = (
            input.get("level", "water_supply")
            or c_info[c_info.index(commodity)].annotations.get("level", None)
            or default_level
        )

        return commodity, level

    def func(row: pd.Series):
        row[["commodity", "level"]] = t_cl(row["technology"])
        return row

    return df.apply(func, axis=1)


def get_vintage_and_active_years(
    info, technology: str, technical_lifetime: int = None
) -> pd.DataFrame:
    """Calculate valid vintage-activity year combinations without scenario dependency.

    This implements the same logic as scenario.vintage_and_active_years() but
    uses the technical lifetime data directly instead of requiring it to be in
    the scenario first.

    Parameters
    ----------
    info : ScenarioInfo
        Contains the base yv_ya combinations
    technology : str
        Technology name (for cache key)
    technical_lifetime : int, optional
        Technical lifetime in years. If None, returns all combinations.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['year_vtg', 'year_act'] containing valid combinations
    """
    # Get base yv_ya from ScenarioInfo property
    yv_ya = info.yv_ya

    # If no technical lifetime specified or is nan, return all combinations
    if technical_lifetime is None or pd.isna(technical_lifetime):
        technical_lifetime = 1

    # Filter to only valid combinations based on technical lifetime
    # A technology can only be active for technical_lifetime years after vintage
    valid_mask = yv_ya["year_act"] <= yv_ya["year_vtg"] + technical_lifetime

    return yv_ya[valid_mask].reset_index(drop=True)


# Legacy function for backwards compatibility - replace calls with get_vintage_and_active_years()
def map_yv_ya_lt(scenario, node: str, technology: str) -> pd.DataFrame:
    """Legacy wrapper - use get_vintage_and_active_years() instead."""
    # This is the old problematic function - should be replaced
    try:
        yv_ya = scenario.vintage_and_active_years((node, technology))
        if not yv_ya.empty:
            return yv_ya
    except (ValueError, KeyError):
        pass

    # Fallback
    model_years = [y for y in scenario.set("year") if y >= scenario.firstmodelyear]
    if model_years:
        data = []
        for year in model_years:
            data.append({"year_vtg": year, "year_act": year})
        return pd.DataFrame(data)
    else:
        return pd.DataFrame(columns=["year_vtg", "year_act"])
