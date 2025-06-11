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


def map_yv_ya_lt(scenario, node: str, technology: str) -> pd.DataFrame:
    """Wrapper for scenario.vintage_and_active_years() with fallback.

    Parameters
    ----------
    scenario : message_ix.Scenario
        The MESSAGEix scenario object.
    node : str
        Node name.
    technology : str
        Technology name for which to generate vintage-activity combinations.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns 'year_vtg' and 'year_act' from MESSAGEix.
    """
    try:
        yv_ya = scenario.vintage_and_active_years((node, technology))

        # The MESSAGEix method returns a DataFrame directly
        if not yv_ya.empty:
            return yv_ya
    except (ValueError, KeyError):
        # No technical lifetime data available for this technology/node combination
        pass

    # Fallback: create minimal valid combinations using only model years
    model_years = [y for y in scenario.set("year") if y >= scenario.firstmodelyear]

    if model_years:
        # Simple approach: each year is both vintage and active
        data = []
        for year in model_years:
            data.append({"year_vtg": year, "year_act": year})
        return pd.DataFrame(data)
    else:
        # Return empty DataFrame with correct columns if no valid years
        return pd.DataFrame(columns=["year_vtg", "year_act"])
