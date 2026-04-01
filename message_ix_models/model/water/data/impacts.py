"""Water-domain RIME impact transformations.

Transforms raw RIME predictions (native 157-basin emulator resolution)
to MESSAGE-compatible arrays (217 basin-region rows). Basin geometry
utilities live in :mod:`message_ix_models.model.water.utils`; this module
owns the RIME-specific index mapping and the domain-level prediction entry
point.

Pure data operations only — no scenario objects. Callers handle I/O.
"""

import functools
import logging

import numpy as np
import pandas as pd
import xarray as xr

from message_ix_models.model.water.utils import (
    NAN_BASIN_IDS,
    load_basin_mapping,
    split_basin_macroregion,
)
from message_ix_models.tools.impacts import (
    clip_gmt,
    impacts_data_path,
    predict_rime,
    sample_to_model_years,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RIME-specific basin index mapping
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _get_rime_region_mapping() -> dict[int, int]:
    """Mapping from BASIN_ID to RIME array index.

    RIME datasets have 157 basins indexed by region IDs [1..162] with gaps.
    Uses a reference dataset (qtot_mean annual window11) to discover the
    mapping; region IDs are identical across all basin-level RIME datasets.
    """
    dataset_path = impacts_data_path(
        "rime", "rime_regionarray_qtot_mean_CWatM_annual_window11.nc"
    )
    ds = xr.open_dataset(dataset_path)
    rime_region_ids = ds.region.values
    return {int(region_id): i for i, region_id in enumerate(rime_region_ids)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_water_rime(
    gmt_array,
    variable: str,
    temporal_res: str = "annual",
    hydro_model: str = "CWatM",
    percentile: str | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Predict water variable at MESSAGE basin resolution (217 rows).

    Wraps :func:`~message_ix_models.tools.impacts.predict_rime` with
    basin expansion from 157 RIME basins to 217 MESSAGE basin-region rows.

    Parameters
    ----------
    gmt_array
        GMT values (degC above pre-industrial).
    variable
        Basin-level variable: ``qtot_mean``, ``qr``, or ``local_temp``.
    temporal_res
        ``"annual"`` or ``"seasonal2step"``.
    hydro_model
        Hydrological model name for dataset selection.
    percentile
        Uncertainty percentile suffix (e.g. ``"p10"``).

    Returns
    -------
    np.ndarray or tuple
        Annual: ``(217, n_years)``.
        Seasonal: ``((217, n_years), (217, n_years))`` tuple (dry, wet).
    """
    gmt_array = np.asarray(gmt_array)

    if temporal_res == "seasonal2step":
        gmt_clipped = clip_gmt(gmt_array, gmt_min=0.8, gmt_ceil=1.2)
    else:
        gmt_clipped = clip_gmt(gmt_array, gmt_min=0.6, gmt_ceil=0.9)

    window = "11"
    dataset_path = impacts_data_path(
        "rime",
        f"rime_regionarray_{variable}_{hydro_model}_{temporal_res}_window{window}.nc",
    )

    basin_mapping = load_basin_mapping()
    basin_id_to_rime_idx = _get_rime_region_mapping()

    def _expand(raw: np.ndarray) -> np.ndarray:
        return split_basin_macroregion(raw, basin_mapping, basin_id_to_rime_idx)

    if temporal_res == "seasonal2step":
        sfx = f"_{percentile}" if percentile else ""
        raw_dry = predict_rime(gmt_clipped, dataset_path, f"{variable}_dry{sfx}")
        raw_wet = predict_rime(gmt_clipped, dataset_path, f"{variable}_wet{sfx}")
        return (_expand(raw_dry), _expand(raw_wet))

    var_name = variable if not percentile else f"{variable}_{percentile}"
    return _expand(predict_rime(gmt_clipped, dataset_path, var_name))


# ---------------------------------------------------------------------------
# MESSAGE-format water CID preparation
# ---------------------------------------------------------------------------

_MESSAGE_YEARS = [
    2020, 2025, 2030, 2035, 2040, 2045, 2050,
    2055, 2060, 2070, 2080, 2090, 2100, 2110,
]


def _rime_to_wide(
    raw: np.ndarray,
    bcu_names: pd.Series,
    annual_years: list[int],
) -> pd.DataFrame:
    """Wrap a (217, n_years) array as a wide DataFrame with BCU_name index."""
    n = min(raw.shape[1], len(annual_years))
    df = pd.DataFrame(raw[:, :n], columns=annual_years[:n])
    df.insert(0, "BCU_name", bcu_names.values)
    return df


def _to_demand_long(
    wide_df: pd.DataFrame,
    commodity: str,
    msg_years: list[int],
    time_val: str = "year",
) -> pd.DataFrame:
    """Wide (217 × msg_years) → long MESSAGE demand format.

    Converts km³ → MCM (×1000) and negates (MESSAGE demand convention).
    """
    long = wide_df.melt(
        id_vars=["BCU_name"], value_vars=msg_years,
        var_name="year", value_name="value",
    )
    result = pd.DataFrame({
        "node": "B" + long["BCU_name"].astype(str),
        "commodity": commodity,
        "level": "water_avail_basin",
        "year": long["year"],
        "time": time_val,
        "value": -long["value"] * 1000,
        "unit": "MCM/year",
    })
    result.loc[result["value"] > 0, "value"] = 0.0
    return result


def _to_share_long(
    qtot: pd.DataFrame,
    qr: pd.DataFrame,
    msg_years: list[int],
    time_val: str = "year",
) -> pd.DataFrame:
    """Compute groundwater share and convert to MESSAGE format.

    Share = 0.95 × qr / qtot, representing sustainable GW fraction.
    """
    share_vals = (qr[msg_years] / qtot[msg_years] * 0.95).clip(0, 1).fillna(0)
    share_df = pd.concat([qtot[["BCU_name"]], share_vals], axis=1)
    long = share_df.melt(
        id_vars=["BCU_name"], value_vars=msg_years,
        var_name="year", value_name="value",
    )
    return pd.DataFrame({
        "shares": "share_low_lim_GWat",
        "node_share": "B" + long["BCU_name"].astype(str),
        "year_act": long["year"],
        "time": time_val,
        "value": long["value"],
        "unit": "-",
    })


def _filter_to_existing(
    new_df: pd.DataFrame,
    old_df: pd.DataFrame,
    node_col: str,
    key_cols: list[str],
) -> pd.DataFrame:
    """Keep basins that exist in old; fall back to old for NaN basins."""
    existing = set(old_df[node_col].unique())
    candidate = new_df[new_df[node_col].isin(existing)].copy()

    candidate["_bid"] = candidate[node_col].str.extract(r"B(\d+)").astype(int)
    nan_mask = candidate["_bid"].isin(NAN_BASIN_IDS) | candidate["value"].isna()
    valid = candidate[~nan_mask].drop(columns=["_bid"])
    nan_keys = candidate.loc[nan_mask, key_cols]

    preserved = old_df.merge(nan_keys, on=key_cols, how="inner")

    missing_nodes = existing - set(valid[node_col].unique())
    missing = (
        old_df[old_df[node_col].isin(missing_nodes)]
        if missing_nodes else pd.DataFrame()
    )

    return pd.concat([valid, preserved, missing], ignore_index=True)


def build_water_cids(
    gmt_array: np.ndarray,
    sw_old: pd.DataFrame,
    gw_old: pd.DataFrame,
    share_old: pd.DataFrame,
    msg_years: list[int] | None = None,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Build water CID replacement DataFrames from GMT ensemble.

    Parameters
    ----------
    gmt_array
        GMT values, shape ``(n_runs, n_years)`` or ``(n_years,)``.
        Covers annual years starting from 2020.
    sw_old
        Existing surfacewater_basin demand rows from scenario.
    gw_old
        Existing groundwater_basin demand rows from scenario.
    share_old
        Existing share_commodity_lo rows for GW share from scenario.
    msg_years
        Target MESSAGE years. Default: standard 14-step grid.

    Returns
    -------
    dict
        ``"sw"``: ``(old, new)`` surfacewater demand DataFrames.
        ``"gw"``: ``(old, new)`` groundwater demand DataFrames.
        ``"share"``: ``(old, new)`` groundwater share DataFrames.
    """
    if msg_years is None:
        msg_years = _MESSAGE_YEARS

    # Predict at 217-basin resolution
    qtot_raw = predict_water_rime(gmt_array, "qtot_mean")
    qr_raw = predict_water_rime(gmt_array, "qr")

    # Build wide DataFrames
    basin_mapping = load_basin_mapping()
    annual_years = list(range(2020, 2101))
    bcu = basin_mapping["BCU_name"]

    qtot_wide = _rime_to_wide(qtot_raw, bcu, annual_years)
    qr_wide = _rime_to_wide(qr_raw, bcu, annual_years)

    # Resample to MESSAGE years
    qtot_msg = sample_to_model_years(qtot_wide, ["BCU_name"], msg_years)
    qr_msg = sample_to_model_years(qr_wide, ["BCU_name"], msg_years)

    # Surfacewater = residual (qtot - qr)
    sw_wide = qtot_msg.copy()
    sw_wide[msg_years] = qtot_msg[msg_years] - qr_msg[msg_years]

    # Convert to MESSAGE format
    sw_new = _to_demand_long(sw_wide, "surfacewater_basin", msg_years)
    gw_new = _to_demand_long(qr_msg, "groundwater_basin", msg_years)
    share_new = _to_share_long(qtot_msg, qr_msg, msg_years)

    # Filter to existing basins, preserve NaN basins
    sw_filt = _filter_to_existing(sw_new, sw_old, "node", ["node", "year", "time"])
    gw_filt = _filter_to_existing(gw_new, gw_old, "node", ["node", "year", "time"])
    share_filt = _filter_to_existing(
        share_new, share_old, "node_share", ["node_share", "year_act", "time"]
    )

    log.info(
        "Built water CIDs: %d sw, %d gw, %d share rows",
        len(sw_filt), len(gw_filt), len(share_filt),
    )

    return {
        "sw": (sw_old, sw_filt),
        "gw": (gw_old, gw_filt),
        "share": (share_old, share_filt),
    }
