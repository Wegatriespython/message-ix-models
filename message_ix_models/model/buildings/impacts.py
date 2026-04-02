"""Building energy CID integration.

The model's rc_spec (cooling) and rc_therm (heating) demands include a
buildings component from calibrated STURM projections. Those projections
held energy intensity (EI) fixed over the horizon — no climate response.
This module replaces that fixed-EI buildings component with a
climate-responsive version using RIME-predicted EI.

Sector fractions identify what share of rc_spec/rc_therm is the STURM
buildings component. We subtract it (the fixed-EI calibration) and add:

    E(t,r,a) = gamma(r,a) * EI(r,a, GSAT(t)) * F(t,r,a)

Where gamma is the correction coefficient (calibrated at GWL=1.2 to
match STURM at reference climate), EI is the CHILLED energy intensity
from RIME emulators (MJ/m2), and F is the STURM floor area (Mm2).

Ensemble averaging: callers pass a 2D GMT array (n_runs, n_years) from
MAGICC. ``predict_rime`` evaluates EI at each ensemble member's GMT and
averages — E[f(GMT)] not f(E[GMT]) — smoothing bin-edge artifacts from
the emulator's discrete GWL bins.

Residential archetypes (mfh_*, sfh_*) map directly to RIME coordinates.
Commercial archetypes (comm_*) use MFH floor-weighted residential EI
as proxy (STURM precedent).

Reference: project.alps.replace_building_cids (PR #466).
"""

import functools
import logging
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from message_ix import Scenario

from message_ix_models.tools.impacts import (
    clip_gmt,
    impacts_data_path,
    predict_rime,
)
from message_ix_models.util import package_data_path

log = logging.getLogger(__name__)

# E [EJ] = gamma * EI [MJ/m2] * F [Mm2] / 1e6
# MJ/m2 * Mm2 = MJ * 1e6 = TJ; TJ / 1e6 = EJ
_MJ_MM2_TO_EJ = 1e-6

# MESSAGE demand parameter expects GWa
# 1 EJ = 1e18 / (1e9 * 365.25 * 24 * 3600) GWa
_EJ_TO_GWA = 1e18 / (1e9 * 365.25 * 24 * 3600)

_REFERENCE_SCENARIO = "SSP2"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _buildings_data_path(*parts: str):
    return package_data_path("buildings", *parts)


@functools.lru_cache(maxsize=2)
def _load_ei_dataset(mode: Literal["cool", "heat"]) -> xr.Dataset:
    return xr.open_dataset(
        str(impacts_data_path("rime", f"region_EI_{mode}_gwl_binned.nc"))
    )


def load_correction_coefficients(
    mode: Literal["cool", "heat"],
    scenario: str = "S1",
    sector: Literal["resid", "comm"] = "resid",
) -> pd.DataFrame:
    """Load gamma and F per (region, arch, urt, year)."""
    path = _buildings_data_path(
        "correction_coefficients",
        f"correction_coefficients_{mode}_{scenario}_{sector}.csv",
    )
    return pd.read_csv(path, comment="#")


def load_sector_fractions() -> pd.DataFrame:
    """Load SSP2 sector fractions of rc_spec/rc_therm per (node, year)."""
    return pd.read_csv(_buildings_data_path("rc_sector_fractions.csv"), comment="#")


def load_floor_areas(sector: Literal["resid", "comm"] = "resid") -> pd.DataFrame:
    """Load SSP2 STURM floor area projections."""
    return pd.read_csv(_buildings_data_path(f"sturm_floor_area_R12_{sector}.csv"))


def load_theta(
    mode: Literal["cool", "heat"],
    reference_scenario: str = _REFERENCE_SCENARIO,
) -> pd.DataFrame:
    """Load theta(node, year) calibrated to the reference scenario demand."""
    path = _buildings_data_path(f"theta_{mode}_{reference_scenario}.csv")
    return pd.read_csv(path, comment="#")


# ---------------------------------------------------------------------------
# EI prediction
# ---------------------------------------------------------------------------


def predict_building_ei(
    gmt_array: np.ndarray,
    mode: Literal["cool", "heat"],
) -> np.ndarray:
    """Predict E[EI(GMT)] at native RIME resolution.

    Parameters
    ----------
    gmt_array
        Shape ``(n_runs, n_years)`` for ensemble or ``(n_years,)`` for single
        trajectory.

    Returns
    -------
    np.ndarray
        Shape ``(12, 10, 3, n_years)`` — (region, arch, urt, year).
    """
    dataset_path = str(impacts_data_path("rime", f"region_EI_{mode}_gwl_binned.nc"))
    gmt_clipped = clip_gmt(gmt_array, gmt_min=0.6, gmt_ceil=0.9)
    return predict_rime(gmt_clipped, dataset_path, f"EI_{mode}", aggregate="mean")


def _ei_to_dataframe(
    ei_all: np.ndarray,
    ds: xr.Dataset,
    msg_years: list[int],
) -> pd.DataFrame:
    """Flatten EI prediction array to long DataFrame.

    Returns columns [region, arch, urt, year, ei].
    """
    regions = ds.region.values
    archs = ds.arch.values
    urts = ds.urt.values
    idx = pd.MultiIndex.from_product(
        [regions, archs, urts, msg_years],
        names=["region", "arch", "urt", "year"],
    )
    return pd.DataFrame({"ei": ei_all.ravel()}, index=idx).reset_index()


def _mfh_weighted_ei(
    ei_df: pd.DataFrame,
    resid_floor_df: pd.DataFrame,
) -> pd.DataFrame:
    """Floor-weighted average of MFH archetype EI per (region, urt, year).

    Commercial buildings use residential MFH archetypes weighted by 2020
    floor area (STURM precedent).

    Returns columns [region, urt, year, ei].
    """
    # MFH floor weights at 2020 reference year
    mfh_floors = resid_floor_df[
        (resid_floor_df["arch"].str.startswith("mfh_"))
        & (resid_floor_df["year"] == 2020)
        & (resid_floor_df["floor_Mm2"] > 0)
    ][["region", "arch", "urt", "floor_Mm2"]]

    if mfh_floors.empty:
        return pd.DataFrame(columns=["region", "urt", "year", "ei"])

    # Join EI with floor weights
    mfh_ei = ei_df[ei_df["arch"].str.startswith("mfh_")].merge(
        mfh_floors,
        on=["region", "arch", "urt"],
        how="inner",
    )
    mfh_ei["weighted_ei"] = mfh_ei["ei"] * mfh_ei["floor_Mm2"]

    # Weighted average per (region, urt, year)
    grouped = mfh_ei.groupby(["region", "urt", "year"], as_index=False).agg(
        weighted_sum=("weighted_ei", "sum"),
        floor_sum=("floor_Mm2", "sum"),
    )
    grouped["ei"] = grouped["weighted_sum"] / grouped["floor_sum"]
    return grouped[["region", "urt", "year", "ei"]]


# ---------------------------------------------------------------------------
# Energy demand computation
# ---------------------------------------------------------------------------


def _compute_sector_energy(
    mode: Literal["cool", "heat"],
    sector: Literal["resid", "comm"],
    ei_df: pd.DataFrame,
    mfh_ei_df: pd.DataFrame,
    scenario: str = "S1",
) -> pd.DataFrame:
    """Compute energy demand for one mode/sector via vectorized join.

    Returns DataFrame with columns [region, year, value] in EJ.
    """
    coeff = load_correction_coefficients(mode, scenario, sector)
    valid = coeff["correction_coeff"].notna() & (coeff["floor_Mm2"] > 0)
    coeff = coeff[valid]

    if sector == "resid":
        # Direct join on (region, arch, urt, year)
        merged = coeff.merge(ei_df, on=["region", "arch", "urt", "year"], how="left")
    else:
        # Commercial: join MFH-weighted EI on (region, urt, year)
        merged = coeff.merge(mfh_ei_df, on=["region", "urt", "year"], how="left")

    has_ei = merged["ei"].notna() & (merged["ei"] > 0)
    n_total = len(merged)
    n_computed = has_ei.sum()
    log.info(
        "%s/%s: %d/%d rows with valid EI (%.0f%%)",
        mode,
        sector,
        n_computed,
        n_total,
        100 * n_computed / n_total if n_total else 0,
    )

    merged = merged[has_ei]
    merged["value"] = (
        merged["correction_coeff"] * merged["ei"] * merged["floor_Mm2"] * _MJ_MM2_TO_EJ
    )
    return merged.groupby(["region", "year"], as_index=False)["value"].sum()


def _apply_theta(
    demand: pd.DataFrame,
    mode: Literal["cool", "heat"],
    reference_scenario: str = _REFERENCE_SCENARIO,
) -> pd.DataFrame:
    """Scale raw RIME demand by theta(node, year) from the reference scenario."""
    theta = load_theta(mode, reference_scenario)
    scaled = demand.merge(theta, on=["node", "year"], how="left")

    missing = scaled.loc[scaled["theta"].isna(), ["node", "year"]]
    if not missing.empty:
        pairs = missing.drop_duplicates().to_dict("records")
        raise ValueError(
            f"Missing theta values for {mode}/{reference_scenario}: {pairs}"
        )

    return scaled.assign(value=lambda df: df["value"] * df["theta"])[
        ["node", "year", "value"]
    ]


def compute_building_cids(
    gmt_array: np.ndarray,
    years: np.ndarray,
    model_years: list[int],
    coeff_scenario: str = "S1",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute building energy CIDs from GMT ensemble.

    Parameters
    ----------
    gmt_array
        GMT values (degC above pre-industrial). Shape ``(n_runs, n_years)``
        for ensemble averaging or ``(n_years,)`` for single trajectory.
    years
        Year labels corresponding to gmt_array columns.
    model_years
        MESSAGE model years from ``ScenarioInfo.Y``. Only positions in
        *years* that match these are used; 2110 is forward-filled from 2100.
    coeff_scenario
        Correction coefficient scenario ('S1', 'S2', 'S3').

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(cooling, heating)`` DataFrames with columns
        ``[node, year, value]`` where value is GWa. Node uses R12_ prefix.
    """
    gmt_array = np.asarray(gmt_array)
    years = np.asarray(years, dtype=int)

    # Subset to model years within input range (exclude 2110 — forward-filled)
    target = [y for y in model_years if y != 2110]
    msg_mask = np.isin(years, target)
    if not msg_mask.any():
        raise ValueError(f"No model years in input. Years: {years[0]}-{years[-1]}")
    gmt_subset = gmt_array[:, msg_mask] if gmt_array.ndim == 2 else gmt_array[msg_mask]
    msg_years = years[msg_mask].tolist()

    log.info(
        "Computing building CIDs: %s scenario, %d model years, GMT shape %s",
        coeff_scenario,
        len(msg_years),
        gmt_subset.shape,
    )

    # Pre-compute MFH-weighted EI for commercial (shared across cool/heat)
    resid_floor = load_floor_areas("resid")

    results = {}
    for mode in ("cool", "heat"):
        ei_all = predict_building_ei(gmt_subset, mode)
        ds = _load_ei_dataset(mode)
        ei_df = _ei_to_dataframe(ei_all, ds, msg_years)
        mfh_ei_df = _mfh_weighted_ei(ei_df, resid_floor)

        resid = _compute_sector_energy(mode, "resid", ei_df, mfh_ei_df, coeff_scenario)
        comm = _compute_sector_energy(mode, "comm", ei_df, mfh_ei_df, coeff_scenario)

        total = (
            pd.concat([resid, comm], ignore_index=True)
            .groupby(["region", "year"], as_index=False)["value"]
            .sum()
            .assign(
                node=lambda df: "R12_" + df["region"],
                value=lambda df: df["value"] * _EJ_TO_GWA,
            )[["node", "year", "value"]]
        )
        total = _apply_theta(total, mode)

        # Forward-fill 2110 from 2100 if 2110 is a model year
        if 2110 in model_years and 2100 in total["year"].values:
            total = pd.concat(
                [total, total[total["year"] == 2100].assign(year=2110)],
                ignore_index=True,
            )

        results[mode] = total.sort_values(["node", "year"]).reset_index(drop=True)

    return results["cool"], results["heat"]


# ---------------------------------------------------------------------------
# Scenario modification
# ---------------------------------------------------------------------------


def prepare_building_demand(
    rc_spec: pd.DataFrame,
    rc_therm: pd.DataFrame,
    cooling_cids: pd.DataFrame,
    heating_cids: pd.DataFrame,
    fractions: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace the STURM buildings component in rc_spec/rc_therm DataFrames.

    Subtracts the climate-driven fraction (from sector fractions) and adds
    the RIME-predicted CID demand. Pure data operation — no scenario I/O.

    Parameters
    ----------
    rc_spec
        Existing ``demand`` rows for commodity ``rc_spec``.
    rc_therm
        Existing ``demand`` rows for commodity ``rc_therm``.
    cooling_cids
        From ``compute_building_cids`` — columns [node, year, value] in GWa.
    heating_cids
        Same format, heating.
    fractions
        Sector fractions. If *None*, loaded from package data.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(new_rc_spec, new_rc_therm)`` — replacement demand DataFrames
        with the same columns as the inputs.
    """
    if fractions is None:
        fractions = load_sector_fractions()

    rc_spec = rc_spec.copy()
    rc_therm = rc_therm.copy()

    # Merge sector fractions; fill missing with 0 (no climate reduction)
    frac_cool_cols = ["node", "year", "frac_resid_cool", "frac_comm_cool"]
    frac_heat_cols = ["node", "year", "frac_resid_heat", "frac_comm_heat"]
    rc_spec = rc_spec.merge(fractions[frac_cool_cols], on=["node", "year"], how="left")
    rc_therm = rc_therm.merge(
        fractions[frac_heat_cols], on=["node", "year"], how="left"
    )

    rc_spec[["frac_resid_cool", "frac_comm_cool"]] = rc_spec[
        ["frac_resid_cool", "frac_comm_cool"]
    ].fillna(0)
    rc_therm[["frac_resid_heat", "frac_comm_heat"]] = rc_therm[
        ["frac_resid_heat", "frac_comm_heat"]
    ].fillna(0)

    # Subtract climate-driven fraction, add RIME-predicted demand
    rc_spec["value"] *= 1 - rc_spec["frac_resid_cool"] - rc_spec["frac_comm_cool"]
    rc_therm["value"] *= 1 - rc_therm["frac_resid_heat"] - rc_therm["frac_comm_heat"]

    rc_spec = rc_spec.merge(
        cooling_cids.rename(columns={"value": "_cid"}),
        on=["node", "year"],
        how="left",
    )
    rc_therm = rc_therm.merge(
        heating_cids.rename(columns={"value": "_cid"}),
        on=["node", "year"],
        how="left",
    )
    rc_spec["value"] += rc_spec["_cid"].fillna(0)
    rc_therm["value"] += rc_therm["_cid"].fillna(0)

    # Drop working columns, return clean demand DataFrames
    demand_cols = [c for c in rc_spec.columns if not c.startswith("frac_") and c != "_cid"]
    return rc_spec[demand_cols], rc_therm[demand_cols]


def apply_building_cids(
    scen: Scenario,
    cooling_demand: pd.DataFrame,
    heating_demand: pd.DataFrame,
    commit_message: str | None = None,
) -> None:
    """Write replacement building demands to scenario.

    Thin wrapper: reads rc_spec/rc_therm, delegates to
    :func:`prepare_building_demand`, writes back via scenario primitives.

    Parameters
    ----------
    scen
        MESSAGE scenario (must not be checked out).
    cooling_demand
        From ``compute_building_cids`` — columns [node, year, value] in GWa.
    heating_demand
        Same format, heating.
    commit_message
        Commit message. Default: "Inject building CIDs".
    """
    demand = scen.par("demand")

    rc_spec = demand[demand["commodity"] == "rc_spec"].copy()
    rc_therm = demand[demand["commodity"] == "rc_therm"].copy()

    assert not rc_spec.empty, "No rc_spec rows in demand — buildings module not built?"
    assert not rc_therm.empty, (
        "No rc_therm rows in demand — buildings module not built?"
    )

    new_spec, new_therm = prepare_building_demand(
        rc_spec, rc_therm, cooling_demand, heating_demand
    )

    msg = commit_message or "Inject building CIDs"
    with scen.transact(msg):
        scen.remove_par("demand", demand[demand["commodity"] == "rc_spec"])
        scen.add_par("demand", new_spec)
        scen.remove_par("demand", demand[demand["commodity"] == "rc_therm"])
        scen.add_par("demand", new_therm)

    scen.set_as_default()
    log.info("Building CIDs applied and committed")
