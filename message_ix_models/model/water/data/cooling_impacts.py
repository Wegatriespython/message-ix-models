"""Cooling CID: thermoelectric capacity degradation under warming.

Two mechanisms, two MESSAGE representations:

**Wet (freshwater) cooling** — Jones et al. degradation. Warming reduces
the thermal margin between discharge limits and intake temperature.
Modeled via ``relation_activity`` constraints bounding freshwater cooling
activity as a function of warming::

    ACT[p__cl_fresh] + ACT[p__ot_fresh]
        <= r_jones(r,t) * s_ref(r) * f_cool(p) * ACT[p]

**Dry (air) cooling** — Qin et al. thermodynamic derating. Higher ambient
temperature increases condenser backpressure and parasitic fan load.
Modeled via direct ``capacity_factor`` replacement on ``__air``
technologies — the degradation is a derating of the unit, not a resource
constraint.

Data: ``r12_thermoelectric_gwl.nc`` (cooling × 12 R12 regions × 72 GWL
bins). Capacity-weighted country→R12 aggregation from plant-level
simulation.
"""

import functools
import logging

import numpy as np
import pandas as pd

from message_ix_models.tools.impacts import (
    ReductionMode,
    clip_gmt,
    impacts_data_path,
    predict_with_reduction,
)
from message_ix_models.util import package_data_path
from message_ix_models.util.node import extract_region_code

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DATASET = "r12_thermoelectric_gwl.nc"
_VAR = "capacity_factor"
_DEFAULT_BASELINE_GWL = 1.0
_DEFAULT_MIN_YEAR = 2045

_WET_SEL = {"cooling": "wet"}
_DRY_SEL = {"cooling": "dry"}


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _region_codes() -> list[str]:
    """Short region codes from the capacity factor dataset."""
    import xarray as xr

    ds = xr.open_dataset(str(impacts_data_path("rime", _DATASET)))
    return list(ds.region.values)


@functools.lru_cache(maxsize=1)
def _freshwater_reference_shares() -> pd.Series:
    """Regional average freshwater share (cl_fresh + ot_fresh).

    Returns Series indexed by short region code (e.g. "AFR").
    """
    path = package_data_path(
        "water", "ppl_cooling_tech", "cooltech_cost_and_shares_ssp_msg_R12.csv"
    )
    df = pd.read_csv(path)
    mix_cols = [c for c in df.columns if c.startswith("mix_")]

    fresh = df[df["cooling"].isin(["cl_fresh", "ot_fresh"])]
    # Sum cl_fresh + ot_fresh shares per region, averaged across parent techs
    regional_fresh = fresh.groupby("cooling")[mix_cols].mean().sum()

    # Convert column names: "mix_R12_AFR" -> "AFR"
    return regional_fresh.rename(index=lambda c: c.replace("mix_R12_", ""))


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_cooling_cf(
    gmt_array: np.ndarray,
    cooling: str = "wet",
    reduction: ReductionMode = "mean",
) -> pd.DataFrame:
    """Predict regional capacity factors from GMT.

    Parameters
    ----------
    gmt_array
        GMT values in degC above pre-industrial. Shape ``(n_years,)`` or
        ``(n_runs, n_years)`` for ensemble.
    cooling
        ``"wet"`` (freshwater) or ``"dry"`` (air).
    reduction
        How to reduce the (MAGICC, RIME-source) ensemble pair into a per-cell
        CF. See :data:`message_ix_models.tools.impacts.ReductionMode`.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with ``region`` index (short codes) and one column
        per GMT input position. Values are capacity factors (fractions).
    """
    sel = _WET_SEL if cooling == "wet" else _DRY_SEL
    gmt_array = np.asarray(gmt_array)
    gmt_clipped = clip_gmt(gmt_array, gmt_min=0.6, gmt_ceil=0.9)

    dataset_path = impacts_data_path("rime", _DATASET)
    raw = predict_with_reduction(
        gmt_clipped, dataset_path, _VAR, sel=sel, reduction=reduction
    )
    # raw shape: (12, n_years) — regions x time positions

    regions = _region_codes()
    return pd.DataFrame(raw, index=pd.Index(regions, name="region"))


def compute_degradation_ratios(
    gmt_array: np.ndarray,
    years: list[int],
    cooling: str = "wet",
    baseline_gwl: float = _DEFAULT_BASELINE_GWL,
    reduction: ReductionMode = "mean",
) -> pd.DataFrame:
    """Compute degradation ratios: CF(GMT) / CF(baseline).

    Parameters
    ----------
    gmt_array
        GMT trajectory. Shape ``(n_years,)`` or ``(n_runs, n_years)``.
    years
        Year labels for the time axis. Length must match ``n_years``.
    cooling
        ``"wet"`` (freshwater) or ``"dry"`` (air).
    baseline_gwl
        Reference warming level (degC). Default 1.0.
    reduction
        How to reduce the ensemble pair. The same mode is used for the
        trajectory and the baseline so both come from the same RIME column —
        otherwise the ratio would compare a mean to a percentile.

    Returns
    -------
    pd.DataFrame
        Rows = R12 region short codes, columns = *years*. Values are ratios
        relative to baseline — below 1 under warming, above 1 if GMT
        dips below baseline.
    """
    sel = _WET_SEL if cooling == "wet" else _DRY_SEL
    cf = predict_cooling_cf(gmt_array, cooling=cooling, reduction=reduction)

    dataset_path = impacts_data_path("rime", _DATASET)
    cf_baseline = predict_with_reduction(
        np.array([baseline_gwl]), dataset_path, _VAR, sel=sel, reduction=reduction
    )[:, 0]  # (12,)

    ratios = cf.values / cf_baseline[:, np.newaxis]
    return pd.DataFrame(ratios, index=cf.index.copy(), columns=years)


# Keep old name as alias for backward compatibility in cid_pipeline.py
compute_jones_ratios = compute_degradation_ratios


# ---------------------------------------------------------------------------
# Constraint building
# ---------------------------------------------------------------------------


def _read_cooling_structure(
    addon_df: pd.DataFrame,
    technologies: set[str],
) -> pd.DataFrame:
    """Extract freshwater cooling techs and their parent-tech cooling fractions.

    Parameters
    ----------
    addon_df
        The ``addon_conversion`` parameter DataFrame from a MESSAGE scenario.
    technologies
        Set of technology names present in the scenario.

    Returns
    -------
    pd.DataFrame
        Columns: parent_tech, cl_fresh_tech, ot_fresh_tech, node_loc,
        cooling_fraction. One row per (parent_tech, node_loc).
    """
    if addon_df.empty:
        raise ValueError("addon_conversion is empty — cooling module not built?")

    # addon_conversion has type_addon = "cooling__<parent>", value = cooling_fraction
    cooling_addon = addon_df[addon_df["type_addon"].str.startswith("cooling__")].copy()
    cooling_addon["parent_tech"] = cooling_addon["type_addon"].str.replace(
        "cooling__", "", n=1
    )

    # Deduplicate: one cooling_fraction per (parent_tech, node_loc)
    # Take mean across vintage/year combinations — cooling_fraction is
    # physically constant for a given parent tech
    grouped = (
        cooling_addon.groupby(["parent_tech", "node"])["value"]
        .mean()
        .reset_index()
        .rename(columns={"node": "node_loc", "value": "cooling_fraction"})
    )

    rows = []
    for _, row in grouped.iterrows():
        parent = row["parent_tech"]
        cl = f"{parent}__cl_fresh"
        ot = f"{parent}__ot_fresh"
        if cl in technologies or ot in technologies:
            rows.append(
                {
                    "parent_tech": parent,
                    "cl_fresh_tech": cl if cl in technologies else None,
                    "ot_fresh_tech": ot if ot in technologies else None,
                    "node_loc": row["node_loc"],
                    "cooling_fraction": row["cooling_fraction"],
                }
            )

    if not rows:
        raise ValueError("No freshwater cooling technologies found")

    return pd.DataFrame(rows)


def _max_vintage_from_lifetime(
    tl: pd.DataFrame,
) -> dict[tuple[str, str], int]:
    """Last vintage year with ``technical_lifetime`` defined, per (node, tech).

    GAMS evaluates ``relation_activity`` by iterating all model years
    ``<= year_act`` as candidate vintages.  If a candidate vintage has no
    ``technical_lifetime`` row, compilation fails ("Technical lifetime not
    defined for node|tech|year").  The safe bound for ``year_act`` is
    therefore ``max(year_vtg)`` where ``technical_lifetime`` is defined —
    any ``year_act`` beyond that will cause GAMS to probe a vintage year
    with no lifetime entry.
    """
    return {
        (node, tech): int(grp["year_vtg"].max())
        for (node, tech), grp in tl.groupby(["node_loc", "technology"])
    }


def build_cooling_constraints(
    addon_df: pd.DataFrame,
    technologies: set[str],
    jones_ratios: pd.DataFrame,
    model_years: list[int] | None = None,
    min_year: int = _DEFAULT_MIN_YEAR,
    technical_lifetime: pd.DataFrame | None = None,
) -> dict:
    """Build ``relation_activity`` and ``relation_upper`` for Jones constraints.

    Parameters
    ----------
    addon_df
        The ``addon_conversion`` parameter DataFrame from a MESSAGE scenario.
    technologies
        Set of technology names present in the scenario.
    jones_ratios
        Output of :func:`compute_jones_ratios`. Rows = regions (short codes),
        columns = model years.
    model_years
        Which years to constrain. If *None*, uses jones_ratios columns.
    min_year
        Earliest year for constraints. Default 2045.
    technical_lifetime
        The ``technical_lifetime`` parameter DataFrame. When provided,
        ``relation_activity`` rows are only emitted for ``year_act`` values
        where the parent tech has active vintage capacity.  Without this,
        phase-out techs (e.g. ``coal_ppl_u``) produce GAMS errors:
        "Technical lifetime not defined for node|tech|year".

    Returns
    -------
    dict
        ``"relation_activity"``: DataFrame of relation coefficients.
        ``"relation_upper"``: DataFrame of upper bounds (all zero).
        ``"relation_names"``: list of relation name strings to add to
        the ``relation`` set.
    """
    structure = _read_cooling_structure(addon_df, technologies)
    s_ref = _freshwater_reference_shares()

    if model_years is None:
        model_years = [int(c) for c in jones_ratios.columns]
    constrained_years = [y for y in model_years if y >= min_year]

    if not constrained_years:
        log.warning("No model years >= %d; returning empty constraints", min_year)
        return {
            "relation_activity": pd.DataFrame(),
            "relation_upper": pd.DataFrame(),
            "relation_names": [],
        }

    # Max vintage per (node, tech) — year_act beyond this triggers GAMS error
    max_vtg = (
        _max_vintage_from_lifetime(technical_lifetime)
        if technical_lifetime is not None
        else None
    )

    rel_act_rows = []
    rel_up_rows = []
    relation_names = set()
    n_skipped = 0

    # Group by parent tech — one relation per parent type
    for parent_tech, group in structure.groupby("parent_tech"):
        rel_name = f"jones_cool_{parent_tech}"
        relation_names.add(rel_name)

        for _, row in group.iterrows():
            node = row["node_loc"]  # "R12_AFR"
            region_short = extract_region_code(node)
            f_cool = row["cooling_fraction"]

            # Reference freshwater share for this region
            if region_short not in s_ref.index:
                log.warning("No freshwater share for region %s, skipping", region_short)
                continue
            share = float(s_ref[region_short])

            # Last vintage year with technical_lifetime defined
            vtg_cap = (
                max_vtg.get((node, parent_tech))
                if max_vtg is not None
                else None
            )

            for year in constrained_years:
                # Skip if year_act exceeds last defined vintage
                if vtg_cap is not None and year > vtg_cap:
                    n_skipped += 1
                    continue

                # Jones ratio for this region-year
                if region_short not in jones_ratios.index:
                    continue
                if year not in jones_ratios.columns:
                    log.warning("Year %d not in jones_ratios columns, skipping", year)
                    continue
                r_jones = float(jones_ratios.loc[region_short, year])

                # Parent tech coefficient: negative (RHS of inequality)
                parent_coeff = -(r_jones * share * f_cool)

                rel_act_rows.append(
                    {
                        "relation": rel_name,
                        "node_rel": node,
                        "year_rel": year,
                        "node_loc": node,
                        "technology": row["parent_tech"],
                        "year_act": year,
                        "mode": "M1",
                        "value": parent_coeff,
                        "unit": "-",
                    }
                )

                # Freshwater variant coefficients: +1 each
                for tech in (row["cl_fresh_tech"], row["ot_fresh_tech"]):
                    if tech is not None:
                        rel_act_rows.append(
                            {
                                "relation": rel_name,
                                "node_rel": node,
                                "year_rel": year,
                                "node_loc": node,
                                "technology": tech,
                                "year_act": year,
                                "mode": "M1",
                                "value": 1.0,
                                "unit": "-",
                            }
                        )

                # Upper bound = 0
                rel_up_rows.append(
                    {
                        "relation": rel_name,
                        "node_rel": node,
                        "year_rel": year,
                        "value": 0.0,
                        "unit": "-",
                    }
                )

    rel_act = pd.DataFrame(rel_act_rows)
    rel_up = pd.DataFrame(rel_up_rows)

    n_parents = len(relation_names)
    n_entries = len(rel_act)
    log.info(
        "Built Jones cooling constraints: %d relations, %d relation_activity entries, "
        "%d skipped (no active vintage), years %d-%d",
        n_parents,
        n_entries,
        n_skipped,
        min(constrained_years),
        max(constrained_years),
    )

    return {
        "relation_activity": rel_act,
        "relation_upper": rel_up,
        "relation_names": sorted(relation_names),
    }


# ---------------------------------------------------------------------------
# Dry cooling: capacity_factor replacement
# ---------------------------------------------------------------------------


def build_dry_cooling_factors(
    cf_air: pd.DataFrame,
    dry_ratios: pd.DataFrame,
    model_years: list[int],
    min_year: int = _DEFAULT_MIN_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build replacement ``capacity_factor`` rows for ``__air`` technologies.

    Parameters
    ----------
    cf_air
        Existing ``capacity_factor`` rows for ``__air`` technologies,
        as returned by ``scenario.par("capacity_factor", ...)``.
    dry_ratios
        Output of :func:`compute_degradation_ratios` with ``cooling="dry"``.
        Rows = R12 short codes, columns = model years.
    model_years
        Which years the ratios cover.
    min_year
        Earliest year for modification. Default 2045.

    Returns
    -------
    old_cf
        Rows that will be removed (affected years only).
    new_cf
        Replacement rows with values scaled by dry degradation ratio.
    """
    if cf_air.empty:
        log.warning("No __air capacity_factor rows provided")
        return pd.DataFrame(), pd.DataFrame()

    constrained_years = [y for y in model_years if y >= min_year]
    if not constrained_years:
        log.warning("No model years >= %d for dry cooling", min_year)
        return pd.DataFrame(), pd.DataFrame()

    # Filter to affected years
    cf_affected = cf_air[cf_air["year_act"].isin(constrained_years)].copy()
    if cf_affected.empty:
        log.warning("No __air capacity_factor rows for years %s", constrained_years)
        return pd.DataFrame(), pd.DataFrame()

    old_cf = cf_affected.copy()

    # Apply dry degradation ratio per (node, year)
    new_cf = cf_affected.copy()
    for idx, row in new_cf.iterrows():
        region_short = extract_region_code(row["node_loc"])
        year = row["year_act"]
        if region_short in dry_ratios.index and year in dry_ratios.columns:
            ratio = float(dry_ratios.loc[region_short, year])
            new_cf.at[idx, "value"] = row["value"] * ratio

    n_modified = len(new_cf)
    log.info(
        "Built dry cooling factors: %d rows, years %d-%d",
        n_modified,
        min(constrained_years),
        max(constrained_years),
    )

    return old_cf, new_cf
