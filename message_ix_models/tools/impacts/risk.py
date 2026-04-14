"""Ensemble reductions over RIME predictions.

Two families of reduction along the MAGICC run axis:

**CVaR (Conditional Value-at-Risk)** — :func:`cvar_pointwise`, :func:`cvar_coherent`.
Average the worst *alpha*% of MAGICC trajectories (independently per cell, or
temporally coherent across whole trajectories). Used for risk-tail analysis.

**Joint quantile** — :func:`joint_at_quantile`. Independently reduces the RIME
source-ensemble axis (via the precomputed ``_pXX`` columns shipped per
``(region, gwl)`` bin in the nc files) and the MAGICC ensemble axis (via
per-cell quantile across runs). The two reductions are independent, mirroring
RIME-X's law-of-total-probability framing in the limit where each axis is
collapsed at the requested quantile. Used for SPARRCLE ``_CI_X`` percentile
deliverables.

:func:`predict_with_reduction` is a thin dispatch helper consumed by domain
modules so they can swap reducers via a single ``reduction`` parameter
without touching the lookup wiring.

All reduction functions return ``(n_spatial, n_years)`` arrays at native
emulator resolution. Callers wrap in DataFrames if they need labels.
"""

from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np

from .rime import predict_rime

ReductionMode: TypeAlias = Literal["mean", "joint_p50"]
"""How to reduce a MAGICC ensemble + RIME source ensemble pair into one
per-cell value:

- ``"mean"`` — read the RIME mean field; take the MAGICC-axis mean. Current
  default and equivalent to ``predict_rime(..., aggregate="mean")``.
- ``"joint_p50"`` — read the RIME ``_p50`` field; take the MAGICC-axis median.
  Both axes at the 50th percentile.
"""

_SHIPPED_IMPACT_QUANTILES = (0.1, 0.5, 0.9)


def joint_at_quantile(
    gmt_array: np.ndarray,
    dataset_path: str | Path,
    var_name: str,
    q_impact: float = 0.5,
    q_warming: float = 0.5,
    sel: dict | None = None,
) -> np.ndarray:
    """Joint quantile reduction across RIME source and MAGICC warming axes.

    Reads ``f"{var_name}_p{int(q_impact*100)}"`` from the dataset to collapse
    the RIME source ensemble at ``q_impact``, then reduces the MAGICC run
    axis via ``np.quantile(..., q_warming, axis=0)``. The two reductions are
    independent. For 1D ``gmt_array`` the warming reduction is a no-op and
    the function returns the impact-percentile lookup directly.

    Parameters
    ----------
    gmt_array
        ``(n_years,)`` for a single trajectory or ``(n_runs, n_years)`` for
        an ensemble. degC above pre-industrial.
    dataset_path
        Path to a RIME NetCDF dataset that ships per-bin percentile siblings
        (``_p10``, ``_p50``, ``_p90``).
    var_name
        Base variable name without the ``_p{XX}`` suffix
        (e.g. ``"capacity_factor"``, ``"EI_cool"``).
    q_impact
        Quantile of the RIME source ensemble. Currently restricted to the
        shipped values ``{0.1, 0.5, 0.9}``. Continuous ``q_impact`` via 3-knot
        piecewise-linear CDF reconstruction is a documented extension path.
    q_warming
        Quantile of the MAGICC ensemble. Continuous in ``(0, 1)``.
    sel
        Optional dimension selections passed to :func:`~.rime.predict_rime`.

    Returns
    -------
    np.ndarray
        ``(n_spatial, n_years)``.
    """
    if q_impact not in _SHIPPED_IMPACT_QUANTILES:
        raise ValueError(
            f"q_impact must be one of {_SHIPPED_IMPACT_QUANTILES}, got {q_impact}; "
            "continuous interpolation across shipped percentiles is not yet implemented"
        )
    if not 0 < q_warming < 1:
        raise ValueError(f"q_warming must be in (0, 1), got {q_warming}")

    impact_var = f"{var_name}_p{int(round(q_impact * 100))}"

    gmt_array = np.asarray(gmt_array)
    if gmt_array.ndim == 1:
        return predict_rime(gmt_array, dataset_path, impact_var, sel=sel)

    ensemble = predict_rime(
        gmt_array, dataset_path, impact_var, sel=sel, aggregate="none"
    )
    return np.quantile(ensemble, q_warming, axis=0)


def predict_with_reduction(
    gmt_array: np.ndarray,
    dataset_path: str | Path,
    var_name: str,
    sel: dict | None = None,
    reduction: ReductionMode = "mean",
) -> np.ndarray:
    """Dispatch :func:`predict_rime` or :func:`joint_at_quantile` by mode.

    Domain modules import this and forward a ``reduction`` parameter without
    needing to know which underlying function maps to which mode. Returns a
    ``(n_spatial, n_years)`` array under both modes.
    """
    if reduction == "mean":
        return predict_rime(
            gmt_array, dataset_path, var_name, sel=sel, aggregate="mean"
        )
    if reduction == "joint_p50":
        return joint_at_quantile(
            gmt_array,
            dataset_path,
            var_name,
            q_impact=0.5,
            q_warming=0.5,
            sel=sel,
        )
    raise ValueError(f"Unsupported reduction mode: {reduction!r}")


def cvar_pointwise(
    gmt_array: np.ndarray,
    dataset_path: str | Path,
    var_name: str,
    alpha: float,
    sel: dict | None = None,
) -> np.ndarray:
    """Pointwise CVaR over RIME ensemble predictions.

    For each (spatial, year) cell independently, sorts runs and averages
    the worst *alpha*% — maximally pessimistic across timesteps.

    Parameters
    ----------
    gmt_array
        Shape ``(n_runs, n_years)``. Must be 2D.
    dataset_path
        Path to RIME NetCDF dataset.
    var_name
        Variable name within the dataset.
    alpha
        CVaR level as percentile (0 < alpha < 100). E.g. 10 = worst 10%.
    sel
        Optional dimension selections passed to :func:`~.rime.predict_rime`.

    Returns
    -------
    np.ndarray
        Shape ``(n_spatial, n_years)``.
    """
    if not 0 < alpha < 100:
        raise ValueError(f"alpha must be between 0 and 100, got {alpha}")
    ensemble = predict_rime(
        gmt_array, dataset_path, var_name, sel=sel, aggregate="none"
    )
    n_runs = ensemble.shape[0]
    cutoff = max(1, int(np.ceil(n_runs * alpha / 100.0)))
    return np.mean(np.sort(ensemble, axis=0)[:cutoff], axis=0)


def cvar_coherent(
    gmt_array: np.ndarray,
    dataset_path: str | Path,
    var_name: str,
    alpha: float,
    sel: dict | None = None,
) -> np.ndarray:
    """Coherent CVaR over RIME ensemble predictions.

    Ranks trajectories by mean impact across all spatial units and years,
    selects the worst *alpha*%, and averages — temporally coherent paths.

    Parameters
    ----------
    gmt_array
        Shape ``(n_runs, n_years)``. Must be 2D.
    dataset_path
        Path to RIME NetCDF dataset.
    var_name
        Variable name within the dataset.
    alpha
        CVaR level as percentile (0 < alpha < 100).
    sel
        Optional dimension selections passed to :func:`~.rime.predict_rime`.

    Returns
    -------
    np.ndarray
        Shape ``(n_spatial, n_years)``.
    """
    if not 0 < alpha < 100:
        raise ValueError(f"alpha must be between 0 and 100, got {alpha}")
    ensemble = predict_rime(
        gmt_array, dataset_path, var_name, sel=sel, aggregate="none"
    )
    n_runs = ensemble.shape[0]
    cutoff = max(1, int(np.ceil(n_runs * alpha / 100.0)))
    scores = np.mean(ensemble, axis=(1, 2))
    worst_idx = np.argsort(scores)[:cutoff]
    return np.mean(ensemble[worst_idx], axis=0)
