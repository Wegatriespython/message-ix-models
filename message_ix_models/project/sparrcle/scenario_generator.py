"""YAML-driven scenario generation for SPARRCLE.

Two phases, both driven by scenario_config.yaml:

Phase 1 — Module builds:
  Clone starter → build cooling → build water/nexus (reduced basins).
  Produces scenarios with the structural modules in place.

Phase 2 — CID pipeline:
  Load MAGICC GMT ensemble → apply buildings → cooling → water CIDs.
  Delegates to _staging/cid_pipeline.py for the actual CID application.

Usage (Python)::

    from message_ix_models.project.sparrcle.scenario_generator import (
        load_config, build_scenario, run_cid,
    )
    from pathlib import Path

    cfg = load_config(Path("message_ix_models/project/sparrcle/scenario_config.yaml"))
    # Phase 1: module builds
    build_scenario(mp, "SSP_SSP2_v6.5_sp", "NPiREF_CI_0", "SSP2", cfg)
    # Phase 2: CID pipeline
    run_cid(mp, "SSP_SSP2_v6.5_sp", "NPiREF_CI_0", cfg, ssp="SSP2")
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

import yaml
from ixmp import Platform

from message_ix_models.tools.impacts import ReductionMode

log = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load and validate SPARRCLE scenario config from YAML."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    required = ["platform_info", "starters", "output", "cooling", "nexus"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    return config


# Phase 1 cooling/nexus module builds delegate to the upstream
# `mix-models water-ix` CLI as a subprocess. The previous in-process
# wrappers (_build_cooling_module / _build_nexus_module) called
# `context.set_scenario(clone)` before invoking the water module's
# cooling/nexus functions, which made the inner `add_water_supply`
# (and similar) call `context.get_scenario()` against the freshly
# cloned and currently-locked scenario — undefined-behaviour deadlock.
#
# The CLI path avoids this because `--url` keeps `scenario_info`
# pointing at the original starter; the cooling/nexus call clones
# internally and the inner read-back lands on the unlocked starter.
# See ticket #337 activity for the original diagnosis.


def _run_water_ix(
    starter_model: str,
    starter_scenario: str,
    subcommand: str,
    *,
    regions: str,
    ssp: str,
    rcps: str,
    rels: str,
    sdgs: Optional[str] = None,
    platform: str = "ixmp_dev",
) -> None:
    """Invoke `mix-models --url=ixmp://... water-ix <subcommand>` per starter."""
    url = f"ixmp://{platform}/{starter_model}/{starter_scenario}"
    cmd = [
        "mix-models",
        "--url",
        url,
        "water-ix",
        subcommand,
        "--regions",
        regions,
        "--ssp",
        ssp,
        "--rcps",
        rcps,
        "--rels",
        rels,
    ]
    if subcommand == "nexus" and sdgs is not None:
        cmd += ["--sdgs", sdgs]
    log.info("→ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_scenario(
    mp: Platform,
    starter_model: str,
    starter_scenario: str,
    ssp: str,
    config: dict,
    step: str = "all",
    dry_run: bool = False,
) -> None:
    """Build cooling and/or nexus modules for a single starter scenario.

    Each step is delegated to a `mix-models --url=ixmp://... water-ix
    <subcommand>` subprocess. The subprocess invocation is the only path
    that keeps `context.scenario_info` pointing at the original starter,
    avoiding the inner-handle deadlock documented in #337.
    """
    cooling_config = config["cooling"]
    nexus_config = config["nexus"]
    platform = config["platform_info"]["name"]
    regions = config.get("regions", "R12")

    log.info("=" * 60)
    log.info(f"SPARRCLE module build: {starter_model}/{starter_scenario} ({ssp})")
    log.info(f"Step: {step}")
    log.info("=" * 60)

    if dry_run:
        log.info("[DRY RUN] Validation passed")
        return

    if step in ("cooling", "all"):
        log.info("Cooling: invoking mix-models water-ix cooling")
        _run_water_ix(
            starter_model,
            starter_scenario,
            "cooling",
            regions=regions,
            ssp=ssp,
            rcps=cooling_config.get("rcps", "no_climate"),
            rels=cooling_config.get("rels", "low"),
            platform=platform,
        )

    if step in ("nexus", "all"):
        log.info("Nexus: invoking mix-models water-ix nexus")
        _run_water_ix(
            starter_model,
            starter_scenario,
            "nexus",
            regions=regions,
            ssp=ssp,
            rcps=nexus_config.get("rcps", "no_climate"),
            rels=nexus_config.get("rels", "low"),
            sdgs=nexus_config.get("sdgs", "baseline"),
            platform=platform,
        )


def build_all(
    config: dict,
    ssp_filter: Optional[str] = None,
    step: str = "all",
    dry_run: bool = False,
) -> list:
    """Build modules for all starters in the config.

    Parameters
    ----------
    config : dict
        From load_config().
    ssp_filter : str, optional
        Restrict to a single SSP (e.g. "SSP2").
    step : str
        "cooling", "nexus", or "all".
    dry_run : bool
        Validate only.

    Returns
    -------
    list[dict]
        One result dict per starter processed.
    """
    platform_name = config["platform_info"]["name"]
    jvmargs = config["platform_info"].get("jvmargs")

    starters = config["starters"]
    if ssp_filter:
        starters = [s for s in starters if s["ssp"] == ssp_filter]

    log.info("=" * 60)
    log.info("SPARRCLE MODULE BUILD")
    log.info("=" * 60)
    log.info(f"Platform: {platform_name}")
    log.info(f"SSP filter: {ssp_filter or 'all'}")
    log.info(f"Step: {step}")
    log.info(f"Starters: {len(starters)}")
    for s in starters:
        log.info(f"  - {s['model']}/{s['scenario']} ({s['ssp']})")
    log.info(f"Mode: {'DRY RUN' if dry_run else 'BUILD'}")

    if not starters:
        log.warning("No starters match filter")
        return []

    mp = (
        Platform(platform_name, jvmargs=jvmargs) if jvmargs else Platform(platform_name)
    )

    results = []
    for starter_spec in starters:
        result = build_scenario(
            mp=mp,
            starter_model=starter_spec["model"],
            starter_scenario=starter_spec["scenario"],
            ssp=starter_spec["ssp"],
            config=config,
            step=step,
            dry_run=dry_run,
        )
        results.append(result)

    log.info("=" * 60)
    log.info(f"BUILD COMPLETE: {len(results)} starters processed")
    log.info("=" * 60)

    return results


# ---------------------------------------------------------------------------
# Phase 2: CID pipeline
# ---------------------------------------------------------------------------


def run_cid(
    mp: Platform,
    model: str,
    source_scenario: str,
    config: dict,
    ssp: str = "SSP2",
    steps: Optional[list[str]] = None,
) -> dict:
    """Run the CID pipeline for a single starter, using config for paths.

    Looks up the MAGICC output directory from the starter's
    ``magicc_output_dir`` field in the config, then delegates to
    ``_staging.cid_pipeline.run_cid_pipeline``.

    Parameters
    ----------
    mp : Platform
        ixmp platform connection.
    model : str
        Model name on the platform.
    source_scenario : str
        Starting scenario name. Must have at least the cooling module
        built; needs the nexus module too if the water CID step is included.
    config : dict
        Full SPARRCLE config from load_config().
    ssp : str
        SSP label to look up the correct starter entry.
    steps : list[str], optional
        CID steps to run. Defaults to config["cid"]["steps"].

    Returns
    -------
    dict[str, Scenario]
        Mapping from step name to created scenario (from cid_pipeline).
    """
    import sys

    # Find the starter entry for this SSP
    starter_entry = None
    for s in config["starters"]:
        if s["ssp"] == ssp and s["model"] == model:
            starter_entry = s
            break
    if starter_entry is None:
        raise ValueError(f"No starter entry for {model}/{ssp} in config")

    magicc_dir = starter_entry.get("magicc_output_dir")
    if not magicc_dir:
        raise ValueError(
            f"Starter {model}/{source_scenario} has no magicc_output_dir in config"
        )

    cid_config = config.get("cid", {})
    if steps is None:
        steps = cid_config.get("steps", ["buildings", "cooling", "water"])
    n_runs = cid_config.get("n_runs")
    reduction: ReductionMode = cid_config.get("reduction", "mean")

    # Import from _staging — add to path if needed
    staging_dir = Path(__file__).resolve().parents[3] / "_staging"
    if str(staging_dir) not in sys.path:
        sys.path.insert(0, str(staging_dir))

    from cid_pipeline import run_cid_pipeline

    log.info("=" * 60)
    log.info(f"SPARRCLE CID PIPELINE: {model}/{source_scenario} ({ssp})")
    log.info(f"MAGICC dir: {magicc_dir}")
    log.info(f"Steps: {steps}, n_runs: {n_runs or 'all'}, reduction: {reduction}")
    log.info("=" * 60)

    return run_cid_pipeline(
        mp=mp,
        model=model,
        source_scenario=source_scenario,
        magicc_dir=magicc_dir,
        steps=steps,
        n_runs=n_runs,
        reduction=reduction,
    )
