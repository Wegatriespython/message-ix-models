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
from pathlib import Path
from typing import Optional

import yaml
from ixmp import Platform
from message_ix import Scenario

from message_ix_models import Context
from message_ix_models.model.water.cli import cooling, nexus, water_ini

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


def _build_cooling_module(
    scen: Scenario,
    rcps: str = "no_climate",
    rels: str = "low",
    regions: str = "R12",
    ssp: str = "SSP2",
) -> Scenario:
    """Build cooling module onto scenario if not already present.

    Checks for freshwater cooling technologies; if absent, builds
    the water/cooling module via the standard water build path.
    """
    existing_cf = scen.par("capacity_factor")
    cooling_techs = existing_cf[
        existing_cf["technology"].str.contains(
            r"__(ot_fresh|cl_fresh)", regex=True, na=False
        )
    ]

    if len(cooling_techs) > 0:
        log.info(
            f"Cooling techs already present: {cooling_techs['technology'].nunique()} types"
        )
        return scen

    log.info("Building cooling module...")

    context = Context.get_instance(-1)
    context.set_scenario(scen)
    context.ssp = ssp
    water_ini(context, regions=regions, time=None)
    cooling(context, regions=regions, rcps=rcps, rels=rels, solve=False, clone=False, scen=scen)
    log.info("Cooling module built successfully")

    return scen


def _build_nexus_module(
    scen: Scenario,
    nexus_config: dict,
    ssp: str = "SSP2",
    regions: str = "R12",
) -> Scenario:
    """Build water/nexus module onto scenario with basin filtering from config.

    Reads reduced_basin, num_basins, basin_selection, and filter_list from
    the nexus section of the YAML config and sets them on Context before
    calling the standard water build path.
    """
    context = Context.get_instance(-1)
    context.set_scenario(scen)
    context.ssp = ssp
    water_ini(context, regions=regions, time=None)
    context.reduced_basin = nexus_config.get("reduced_basin", False)
    if nexus_config.get("num_basins") is not None:
        context.num_basins = nexus_config["num_basins"]
    context.basin_selection = nexus_config.get("basin_selection", "first_k")
    if nexus_config.get("filter_list"):
        context.filter_list = nexus_config["filter_list"]

    log.info(
        f"Building nexus module: reduced_basin={context.reduced_basin}, "
        f"num_basins={getattr(context, 'num_basins', 'default')}, "
        f"filter_list has {len(nexus_config.get('filter_list', []))} entries"
    )

    nexus(
        context,
        regions=regions,
        rcps=nexus_config.get("rcps", "no_climate"),
        sdgs=nexus_config.get("sdgs", "baseline"),
        rels=nexus_config.get("rels", "low"),
        solve=False,
        clone=False,
        scen=scen,
    )
    log.info("Nexus module built successfully")

    return scen


def build_scenario(
    mp: Platform,
    starter_model: str,
    starter_scenario: str,
    ssp: str,
    config: dict,
    step: str = "all",
    dry_run: bool = False,
) -> dict:
    """Build cooling and/or nexus modules for a single starter scenario.

    Parameters
    ----------
    mp : Platform
        ixmp platform connection.
    starter_model, starter_scenario : str
        Source scenario identifiers on the platform.
    ssp : str
        SSP label (SSP2, SSP3) — passed to nexus build for demand data.
    config : dict
        Full SPARRCLE config from load_config().
    step : str
        Which step to run: "cooling", "nexus", or "all" (default).
    dry_run : bool
        If True, validate only — do not clone or build.

    Returns
    -------
    dict
        Keys "cooling" and/or "nexus" → Scenario objects created.
    """
    cooling_suffix = config["output"]["cooling_suffix"]
    nexus_suffix = config["output"]["nexus_suffix"]
    cooling_config = config["cooling"]
    nexus_config = config["nexus"]
    regions = config.get("regions", "R12")

    log.info("=" * 60)
    log.info(f"SPARRCLE module build: {starter_model}/{starter_scenario} ({ssp})")
    log.info(f"Step: {step}")
    log.info("=" * 60)

    if dry_run:
        log.info("[DRY RUN] Validation passed")
        return {}

    result = {}

    # --- Step 1: Cooling ---
    if step in ("cooling", "all"):
        log.info("1. Loading starter scenario...")
        starter = Scenario(mp, starter_model, starter_scenario)
        log.info(f"   Loaded version {starter.version}")

        cooling_scenario_name = starter_scenario + cooling_suffix
        log.info(f"2. Cloning to {starter_model}/{cooling_scenario_name}...")
        scen_cool = starter.clone(
            model=starter_model,
            scenario=cooling_scenario_name,
            keep_solution=False,
        )

        log.info("3. Building cooling module...")
        _build_cooling_module(
            scen_cool,
            rcps=cooling_config.get("rcps", "no_climate"),
            rels=cooling_config.get("rels", "low"),
            regions=regions,
            ssp=ssp,
        )
        scen_cool.set_as_default()
        log.info(f"   Cooling scenario ready: v{scen_cool.version}")
        result["cooling"] = scen_cool

    # --- Step 2: Nexus ---
    if step in ("nexus", "all"):
        # If we just built cooling, use that as the starter for nexus
        if "cooling" in result:
            scen_base = result["cooling"]
        else:
            # Load the expected cooling scenario
            cooling_scenario_name = starter_scenario + cooling_suffix
            log.info(f"Loading existing cooling scenario: {cooling_scenario_name}...")
            scen_base = Scenario(mp, starter_model, cooling_scenario_name)

        nexus_scenario_name = scen_base.scenario + nexus_suffix
        log.info(f"4. Cloning to {starter_model}/{nexus_scenario_name}...")
        scen_nexus = scen_base.clone(
            model=starter_model,
            scenario=nexus_scenario_name,
            keep_solution=False,
        )

        log.info("5. Building nexus module...")
        _build_nexus_module(scen_nexus, nexus_config, ssp=ssp, regions=regions)
        scen_nexus.set_as_default()
        log.info(f"   Nexus scenario ready: v{scen_nexus.version}")
        result["nexus"] = scen_nexus

    return result


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

    # Import from _staging — add to path if needed
    staging_dir = Path(__file__).resolve().parents[3] / "_staging"
    if str(staging_dir) not in sys.path:
        sys.path.insert(0, str(staging_dir))

    from cid_pipeline import run_cid_pipeline

    log.info("=" * 60)
    log.info(f"SPARRCLE CID PIPELINE: {model}/{source_scenario} ({ssp})")
    log.info(f"MAGICC dir: {magicc_dir}")
    log.info(f"Steps: {steps}, n_runs: {n_runs or 'all'}")
    log.info("=" * 60)

    return run_cid_pipeline(
        mp=mp,
        model=model,
        source_scenario=source_scenario,
        magicc_dir=magicc_dir,
        steps=steps,
        n_runs=n_runs,
    )
