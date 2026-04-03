"""Command-line tools specific to the SPARRCLE project."""

from pathlib import Path

import click

from message_ix_models.util.click import common_params


@click.group("sparrcle")
@click.pass_obj
def cli(context):
    """SPARRCLE project."""


@cli.command("build")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("message_ix_models/project/sparrcle/scenario_config.yaml"),
    show_default=True,
    help="Path to the SPARRCLE scenario configuration YAML.",
)
@click.option(
    "--ssp",
    "ssp_filter",
    type=click.Choice(["SSP2", "SSP3"]),
    help="Restrict the build to one SSP starter from the YAML config.",
)
@click.option(
    "--step",
    type=click.Choice(["cooling", "nexus", "all"]),
    default="all",
    show_default=True,
    help="Build only cooling, only nexus, or both in sequence.",
)
@common_params("dry_run")
@click.pass_obj
def build_cmd(context, config_path: Path, ssp_filter: str | None, step: str, dry_run: bool):
    """Build SPARRCLE cooling and water/nexus module scenarios."""
    from message_ix_models.project.sparrcle.scenario_generator import build_all, load_config

    config = load_config(config_path)
    build_all(config=config, ssp_filter=ssp_filter, step=step, dry_run=dry_run)
