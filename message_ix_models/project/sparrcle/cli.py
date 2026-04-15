"""Command-line tools specific to the SPARRCLE project."""

from pathlib import Path

import click


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
@click.option("--dry-run", is_flag=True, help="Validate config and report what would be built; no DB writes.")
@click.pass_obj
def build_cmd(context, config_path: Path, ssp_filter: str | None, step: str, dry_run: bool):
    """Build SPARRCLE cooling and water/nexus module scenarios."""
    from message_ix_models.project.sparrcle.scenario_generator import build_all, load_config

    config = load_config(config_path)
    build_all(config=config, ssp_filter=ssp_filter, step=step, dry_run=dry_run)


@cli.command("report")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for output files. If not set, returns data without saving.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["csv", "xlsx", "parquet"]),
    default="csv",
    show_default=True,
    help="Output file format.",
)
@click.option(
    "--keys",
    multiple=True,
    help="Specific report keys (e.g. cooling_cap, water_act). Repeatable.",
)
@click.pass_obj
def report_cmd(context, output_dir: Path | None, fmt: str, keys: tuple[str, ...]):
    """Run water/cooling genno reporting on a solved SPARRCLE scenario."""
    from message_ix_models.project.sparrcle.report import report_water_nexus

    scenario = context.get_scenario()
    key_list = list(keys) if keys else None

    report_water_nexus(
        scenario,
        output_dir=output_dir,
        keys=key_list,
        format=fmt,
    )
