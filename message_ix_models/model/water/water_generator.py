"""Water module technology groupings.

Provides hierarchical technology groupings from water_tech_spec.yaml,
following the same patterns as :mod:`.transport.structure`.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Sequence, Union

import yaml

if TYPE_CHECKING:
    from message_ix_models import ScenarioInfo, Spec


def load_water_spec(spec_path: Union[str, Path] = None) -> dict:
    """Load water technology spec from YAML.

    Parameters
    ----------
    spec_path : str or Path, optional
        Path to water_tech_spec.yaml. If None, loads from default location
        relative to this file.

    Returns
    -------
    dict
        Loaded YAML specification with technology hierarchies.
    """
    if spec_path is None:
        spec_path = Path(__file__).parent / "water_tech_spec.yaml"

    with open(spec_path, "r") as f:
        return yaml.safe_load(f)


def get_water_technology_groups(
    spec: Union["Spec", "ScenarioInfo", dict, None] = None,
) -> dict[str, list[str]]:
    """Subsets of water technologies for aggregation and filtering.

    Flattens hierarchical technology groupings from :file:`water_tech_spec.yaml`
    into a flat mapping of group names to technology lists.

    Parameters
    ----------
    spec : Spec or ScenarioInfo or dict, optional
        Water spec. If a dict, assumed to be loaded YAML. If None, loads from default
        location. Following transport.structure.get_technology_groups() conventions,
        could also accept Spec or ScenarioInfo in future.

    Returns
    -------
    dict[str, list[str]]
        Mapping of group identifiers to lists of technologies. Keys represent
        reporting categories (e.g., "extraction|freshwater|groundwater"), values
        are lists of individual technology IDs (e.g., ["extract_groundwater", "extract_gw_fossil"]).

    Notes
    -----
    Following transport module's hierarchy flattening approach, this function
    recursively traverses the technology spec hierarchy and produces all possible
    parent-child groupings. For example:

      extraction:
        freshwater:
          groundwater:
            renewable:
              - extract_groundwater

    Produces keys:
      - "extraction" -> [all extraction techs]
      - "extraction|freshwater" -> [all freshwater extraction techs]
      - "extraction|freshwater|groundwater" -> [all groundwater extraction techs]
      - "extraction|freshwater|groundwater|renewable" -> ["extract_groundwater"]
    """
    if isinstance(spec, dict):
        spec_dict = spec
    elif spec is None:
        spec_dict = load_water_spec()
    else:
        # Future: accept Spec or ScenarioInfo
        raise NotImplementedError("Spec/ScenarioInfo support not yet implemented")

    result: dict[str, list[str]] = {}

    def _flatten_branch(
        node,
        parent_path: str = "",
        parent_techs: list[str] = None,
    ) -> None:
        """Recursively flatten hierarchy, accumulating technologies."""
        if parent_techs is None:
            parent_techs = []

        if isinstance(node, list):
            # Leaf node: list of technologies
            accumulated = parent_techs + node
            if parent_path:
                result[parent_path] = accumulated

        elif isinstance(node, dict):
            # Branch node: recurse into children
            for key, child in node.items():
                child_path = f"{parent_path}|{key}" if parent_path else key
                _flatten_branch(child, child_path, parent_techs)

    # Process each top-level category
    for category_name, category_spec in spec_dict.items():
        _flatten_branch(category_spec, category_name, [])

    return result


if __name__ == "__main__":
    spec = load_water_spec()
    groups = get_water_technology_groups(spec)

    print("=" * 70)
    print("WATER TECHNOLOGY GROUPS")
    print("=" * 70)
    for path in sorted(groups.keys()):
        techs = groups[path]
        if len(techs) <= 3:
            print(f"{path:50s} -> {techs}")
        else:
            print(f"{path:50s} -> {len(techs)} techs: {techs[:2]}...")
