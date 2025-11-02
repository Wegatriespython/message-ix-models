"""Reporting for MESSAGEix-Water using genno.

Provides genno-based reporting infrastructure for water module, following
the same patterns as :mod:`.transport.report`.
"""

import logging
from typing import TYPE_CHECKING

from message_ix import Reporter

from message_ix_models import Context

if TYPE_CHECKING:
    from genno import Computer, Key, KeySeq, MissingKeyError

log = logging.getLogger(__name__)


def aggregate(c: Computer) -> None:
    """Aggregate individual water technologies to reporting categories.

    Uses genno operators to:
    1. Aggregate individual technologies to groups using t::water agg mapping
    2. Apply spatial aggregations if needed
    3. Select and combine results for reporting

    Parameters
    ----------
    c : genno.Computer
        Computer instance with registered water structure.
    """
    from genno.operator import aggregate as aggregate_op

    # For water, aggregate technologies in 'in' and 'out' quantities
    # This mirrors the transport approach in report.py lines 164-181
    for quantity in ["in", "out"]:
        try:
            # Infer keys matching the pattern
            keys = list(c.infer_keys(f"{quantity}:*"))
            if not keys:
                log.debug(f"No keys matching '{quantity}:*'")
                continue

            # For each key, aggregate by technology groups
            for base_key in keys:
                # Create aggregated version using t::water agg mapping
                agg_key = KeySeq(base_key)
                try:
                    # Aggregate individual techs to groups
                    c.add(
                        agg_key[0],
                        aggregate_op,
                        base_key,
                        "t::water agg",
                        keep=False,
                    )
                except (MissingKeyError, KeyError):
                    # Skip if aggregation key doesn't exist or key format incompatible
                    log.debug(f"Could not aggregate {base_key}")
                    continue

        except (MissingKeyError, KeyError) as e:
            log.debug(f"Skipping aggregation for {quantity}: {e}")


def callback(rep: Reporter, context: Context) -> None:
    """Callback for prepare_reporter to add water reporting.

    Registers technology structures and adds aggregation tasks for
    water module reporting.

    Parameters
    ----------
    rep : message_ix.Reporter
        Reporter instance to add tasks to.
    context : message_ix_models.Context
        Context with configuration.
    """
    from .build import add_water_structure

    log.info("Adding water module reporting")

    # Register water technology structures in the reporter's Computer
    add_water_structure(rep)

    # Apply aggregation operations
    aggregate(rep)

    log.info("Water module reporting added")
