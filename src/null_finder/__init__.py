"""Robust magnetic null finder for rectilinear 3D grids (trilinear + Newton)."""

from .core import (
    NULL_TYPE_A,
    NULL_TYPE_AS,
    NULL_TYPE_B,
    NULL_TYPE_BS,
    NULL_TYPE_UNKNOWN,
    POINCARE_DEGENERATE,
    POINCARE_NOT_COMPUTED,
    RESOLUTION_DEGENERATE,
    RESOLUTION_ISOLATED,
    RESOLUTION_MARGINAL,
    RESOLUTION_RESOLVED,
    RESOLUTION_UNRESOLVED,
    find_magnetic_nulls,
    get_candidate_cells,
    null_type_name,
    pad_periodic,
    resolution_flag_name,
    topology_report,
)

__version__ = "0.1.0"

__all__ = [
    "NULL_TYPE_A",
    "NULL_TYPE_AS",
    "NULL_TYPE_B",
    "NULL_TYPE_BS",
    "NULL_TYPE_UNKNOWN",
    "POINCARE_DEGENERATE",
    "POINCARE_NOT_COMPUTED",
    "RESOLUTION_DEGENERATE",
    "RESOLUTION_ISOLATED",
    "RESOLUTION_MARGINAL",
    "RESOLUTION_RESOLVED",
    "RESOLUTION_UNRESOLVED",
    "__version__",
    "find_magnetic_nulls",
    "get_candidate_cells",
    "null_type_name",
    "pad_periodic",
    "resolution_flag_name",
    "topology_report",
]
