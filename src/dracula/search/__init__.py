"""Expose the stable, lightweight contracts shared by search consumers.

Only actor-visible information and authoritative symmetry helpers are exported
here. Search controllers require explicit imports from their owning modules.
"""

from dracula.search.information import (
    InformationContractViolation,
    PublicGameHistory,
    PublicPlayedMove,
    PublicRoundRecord,
    SearchInformationState,
    canonical_information_data,
    canonical_information_json,
    information_state_fingerprint,
    information_state_from_engine,
    information_state_from_simulation,
    public_history_from_engine,
)
from dracula.search.symmetry import (
    DestinationSymmetryError,
    DestinationSymmetryGroup,
    destination_symmetry_groups,
)

__all__ = (
    "DestinationSymmetryError",
    "DestinationSymmetryGroup",
    "InformationContractViolation",
    "PublicGameHistory",
    "PublicPlayedMove",
    "PublicRoundRecord",
    "SearchInformationState",
    "canonical_information_data",
    "canonical_information_json",
    "destination_symmetry_groups",
    "information_state_fingerprint",
    "information_state_from_engine",
    "information_state_from_simulation",
    "public_history_from_engine",
)
