from typing import Any, Dict, List, Optional, Protocol
from dataclasses import dataclass, field
from gepa.core.state import GEPAState

@dataclass
class CandidateProposal:
    candidate: Dict[str, str]
    parent_program_ids: List[int]
    # Optional mini-batch / subsample info
    subsample_indices: Optional[List[int]] = None
    subsample_scores_before: Optional[List[float]] = None
    subsample_scores_after: Optional[List[float]] = None
    # Free-form metadata for logging/trace
    tag: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProposeNewCandidate(Protocol):
    """
    Strategy that receives the current optimizer state and proposes a new candidate or returns None.
    It may compute subsample evaluations, set trace fields in state, etc.
    The engine will handle acceptance and full eval unless the strategy already did those and encoded in metadata.
    """
    def propose(self, state: GEPAState) -> Optional[CandidateProposal]:
        ...