from typing import Optional
import random
from gepa.core.state import GEPAState
from gepa.proposer.reflective_mutation.base import CandidateSelector
from gepa.gepa_utils import idxmax, select_program_candidate_from_pareto_front

class ParetoCandidateSelector(CandidateSelector):
    def __init__(self, rng: Optional[random.Random]):
        if rng is None:
            self.rng = random.Random(0)
        else:
            self.rng = rng

    def select_candidate_idx(self, state: GEPAState) -> int:
        assert len(state.per_program_tracked_scores) == len(state.program_candidates)
        return select_program_candidate_from_pareto_front(
            state.program_at_pareto_front_valset,
            state.per_program_tracked_scores,
            self.rng,
        )

class CurrentBestCandidateSelector(CandidateSelector):
    def __init__(self):
        pass

    def select_candidate_idx(self, state: GEPAState) -> int:
        assert len(state.per_program_tracked_scores) == len(state.program_candidates)
        return idxmax(state.per_program_tracked_scores)