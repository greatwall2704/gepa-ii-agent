from typing import List, Dict
from gepa.core.state import GEPAState
from gepa.proposer.reflective_mutation.base import ReflectionComponentSelector
from gepa.core.adapter import Trajectory

class RoundRobinReflectionComponentSelector(ReflectionComponentSelector):
    def select_modules(
        self,
        state: GEPAState,
        trajectories: List[Trajectory],
        subsample_scores: List[float],
        candidate_idx: int,
        candidate: Dict[str, str],
    ) -> List[str]:
        pid = state.named_predictor_id_to_update_next_for_program_candidate[candidate_idx]
        state.named_predictor_id_to_update_next_for_program_candidate[candidate_idx] = (
            pid + 1
        ) % len(state.list_of_named_predictors)
        name = state.list_of_named_predictors[pid]
        return [name]
