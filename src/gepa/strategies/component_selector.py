# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa


from gepa.core.adapter import Trajectory
from gepa.core.state import GEPAState
from gepa.proposer.reflective_mutation.base import ReflectionComponentSelector


def round_robin_reflection_component_selector(
    state: GEPAState,
    trajectories: list[Trajectory],
    subsample_scores: list[float],
    candidate_idx: int,
    candidate: dict[str, str],
) -> list[str]:
    """Select components in round-robin fashion."""
    pid = state.named_predictor_id_to_update_next_for_program_candidate[candidate_idx]
    state.named_predictor_id_to_update_next_for_program_candidate[candidate_idx] = (pid + 1) % len(
        state.list_of_named_predictors
    )
    name = state.list_of_named_predictors[pid]
    return [name]


def all_reflection_component_selector(
    state: GEPAState,
    trajectories: list[Trajectory],
    subsample_scores: list[float],
    candidate_idx: int,
    candidate: dict[str, str],
) -> list[str]:
    """Select all components for modification."""
    return list(candidate.keys())


# Explicit declarations that the functions implement the ReflectionComponentSelector protocol
round_robin_reflection_component_selector: ReflectionComponentSelector = round_robin_reflection_component_selector
all_reflection_component_selector: ReflectionComponentSelector = all_reflection_component_selector
