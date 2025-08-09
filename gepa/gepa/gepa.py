import math
import random
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Protocol, Tuple, TypeVar, Callable

from collections import Counter

from .gepa_utils import (
    GEPAState,
    idxmax,
    select_program_candidate_from_pareto_front,
    log_detailed_metrics_after_discovering_new_program,
    initialize_gepa_state,
    initialize_wandb,
    find_dominator_programs
)
from .merge_programs import (
    sample_and_attempt_merge_programs_by_common_predictors
)

# Generic type aliases matching your original
RolloutOutput = TypeVar('RolloutOutput')
Trajectory = TypeVar('Trajectory')
DataInst = TypeVar('DataInst')

@dataclass
class EvaluationBatch(Generic[Trajectory, RolloutOutput]):
    outputs: List[RolloutOutput]
    scores: List[float]
    trajectories: Optional[List[Trajectory]] = None

class GEPAAdapter(Protocol[DataInst, Trajectory, RolloutOutput]):
    # Evaluate a batch and optionally capture trajectories
    def evaluate(
        self,
        batch: List[DataInst],
        candidate: Dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Trajectory, RolloutOutput]:
        ...

    # Create reflective dataset per predictor name
    def make_reflective_dataset(
        self,
        candidate: Dict[str, str],
        eval_batch: EvaluationBatch[Trajectory, RolloutOutput],
        predictors_to_update: List[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        ...

    # Propose new instruction text per predictor name
    def propose_new_texts(
        self,
        candidate: Dict[str, str],
        reflective_dataset: Dict[str, List[Dict[str, Any]]],
        predictors_to_update: List[str],
    ) -> Dict[str, str]:
        ...

# =========================
# Protocols and Data Models
# =========================

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


class CandidateSelector(Protocol):
    def select_candidate_idx(self, state: GEPAState) -> int:
        ...


class ModuleSelector(Protocol):
    def select_modules(
        self,
        state: GEPAState,
        trajectories: List[Trajectory],
        subsample_scores: List[float],
        candidate_idx: int,
        candidate: Dict[str, str],
    ) -> List[str]:
        ...


class BatchSampler(Protocol):
    def next_minibatch_indices(self, trainset_size: int, iteration: int) -> List[int]:
        ...

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

class RoundRobinModuleSelector(ModuleSelector):
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

class EpochShuffledBatchSampler(BatchSampler):
    """
    Mirrors the original batching logic:
    - Shuffle ids each epoch
    - Pad to minibatch size with least frequent ids
    - Deterministic via state.rng1
    """
    def __init__(self, minibatch_size: int, rng: Optional[random.Random] = None):
        self.minibatch_size = minibatch_size
        self.shuffled_ids: List[int] = []
        self.epoch = -1
        self.id_freqs = Counter()
        if rng is None:
            self.rng = random.Random(0)
        else:
            self.rng = rng

    def _update_shuffled(self, trainset_size: int):
        self.shuffled_ids = list(range(trainset_size))
        self.rng.shuffle(self.shuffled_ids)
        for i in self.shuffled_ids:
            self.id_freqs[i] += 1

        mod = trainset_size % self.minibatch_size
        num_to_pad = (self.minibatch_size - mod) if mod != 0 else 0
        if num_to_pad > 0:
            for _ in range(num_to_pad):
                selected_id = self.id_freqs.most_common()[::-1][0][0]
                self.shuffled_ids.append(selected_id)
                self.id_freqs[selected_id] += 1

    def next_minibatch_indices(self, trainset_size: int, iteration: int) -> List[int]:
        base_idx = iteration * self.minibatch_size
        curr_epoch = 0 if self.epoch == -1 else base_idx // max(len(self.shuffled_ids), 1)
        if curr_epoch > self.epoch:
            self.epoch = curr_epoch
            self._update_shuffled(trainset_size)

        assert len(self.shuffled_ids) >= self.minibatch_size
        assert len(self.shuffled_ids) % self.minibatch_size == 0

        base_idx = base_idx % len(self.shuffled_ids)
        end_idx = base_idx + self.minibatch_size
        assert end_idx <= len(self.shuffled_ids)
        return self.shuffled_ids[base_idx:end_idx]


# =========================
# Proposers
# =========================

class ReflectiveMutationProposer(ProposeNewCandidate):
    """
    Implements current reflective mutation flow:
    - Select candidate via selector
    - Select minibatch via sampler
    - capture_traces_and_eval -> trajectories, subsample_scores
    - skip if all scores==perfect and skip_perfect_score
    - reflection + mutate -> new candidate
    - evaluate new candidate on same minibatch -> new_subsample_scores
    - Return proposal if improved; else None
    """
    def __init__(
        self,
        logger: Any,
        trainset: List[DataInst],
        evaluator: Callable[[List[DataInst], Dict[str, str]], Tuple[List[RolloutOutput], List[float]]],
        capture_traces_and_eval: Callable[[List[DataInst], Dict[str, str]], Tuple[List[Trajectory], List[float]]],
        extract_reflection_content_from_trajectories: Callable[[Dict[str, str], List[Trajectory], List[float], List[str]], Dict[str, List[Dict[str, str]]]],
        reflect_and_propose_new_text_candidate: Callable[[Dict[str, str], Dict[str, List[Dict[str, str]]], List[str]], Dict[str, str]],
        candidate_selector: CandidateSelector,
        module_selector: ModuleSelector,
        batch_sampler: BatchSampler,
        perfect_score: float,
        skip_perfect_score: bool,
        use_wandb: bool,
    ):
        self.logger = logger
        self.trainset = trainset
        self.evaluator = evaluator
        self.capture_traces_and_eval = capture_traces_and_eval
        self.extract_reflection_content_from_trajectories = extract_reflection_content_from_trajectories
        self.reflect_and_propose_new_text_candidate = reflect_and_propose_new_text_candidate
        self.candidate_selector = candidate_selector
        self.module_selector = module_selector
        self.batch_sampler = batch_sampler
        self.perfect_score = perfect_score
        self.skip_perfect_score = skip_perfect_score
        self.use_wandb = use_wandb

    def propose(self, state: GEPAState) -> Optional[CandidateProposal]:
        i = state.i + 1
        # Select candidate
        curr_prog_id = self.candidate_selector.select_candidate_idx(state)
        curr_prog = state.program_candidates[curr_prog_id]
        state.full_program_trace[-1]['selected_program_candidate'] = curr_prog_id
        self.logger.log(f"Iteration {i}: Selected program candidate {curr_prog_id} with base score: {state.per_program_tracked_scores[curr_prog_id]}")

        if self.use_wandb:
            import wandb  # type: ignore
            wandb.log({
                "iteration": i,
                "selected_program_candidate": curr_prog_id,
            }, step=i)

        # Select minibatch
        subsample_ids = self.batch_sampler.next_minibatch_indices(len(self.trainset), i-1)
        state.full_program_trace[-1]['subsample_ids'] = subsample_ids

        # Capture trajectories and scores
        trajectories, subsample_scores = self.capture_traces_and_eval([self.trainset[j] for j in subsample_ids], curr_prog)
        assert len(trajectories) == len(subsample_scores), "Trajectories and subsample scores must have the same length"
        if len(trajectories) == 0:
            self.logger.log(f"Iteration {i}: No trajectories captured for current program {curr_prog_id}. Skipping reflective mutation.")
            return None

        if self.skip_perfect_score and all(s >= self.perfect_score for s in subsample_scores):
            self.logger.log(f"Iteration {i}: All scores are perfect for current program {curr_prog_id}. Skipping reflective mutation.")
            return None

        if self.use_wandb:
            import wandb  # type: ignore
            wandb.log({"subsample_score": sum(subsample_scores)}, step=i)
        
        state.total_num_evals += len(subsample_ids)

        state.full_program_trace[-1]['subsample_scores'] = subsample_scores

        # Select module(s) to update
        predictor_names_to_update = self.module_selector.select_modules(
            state, trajectories, subsample_scores, curr_prog_id, curr_prog
        )

        # Build reflective dataset
        try:
            reflective_dataset = self.extract_reflection_content_from_trajectories(
                curr_prog,
                trajectories,
                subsample_scores,
                predictor_names_to_update
            )
            new_texts = self.reflect_and_propose_new_text_candidate(
                curr_prog,
                reflective_dataset,
                predictor_names_to_update,
            )
            for pname, text in new_texts.items():
                self.logger.log(f"Iteration {i}: Proposed new text for {pname}: {text}")
            
            if self.use_wandb:
                import wandb  # type: ignore
                wandb.log({f"new_instruction_{pname}": text for pname, text in new_texts.items()}, step=i)
        except Exception as e:
            self.logger.log(f"Iteration {i}: Exception during reflection/proposal: {e}")
            self.logger.log(traceback.format_exc())
            return None

        # Create new candidate
        new_candidate = curr_prog.copy()
        for pname, text in new_texts.items():
            assert pname in new_candidate, f"Predictor {pname} not found in current program"
            new_candidate[pname] = text

        # Evaluate new candidate on the same minibatch
        subsample_evaluator = lambda prog: self.evaluator([self.trainset[k] for k in subsample_ids], prog)
        _, new_subsample_scores = subsample_evaluator(new_candidate)
        state.total_num_evals += len(subsample_ids)
        state.full_program_trace[-1]['new_subsample_scores'] = new_subsample_scores

        new_sum = sum(new_subsample_scores)
        old_sum = sum(subsample_scores)
        self.logger.log(f"Iteration {i}: New subsample score: {new_sum}")
        if self.use_wandb:
            import wandb  # type: ignore
            wandb.log({"new_subsample_score": new_sum}, step=i)

        # Acceptance is evaluated by engine; we just return the proposal
        return CandidateProposal(
            candidate=new_candidate,
            parent_program_ids=[curr_prog_id],
            subsample_indices=subsample_ids,
            subsample_scores_before=subsample_scores,
            subsample_scores_after=new_subsample_scores,
            tag="reflective_mutation",
        )

class MergeProposer(ProposeNewCandidate):
    """
    Implements current merge flow:
    - Find merge candidates among Pareto front dominators
    - Attempt a merge via sample_and_attempt_merge_programs_by_common_predictors
    - Subsample eval on valset-driven selected indices
    - Return proposal if merge's subsample score >= max(parents)
    The engine handles full eval + adding to state.
    """
    def __init__(
        self,
        logger: Any,
        valset: List[DataInst],
        evaluator: Callable[[List[DataInst], Dict[str, str]], Tuple[List[RolloutOutput], List[float]]],
        use_merge: bool,
        max_merge_invocations: int,
        rng: Optional[random.Random] = None,
    ):
        self.logger = logger
        self.valset = valset
        self.evaluator = evaluator
        self.use_merge = use_merge
        self.max_merge_invocations = max_merge_invocations
        if rng is None:
            self.rng = random.Random(0)
        else:
            self.rng = rng

        # Internal counters matching original behavior
        self.merges_due = 0
        self.total_merges_tested = 0
        self.merges_performed: Tuple[List[Tuple[int, int, int]], Any] = ([], [])

        # Toggle controlled by engine: set True when last iter found new program
        self.last_iter_found_new_program = False

    def schedule_if_needed(self):
        if self.use_merge and self.total_merges_tested < self.max_merge_invocations:
            self.merges_due += 1

    def select_eval_subsample_for_merged_program(
        self,
        scores1: List[float],
        scores2: List[float],
        num_subsample_ids: int = 5,
    ) -> List[int]:
        all_indices = set(range(len(scores1)))
        p1 = [i for i, (s1, s2) in enumerate(zip(scores1, scores2)) if s1 > s2]
        p2 = [i for i, (s1, s2) in enumerate(zip(scores1, scores2)) if s2 > s1]
        p3 = [i for i in all_indices if i not in p1 and i not in p2]

        n_each = math.ceil(num_subsample_ids / 3)
        n1 = min(len(p1), n_each)
        n2 = min(len(p2), n_each)
        n3 = min(len(p3), num_subsample_ids - (n1 + n2))
        selected = []
        if n1: selected += self.rng.sample(p1, k=n1)
        if n2: selected += self.rng.sample(p2, k=n2)
        if n3: selected += self.rng.sample(p3, k=n3)

        remaining = num_subsample_ids - len(selected)
        unused = list(all_indices - set(selected))
        if remaining > 0:
            if len(unused) >= remaining:
                selected += self.rng.sample(unused, k=remaining)
            else:
                selected += self.rng.choices(list(all_indices), k=remaining)
        return selected[:num_subsample_ids]

    def propose(self, state: GEPAState) -> Optional[CandidateProposal]:
        i = state.i + 1
        state.full_program_trace[-1]['invoked_merge'] = True

        # Only attempt when scheduled by engine and after a new program in last iteration
        if not (self.use_merge and self.last_iter_found_new_program and self.merges_due > 0):
            self.logger.log(f"Iteration {i}: No merge candidates scheduled")
            return None

        pareto_front_programs = state.program_at_pareto_front_valset
        merge_candidates = find_dominator_programs(pareto_front_programs, state.per_program_tracked_scores)
        merge_output = sample_and_attempt_merge_programs_by_common_predictors(
            agg_scores=state.per_program_tracked_scores,
            rng=self.rng,
            merge_candidates=merge_candidates,
            merges_performed=self.merges_performed,
            program_candidates=state.program_candidates,
            parent_program_for_candidate=state.parent_program_for_candidate,
        )

        if not merge_output[0]:
            self.logger.log(f"Iteration {i}: No merge candidates found")
            return None

        # success, new_program, id1, id2, ancestor
        success, new_program, id1, id2, ancestor = merge_output
        state.full_program_trace[-1]['merged'] = True
        state.full_program_trace[-1]['merged_entities'] = (id1, id2, ancestor)
        self.merges_performed[0].append((id1, id2, ancestor))
        self.logger.log(f"Iteration {i}: Merged programs {id1} and {id2} via ancestor {ancestor}")

        subsample_ids = self.select_eval_subsample_for_merged_program(
            state.prog_candidate_val_subscores[id1],
            state.prog_candidate_val_subscores[id2],
        )
        mini_devset = [self.valset[k] for k in subsample_ids]
        id1_sub_scores = [state.prog_candidate_val_subscores[id1][k] for k in subsample_ids]
        id2_sub_scores = [state.prog_candidate_val_subscores[id2][k] for k in subsample_ids]
        state.full_program_trace[-1]['subsample_ids'] = subsample_ids

        _, new_sub_scores = self.evaluator(mini_devset, new_program)

        state.full_program_trace[-1]['id1_subsample_scores'] = id1_sub_scores
        state.full_program_trace[-1]['id2_subsample_scores'] = id2_sub_scores
        state.full_program_trace[-1]['new_program_subsample_scores'] = new_sub_scores

        # Count evals
        state.total_num_evals += len(subsample_ids)

        # Acceptance will be evaluated by engine (>= max(parents))
        return CandidateProposal(
            candidate=new_program,
            parent_program_ids=[id1, id2],
            subsample_indices=subsample_ids,
            subsample_scores_before=[sum(id1_sub_scores), sum(id2_sub_scores)],  # packed as [parent1_sum, parent2_sum]
            subsample_scores_after=new_sub_scores,
            tag="merge",
            metadata={"ancestor": ancestor}
        )


# =========================
# Engine
# =========================

class GEPAEngine(Generic[DataInst, Trajectory, RolloutOutput]):
    """
    Orchestrates the optimization loop. It uses pluggable ProposeNewCandidate strategies.
    """
    def __init__(
        self,
        logger: Any,
        run_dir: str,
        evaluator: Callable[[List[DataInst], Dict[str, str]], Tuple[List[RolloutOutput], List[float]]],
        valset: Optional[List[DataInst]],
        base_program: Dict[str, str],
        # Controls
        num_iters: Optional[int],
        max_metric_calls: Optional[int],
        perfect_score: float,
        use_wandb: bool,
        wandb_api_key: Optional[str],
        seed: int,
        # Strategies and helpers
        reflective_proposer: ReflectiveMutationProposer,
        merge_proposer: Optional[MergeProposer],
    ):
        # Budget constraint: exactly one of max_metric_calls or num_iters must be set
        assert (max_metric_calls is not None) + (num_iters is not None) == 1, \
            f"Exactly one of max_metric_calls or num_iters should be set. You set max_metric_calls={max_metric_calls}, num_iters={num_iters}"

        self.logger = logger
        self.run_dir = run_dir
        self.evaluator = evaluator
        self.valset = valset
        self.base_program = base_program

        self.num_iters = num_iters
        self.max_metric_calls = max_metric_calls

        self.perfect_score = perfect_score
        self.use_wandb = use_wandb
        self.wandb_api_key = wandb_api_key
        self.seed = seed

        self.reflective_proposer = reflective_proposer
        self.merge_proposer = merge_proposer

        # Merge scheduling flags (mirroring previous behavior)
        if self.merge_proposer is not None:
            self.merge_proposer.last_iter_found_new_program = False

    def _val_evaluator(self) -> Callable[[Dict[str, str]], Tuple[List[RolloutOutput], List[float]]]:
        assert self.valset is not None
        return lambda prog: self.evaluator(self.valset, prog)

    def _get_pareto_front_programs(self, state: GEPAState) -> List:
        return state.program_at_pareto_front_valset

    def _run_full_eval_and_add(
        self,
        new_program: Dict[str, str],
        state: GEPAState,
        parent_program_idx: List[int],
    ) -> Tuple[int, int]:
        num_metric_calls_by_discovery = state.total_num_evals

        valset_outputs, valset_subscores = self._val_evaluator()(new_program)
        valset_score = sum(valset_subscores) / len(valset_subscores)

        state.num_full_ds_evals += 1
        state.total_num_evals += len(valset_subscores)

        new_program_idx, linear_pareto_front_program_idx = state.update_state_with_new_program(
            parent_program_idx=parent_program_idx,
            new_program=new_program,
            valset_score=valset_score,
            valset_outputs=valset_outputs,
            valset_subscores=valset_subscores,
            run_dir=self.run_dir,
            num_metric_calls_by_discovery_of_new_program=num_metric_calls_by_discovery
        )
        state.full_program_trace[-1]['new_program_idx'] = new_program_idx

        if new_program_idx == linear_pareto_front_program_idx:
            self.logger.log(f"Iteration {state.i+1}: New program is on the linear pareto front")

        log_detailed_metrics_after_discovering_new_program(
            logger=self.logger,
            gepa_state=state,
            valset_score=valset_score,
            new_program_idx=new_program_idx,
            valset_subscores=valset_subscores,
            # new_instruction="Merged or Reflective program",
            use_wandb=self.use_wandb,
            linear_pareto_front_program_idx=linear_pareto_front_program_idx,
        )
        return new_program_idx, linear_pareto_front_program_idx

    def run(self) -> GEPAState:
        if self.use_wandb:
            initialize_wandb(wandb_api_key=self.wandb_api_key, run_dir=self.run_dir)

        # Prepare valset
        if self.valset is None:
            raise ValueError("valset must be provided to GEPAEngine.run()")

        # Initialize state (keeps your previous semantics)
        state = initialize_gepa_state(
            run_dir=self.run_dir,
            logger=self.logger,
            base_program=self.base_program,
            valset_evaluator=self._val_evaluator(),
            seed=self.seed,
        )

        assert len(state.pareto_front_valset) == len(self.valset)

        if self.use_wandb:
            import wandb  # type: ignore
            wandb.log({
                "base_program_full_valset_score": state.program_full_scores_val_set[0],
                "iteration": state.i+1,
            })
        self.logger.log(f"Iteration {state.i+1}: Base program full valset score: {state.program_full_scores_val_set[0]}")

        # Merge scheduling
        if self.merge_proposer is not None:
            self.merge_proposer.last_iter_found_new_program = False

        # Main loop
        while (
            (self.num_iters is None or state.num_full_ds_evals < self.num_iters) and
            (self.max_metric_calls is None or state.total_num_evals < self.max_metric_calls)
        ):
            assert state.is_consistent()
            try:
                state.save(self.run_dir)
                state.i += 1
                state.full_program_trace.append({"i": state.i})

                # 1) Attempt merge first if scheduled and last iter found new program
                if self.merge_proposer is not None and self.merge_proposer.use_merge:
                    if self.merge_proposer.merges_due > 0 and self.merge_proposer.last_iter_found_new_program:
                        proposal = self.merge_proposer.propose(state)
                        # Old behavior: clear the flag as soon as we attempt a merge
                        self.merge_proposer.last_iter_found_new_program = False

                        if proposal is not None and proposal.tag == "merge":
                            parent_sums = proposal.subsample_scores_before or [float("-inf"), float("-inf")]
                            new_sum = sum(proposal.subsample_scores_after or [])

                            if new_sum >= max(parent_sums):
                                # ACCEPTED: consume one merge attempt and record it
                                self._run_full_eval_and_add(
                                    new_program=proposal.candidate,
                                    state=state,
                                    parent_program_idx=proposal.parent_program_ids,
                                )
                                self.merge_proposer.merges_due -= 1
                                self.merge_proposer.total_merges_tested += 1
                                # Skip reflective this iteration (old behavior)
                                continue
                            else:
                                # REJECTED: do NOT consume merges_due or total_merges_tested
                                self.logger.log(
                                    f"Iteration {state.i+1}: New program subsample score {new_sum} is worse than both parents {parent_sums}, skipping merge"
                                )
                                # Skip reflective this iteration (old behavior)
                                continue
                    # Old behavior: regardless of whether we attempted, clear the flag before reflective
                    self.merge_proposer.last_iter_found_new_program = False

                # 2) Reflective mutation proposer
                proposal = self.reflective_proposer.propose(state)
                if proposal is None:
                    self.logger.log(f"Iteration {state.i+1}: Reflective mutation did not propose a new candidate")
                    continue

                # Acceptance: require strict improvement on subsample
                old_sum = sum(proposal.subsample_scores_before or [])
                new_sum = sum(proposal.subsample_scores_after or [])
                if new_sum <= old_sum:
                    self.logger.log(f"Iteration {state.i+1}: New subsample score is not better, skipping")
                    continue

                # Accept: full eval + add
                self._run_full_eval_and_add(
                    new_program=proposal.candidate,
                    state=state,
                    parent_program_idx=proposal.parent_program_ids
                )

                # Schedule merge attempts like original behavior
                if self.merge_proposer is not None:
                    self.merge_proposer.last_iter_found_new_program = True
                    if self.merge_proposer.total_merges_tested < self.merge_proposer.max_merge_invocations:
                        self.merge_proposer.merges_due += 1

            except Exception as e:
                self.logger.log(f"Iteration {state.i+1}: Exception during optimization: {e}")
                self.logger.log(traceback.format_exc())
                continue

        state.save(self.run_dir)
        return state

# =========================
# Backward-compatible Wrapper
# =========================

class GEPA(Generic[DataInst, Trajectory, RolloutOutput]):
    """
    Backward compatible wrapper exposing the same public API (.gepa) as your original class.
    Internally builds the new engine and default strategies to preserve behavior.
    """
    def __init__(
        self,
        logger,
        run_dir: str,
        candidate_selection_strategy: str = "pareto",
        num_iters=None,
        perfect_score=1,
        use_wandb: bool = False,
        wandb_api_key: Optional[str] = None,
        seed=0,
        skip_perfect_score=True,
        use_merge=False,
        max_merge_invocations=5,
        num_examples_per_gepa_step=3,
        max_metric_calls=None,
    ):
        assert (max_metric_calls is not None) + (num_iters is not None) == 1, \
            f"Exactly one of max_metric_calls or num_iters should be set. You set max_metric_calls={max_metric_calls}, num_iters={num_iters}"

        self.logger = logger
        self.run_dir = run_dir
        self.candidate_selection_strategy = candidate_selection_strategy

        self.perfect_score = perfect_score
        self.use_wandb = use_wandb
        self.wandb_api_key = wandb_api_key

        self.num_iters = num_iters
        self.max_metric_calls = max_metric_calls

        self.seed = seed
        self.skip_perfect_score = skip_perfect_score
        self.use_merge = use_merge
        self.max_merge_invocations = max_merge_invocations

        self.num_examples_per_gepa_step = num_examples_per_gepa_step

        # Will be set on .gepa()
        self.gepa_state: Optional[GEPAState] = None

    def gepa(
        self,
        base_program: Dict[str, str],
        trainset: List[DataInst],
        eval_and_get_outputs: Callable[[List[DataInst], Dict[str, str]], Tuple[List[RolloutOutput], List[float]]],
        capture_traces_and_eval: Callable[[List[DataInst], Dict[str, str]], Tuple[List[Trajectory], List[float]]],
        extract_reflection_content_from_trajectories: Callable[[Dict[str, str], List[Trajectory], List[float], List[str]], Dict[str, List[Dict[str, str]]]],
        reflect_and_propose_new_text_candidate: Callable[[Dict[str, str], Dict[str, List[Dict[str, str]]], List[str]], Dict[str, str]],
        valset: Optional[List[DataInst]] = None,
    ) -> GEPAState:
        # Default to trainset if valset not given (same as before)
        if valset is None:
            valset = trainset

        # Build default strategies mirroring old behavior
        rng = random.Random(self.seed)
        if self.candidate_selection_strategy == "current_best":
            candidate_selector = CurrentBestCandidateSelector()
        elif self.candidate_selection_strategy == "pareto":
            candidate_selector = ParetoCandidateSelector(rng=rng)
        else:
            raise ValueError(f"Invalid candidate_selection_strategy: {self.candidate_selection_strategy}")

        module_selector = RoundRobinModuleSelector()
        batch_sampler = EpochShuffledBatchSampler(minibatch_size=self.num_examples_per_gepa_step, rng=rng)

        reflective_proposer = ReflectiveMutationProposer(
            logger=self.logger,
            trainset=trainset,
            evaluator=eval_and_get_outputs,
            capture_traces_and_eval=capture_traces_and_eval,
            extract_reflection_content_from_trajectories=extract_reflection_content_from_trajectories,
            reflect_and_propose_new_text_candidate=reflect_and_propose_new_text_candidate,
            candidate_selector=candidate_selector,
            module_selector=module_selector,
            batch_sampler=batch_sampler,
            perfect_score=self.perfect_score,
            skip_perfect_score=self.skip_perfect_score,
            use_wandb=self.use_wandb,
        )

        merge_proposer = None
        if self.use_merge:
            merge_proposer = MergeProposer(
                logger=self.logger,
                valset=valset,
                evaluator=eval_and_get_outputs,
                use_merge=self.use_merge,
                max_merge_invocations=self.max_merge_invocations,
                rng=rng,
            )

        engine = GEPAEngine(
            logger=self.logger,
            run_dir=self.run_dir,
            evaluator=eval_and_get_outputs,
            valset=valset,
            base_program=base_program,
            num_iters=self.num_iters,
            max_metric_calls=self.max_metric_calls,
            perfect_score=self.perfect_score,
            use_wandb=self.use_wandb,
            wandb_api_key=self.wandb_api_key,
            seed=self.seed,
            reflective_proposer=reflective_proposer,
            merge_proposer=merge_proposer,
        )

        state = engine.run()
        self.gepa_state = state
        return state
