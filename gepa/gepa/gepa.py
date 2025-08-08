from ast import Set
import math
import random
import traceback
from typing import Any, Dict, Tuple, TypeVar, Union, Callable, Generic

from typing import List, Set
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

class ContinueException(Exception):
    """Exception to indicate that the current iteration should be skipped."""
    pass

RolloutOutput = TypeVar('RolloutOutput')
Trajectory = TypeVar('Trajectory')
DataInst = TypeVar('DataInst')

class GEPA(Generic[DataInst, Trajectory, RolloutOutput]):
    gepa_state: GEPAState
    eval_and_get_outputs: Callable[[List[DataInst], Dict[str, str]], Tuple[List[RolloutOutput], List[float]]]
    capture_traces_and_eval: Callable[[List[DataInst], Dict[str, str]], Tuple[List[Trajectory], List[float]]]
    extract_reflection_content_from_trajectories: Callable[[Dict[str, str], List[Trajectory], List[float], List[str]], Dict[str, List[Dict[str, str]]]]
    reflect_and_propose_new_text_candidate: Callable[[Dict[str, str], Dict[str, List[Dict[str, str]]], List[str]], Dict[str, str]]

    def __init__(
        self,
        logger,
        run_dir: str,
        run_linearized_gepa: bool=False,
        num_iters=None,
        perfect_score=1,
        use_wandb: bool = False,
        wandb_api_key: Union[None, str] = None,
        seed=0,
        skip_perfect_score=True,
        use_merge=False,
        max_merge_invocations=5,
        num_examples_per_gepa_step=3,
        max_metric_calls=None,
    ):
        # Exactly one of max_metric_calls or num_iters should be set
        assert (max_metric_calls is not None) + (num_iters is not None) == 1, "Exactly one of max_metric_calls or num_iters should be set. You set max_metric_calls={}, num_iters={}".format(
            max_metric_calls, num_iters
        )

        self.logger = logger
        self.run_dir = run_dir
        self.run_linearized_gepa = run_linearized_gepa

        self.perfect_score = perfect_score
        self.use_wandb = use_wandb
        self.wandb_api_key = wandb_api_key

        # Run constraints
        self.num_iters = num_iters
        self.max_metric_calls = max_metric_calls

        self.seed = seed
        self.skip_perfect_score = skip_perfect_score
        self.use_merge = use_merge
        self.max_merge_invocations = max_merge_invocations

        self.num_examples_per_gepa_step = num_examples_per_gepa_step

        self.shuffled_trainset_ids = []
        self.epoch = -1
        self.id_freqs = Counter()

    def update_shuffled_trainset(self, original_trainset: List):
        self.shuffled_trainset_ids = list(range(len(original_trainset)))
        self.gepa_state.rng1.shuffle(self.shuffled_trainset_ids)
        for id in self.shuffled_trainset_ids:
            self.id_freqs[id] += 1

        num_to_pad = self.num_examples_per_gepa_step - (len(original_trainset) % self.num_examples_per_gepa_step)
        if num_to_pad > 0:
            # Select ids based on least frequent ids
            for _ in range(num_to_pad):
                selected_id = self.id_freqs.most_common()[::-1][0][0]
                self.shuffled_trainset_ids.append(selected_id)
                self.id_freqs[selected_id] += 1

    def select_training_sample_and_update_shuffled_trainset(
        self,
        original_trainset: List,
        train_step_idx: int,
    ) -> List:
        base_idx = train_step_idx * self.num_examples_per_gepa_step
        if self.epoch == -1:
            curr_epoch = 0
        else:
            curr_epoch = base_idx // len(self.shuffled_trainset_ids)
        if curr_epoch > self.epoch:
            print(f"Updating shuffled trainset for epoch {curr_epoch}...")
            self.epoch = curr_epoch
            self.update_shuffled_trainset(original_trainset)

        assert len(self.shuffled_trainset_ids) >= self.num_examples_per_gepa_step, f"Shuffled trainset length {len(self.shuffled_trainset_ids)} is less than num_dspy_examples_per_grpo_step {self.num_examples_per_gepa_step}"
        assert len(self.shuffled_trainset_ids) % self.num_examples_per_gepa_step == 0, f"Shuffled trainset length {len(self.shuffled_trainset_ids)} is not divisible by num_dspy_examples_per_grpo_step {self.num_examples_per_gepa_step}"

        base_idx = base_idx % len(self.shuffled_trainset_ids)
        end_idx = base_idx + self.num_examples_per_gepa_step
        assert end_idx <= len(self.shuffled_trainset_ids), f"End index {end_idx} is out of bounds for shuffled trainset length {len(self.shuffled_trainset_ids)}"
        selected_ids = self.shuffled_trainset_ids[base_idx:end_idx]
        return selected_ids

    def select_eval_subsample_for_merged_program(
        self,
        scores1: List[float],
        scores2: List[float],
        rng: random.Random,
        num_subsample_ids: int = 5,
    ) -> List[int]:
        all_indices = set(range(len(scores1)))
        # Partitioning
        partition1_ids = [i for i, (s1, s2) in enumerate(zip(scores1, scores2)) if s1 > s2]
        partition2_ids = [i for i, (s1, s2) in enumerate(zip(scores1, scores2)) if s2 > s1]
        partition3_ids = [i for i in all_indices if i not in partition1_ids and i not in partition2_ids]
        # Set up sample sizes
        n_each = math.ceil(num_subsample_ids / 3)
        n1 = min(len(partition1_ids), n_each)
        n2 = min(len(partition2_ids), n_each)
        n3 = min(len(partition3_ids), num_subsample_ids - (n1 + n2))
        # Sample
        # rng = gepa_state.rng1
        selected = []
        if n1: selected += rng.sample(partition1_ids, k=n1)
        if n2: selected += rng.sample(partition2_ids, k=n2)
        if n3: selected += rng.sample(partition3_ids, k=n3)
        # Pad if needed to desired length, without duplicates if possible
        remaining = num_subsample_ids - len(selected)
        unused = list(all_indices - set(selected))
        if remaining > 0:
            if len(unused) >= remaining:
                selected += rng.sample(unused, k=remaining)
            else:
                # All unique exhausted; use replacement
                selected += rng.choices(list(all_indices), k=remaining)
        return selected[:num_subsample_ids]

    def get_pareto_front_programs(self, gepa_state: GEPAState) -> List[Set[int]]:
        # [
        #     # {idxmax(train_val_weighted_agg_scores_for_all_programs)}, # Add best aggregate program
        #     # {idxmax(gepa_state.program_full_scores_val_set)}, # Add best on valset
        # ] # TODO: Think about whether this should be added or not. Make this configurable.
        return gepa_state.program_at_pareto_front_valset

    def select_next_candidate_to_update(self, gepa_state: GEPAState):
        # TODO: Update this method to use pareto front from both train and val sets configurable
        assert len(gepa_state.per_program_tracked_scores) == len(gepa_state.program_candidates)

        if not gepa_state.running_linearized_gepa:
            curr_prog_id = select_program_candidate_from_pareto_front(
                self.get_pareto_front_programs(gepa_state), 
                gepa_state.per_program_tracked_scores,
                gepa_state.rng1,
            )
        else:
            curr_prog_id = idxmax(gepa_state.per_program_tracked_scores)

        return curr_prog_id

    def run_full_eval_add_new_program_to_gepa_tree(
        self,
        new_program: Dict[str, str],
        gepa_state: GEPAState,
        valset_evaluator: Callable[[Dict[str, str]], Tuple[List[RolloutOutput], List[float]]],
        parent_program_idx: List[int]
    ):
        num_metric_calls_by_discovery_of_new_program = gepa_state.total_num_evals

        # Calculate metrics for new program and update gepa state

        valset_outputs, valset_subscores = valset_evaluator(new_program)
        valset_score = sum(valset_subscores) / len(valset_subscores)

        # We have run one full eval of the new program on train set and val set
        gepa_state.num_full_ds_evals += 1
        gepa_state.total_num_evals += len(valset_subscores)

        new_program_idx, linear_pareto_front_program_idx = gepa_state.update_state_with_new_program(
            parent_program_idx=parent_program_idx, # TODO: Handle this better. Mark both parents
            new_program=new_program, # TODO: Make DSPy changes
            valset_score=valset_score,
            valset_outputs=valset_outputs,
            valset_subscores=valset_subscores,
            run_dir=self.run_dir,
            num_metric_calls_by_discovery_of_new_program=num_metric_calls_by_discovery_of_new_program
        )

        gepa_state.full_program_trace[-1]['new_program_idx'] = new_program_idx

        if new_program_idx == linear_pareto_front_program_idx:
            self.logger.log(f"Iteration {gepa_state.i+1}: New program is on the linear pareto front")

        log_detailed_metrics_after_discovering_new_program(
            logger=self.logger,
            gepa_state=gepa_state,
            valset_score=valset_score,
            new_program_idx=new_program_idx,
            valset_subscores=valset_subscores,
            new_instruction="Merged program",
            use_wandb=self.use_wandb,
            linear_pareto_front_program_idx=linear_pareto_front_program_idx,
        )

        return new_program_idx, linear_pareto_front_program_idx
    
    def select_modules_to_update(
        self,
        gepa_state: GEPAState,
        trajectories: List[Trajectory],
        subsample_scores: List[float],
        curr_prog_idx: int,
        curr_prog: Dict[str, str],
    ) -> List[str]:
        # Select the module to update based on the trajectories and subsample scores

        # Currently this just performs round robin selection of the next module to update
        predictor_to_update_id = gepa_state.named_predictor_id_to_update_next_for_program_candidate[curr_prog_idx]
        gepa_state.named_predictor_id_to_update_next_for_program_candidate[curr_prog_idx] = (predictor_to_update_id + 1) % len(gepa_state.list_of_named_predictors)
        predictor_name_to_update = gepa_state.list_of_named_predictors[predictor_to_update_id]
        return [predictor_name_to_update]

    def reflective_mutation(
        self,
        curr_prog_idx: int,
        curr_prog: Dict[str, str],
        gepa_state: GEPAState,
        minibatch: List,
    ):
        trajectories, subsample_scores = self.capture_traces_and_eval(minibatch, curr_prog)
        assert len(trajectories) == len(subsample_scores), "Trajectories and subsample scores must have the same length"
        if len(trajectories) == 0:
            self.logger.log(f"Iteration {gepa_state.i+1}: No trajectories captured for current program {curr_prog_idx}. Skipping reflective mutation.")
            raise ContinueException("No trajectories captured for current program. Skipping reflective mutation.")

        if self.skip_perfect_score and all(score >= self.perfect_score for score in subsample_scores):
            self.logger.log(f"Iteration {gepa_state.i+1}: All scores are perfect for current program {curr_prog_idx}. Skipping reflective mutation.")
            raise ContinueException("All scores are perfect for current program. Skipping reflective mutation.")

        if self.use_wandb:
            import wandb # type: ignore
            wandb.log({
                "subsample_score": sum(subsample_scores),
            }, step=gepa_state.i+1)

        gepa_state.full_program_trace[-1]['subsample_scores'] = subsample_scores
        predictor_names_to_update = self.select_modules_to_update(gepa_state, trajectories, subsample_scores, curr_prog_idx, curr_prog)
        reflective_dataset = self.extract_reflection_content_from_trajectories(
            curr_prog,
            trajectories,
            subsample_scores,
            predictor_names_to_update
        )
        try:
            new_texts = self.reflect_and_propose_new_text_candidate(
                curr_prog,
                reflective_dataset,
                predictor_names_to_update,
            )

            for predictor_name, new_text in new_texts.items():
                self.logger.log(f"Iteration {gepa_state.i+1}: Proposed new text for {predictor_name}: {new_text}")
        except Exception as e:
            self.logger.log(f"Iteration {gepa_state.i+1}: Exception during reflection and proposal of new text: {e}")
            self.logger.log(traceback.format_exc())
            raise ContinueException("Reflection and proposal of new text failed, skipping reflective mutation")

        new_candidate = curr_prog.copy()
        for predictor_name, new_text in new_texts.items():
            assert predictor_name in new_candidate, f"Predictor {predictor_name} not found in current program"
            new_candidate[predictor_name] = new_text
        return new_candidate, subsample_scores

    def attempt_merge(
        self,
        gepa_state: GEPAState,
        valset: List,
        eval_and_get_outputs: Callable[[List, Dict[str, str]], Tuple[List[RolloutOutput], List[float]]],
        valset_evaluator: Callable[[Dict[str, str]], Tuple[List[RolloutOutput], List[float]]],
        merges_performed: Tuple[List[Tuple[int, int, int]], Any],
        merges_due: int,
        total_merges_tested: int,
    ):
        gepa_state.full_program_trace[-1]['invoked_merge'] = True

        pareto_front_programs = self.get_pareto_front_programs(gepa_state)

        merge_candidates = find_dominator_programs(pareto_front_programs, gepa_state.per_program_tracked_scores)
        merge_output = sample_and_attempt_merge_programs_by_common_predictors(
            agg_scores=gepa_state.per_program_tracked_scores,
            rng=gepa_state.rng1,
            merge_candidates=merge_candidates,
            merges_performed=merges_performed,
            program_candidates=gepa_state.program_candidates,
            parent_program_for_candidate=gepa_state.parent_program_for_candidate,
        )

        if merge_output[0]:
            gepa_state.full_program_trace[-1]['merged'] = True
            success, new_program, id1, id2, ancestor = merge_output
            assert success, "Merge output should be successful"
            self.logger.log(f"Iteration {gepa_state.i+1}: Merged programs {id1} and {id2} via ancestor {ancestor}")
            gepa_state.full_program_trace[-1]['merged_entities'] = (id1, id2, ancestor)

            merges_performed[0].append((id1, id2, ancestor))

            mini_devset = None

            subsample_ids = self.select_eval_subsample_for_merged_program(
                gepa_state.prog_candidate_val_subscores[id1],
                gepa_state.prog_candidate_val_subscores[id2],
                gepa_state.rng1,
            )
            mini_devset = [valset[i] for i in subsample_ids]
            id1_subsample_scores = [gepa_state.prog_candidate_val_subscores[id1][i] for i in subsample_ids]
            id2_subsample_scores = [gepa_state.prog_candidate_val_subscores[id2][i] for i in subsample_ids]

            gepa_state.full_program_trace[-1]['subsample_ids'] = subsample_ids

            subsample_evaluator = lambda prog: eval_and_get_outputs(mini_devset, prog)

            new_program_subsample_scores = subsample_evaluator(new_program)[1]

            id1_subsample_score = sum(id1_subsample_scores)
            id2_subsample_score = sum(id2_subsample_scores)
            new_subsample_score = sum(new_program_subsample_scores)

            gepa_state.full_program_trace[-1]['id1_subsample_scores'] = id1_subsample_scores
            gepa_state.full_program_trace[-1]['id2_subsample_scores'] = id2_subsample_scores
            gepa_state.full_program_trace[-1]['new_program_subsample_scores'] = new_program_subsample_scores

            gepa_state.total_num_evals += len(subsample_ids)

            if new_subsample_score >= max(id1_subsample_score, id2_subsample_score):
                self.logger.log(f"Iteration {gepa_state.i+1}: New program subsample score {new_subsample_score} for merged program is better than min of both parents {id1_subsample_score} and {id2_subsample_score}. proceeding with full eval")
            else:
                self.logger.log(f"Iteration {gepa_state.i+1}: New program subsample score {new_subsample_score} is worse than both parent programs {id1_subsample_score} and {id2_subsample_score}, skipping merge")
                raise ContinueException(merges_due, total_merges_tested)

            merges_due -= 1
            total_merges_tested += 1

            self.run_full_eval_add_new_program_to_gepa_tree(
                new_program=new_program,
                gepa_state=gepa_state,
                valset_evaluator=valset_evaluator,
                parent_program_idx=[id1, id2]
            )
            raise ContinueException(merges_due, total_merges_tested)
        else:
            self.logger.log(f"Iteration {gepa_state.i+1}: No merge candidates found")
            return merges_due, total_merges_tested

    def gepa(
        self,
        base_program: Dict[str, str], 
        trainset: List, 
        # This function is called with list of inputs (from train/val set), and a proposed new program, and returns a tuple of aggregate score, point wise outputs, and point wise scores
        eval_and_get_outputs: Callable[[List, Dict[str, str]], Tuple[List[RolloutOutput], List[float]]],
        capture_traces_and_eval: Callable[[List[DataInst], Dict[str, str]], Tuple[List[Trajectory], List[float]]],
        extract_reflection_content_from_trajectories: Callable,
        reflect_and_propose_new_text_candidate: Callable[[Dict[str, str], Dict[str, List[Dict[str, str]]], List[str]], Dict[str, str]],
        valset: Union[List, None] = None,
        gepa_state_to_use: Union[GEPAState, None]=None,
    ):
        self.eval_and_get_outputs = eval_and_get_outputs
        self.capture_traces_and_eval = capture_traces_and_eval
        self.extract_reflection_content_from_trajectories = extract_reflection_content_from_trajectories
        self.reflect_and_propose_new_text_candidate = reflect_and_propose_new_text_candidate
        if self.use_wandb:
            initialize_wandb(wandb_api_key=self.wandb_api_key, run_dir=self.run_dir)

        if valset is None:
            valset = trainset

        valset_evaluator = lambda prog: self.eval_and_get_outputs(valset, prog)

        gepa_state = initialize_gepa_state(
            gepa_state_to_use=gepa_state_to_use,
            run_dir=self.run_dir,
            logger=self.logger,
            base_program=base_program,
            valset_evaluator=valset_evaluator,
            seed=self.seed,
            run_linearized_gepa=self.run_linearized_gepa,
        )

        self.gepa_state = gepa_state

        assert len(gepa_state.pareto_front_valset) == len(valset), f"Pareto front valset length {len(gepa_state.pareto_front_valset)} does not match valset length {len(valset)}"

        if self.use_wandb:
            # assert gepa_state.i + 1 == 0
            import wandb # type: ignore
            wandb.log({
                "base_program_full_valset_score": gepa_state.program_full_scores_val_set[0],
                "iteration": gepa_state.i+1,
            })
        self.logger.log(f"Iteration {gepa_state.i+1}: Base program full valset score: {gepa_state.program_full_scores_val_set[0]}")

        merges_due = 0
        total_merges_tested = 0

        last_iter_found_new_program = False

        merges_performed = ([], [])

        while (
            (self.num_iters is None or gepa_state.num_full_ds_evals < self.num_iters) and
            (self.max_metric_calls is None or gepa_state.total_num_evals < self.max_metric_calls)
        ):
            assert gepa_state.is_consistent(), "GEPA state is inconsistent, please check the implementation"
            try:
                gepa_state.save(self.run_dir)
                gepa_state.i += 1
                gepa_state.full_program_trace.append({"i": gepa_state.i})

                if merges_due > 0 and last_iter_found_new_program and self.use_merge:
                    last_iter_found_new_program = False
                    try:
                        merges_due, total_merges_tested = self.attempt_merge(
                            gepa_state=gepa_state,
                            valset=valset,
                            eval_and_get_outputs=self.eval_and_get_outputs,
                            valset_evaluator=valset_evaluator,
                            merges_performed=merges_performed,
                            merges_due=merges_due,
                            total_merges_tested=total_merges_tested,
                        )
                    except ContinueException as e:
                        merges_due, total_merges_tested = e.args
                        continue

                last_iter_found_new_program = False

                # Try Reflective Prompt Mutation
                curr_prog_id = self.select_next_candidate_to_update(gepa_state)
                curr_prog = gepa_state.program_candidates[curr_prog_id] # TODO: Make DSPy changes here
                gepa_state.full_program_trace[-1]['selected_program_candidate'] = curr_prog_id

                self.logger.log(f"Iteration {gepa_state.i+1}: Selected program candidate {curr_prog_id} with base score: {gepa_state.per_program_tracked_scores[curr_prog_id]}")

                if self.use_wandb:
                    import wandb # type: ignore
                    wandb.log({
                        "iteration": gepa_state.i+1,
                        "selected_program_candidate": curr_prog_id,
                        # "predictor_to_update_id": predictor_to_update_id,
                    }, step=gepa_state.i+1)
                
                subsample_ids = self.select_training_sample_and_update_shuffled_trainset(trainset, gepa_state.i)
                gepa_state.full_program_trace[-1]['subsample_ids'] = subsample_ids

                try:
                    new_program, subsample_scores = self.reflective_mutation(
                        curr_prog_idx=curr_prog_id,
                        curr_prog=curr_prog,
                        gepa_state=gepa_state,
                        minibatch=[trainset[i] for i in subsample_ids],
                    )
                    subsample_score = sum(subsample_scores)
                except ContinueException as e:
                    self.logger.log(f"Iteration {gepa_state.i+1}: Reflective mutation failed: {e}")
                    continue

                gepa_state.total_num_evals += len(subsample_ids)

                subsample_evaluator = lambda prog: eval_and_get_outputs([trainset[i] for i in subsample_ids], prog)
                _, new_subsample_scores = subsample_evaluator(new_program)
                new_subsample_score = sum(new_subsample_scores)

                gepa_state.full_program_trace[-1]['new_subsample_scores'] = new_subsample_scores

                gepa_state.total_num_evals += len(subsample_ids)

                self.logger.log(f"Iteration {gepa_state.i+1}: New subsample score: {new_subsample_score}")
                if self.use_wandb:
                    import wandb # type: ignore
                    wandb.log({
                        "new_subsample_score": new_subsample_score,
                    }, step=gepa_state.i+1)
                
                if new_subsample_score <= subsample_score:
                    self.logger.log(f"Iteration {gepa_state.i+1}: New subsample score is not better, skipping")
                    continue

                last_iter_found_new_program = True

                self.logger.log(f"Iteration {gepa_state.i+1}: New subsample score is better, going from {subsample_score} to {new_subsample_score}, updating program candidate!")

                self.run_full_eval_add_new_program_to_gepa_tree(
                    new_program=new_program,
                    gepa_state=gepa_state,
                    valset_evaluator=valset_evaluator,
                    parent_program_idx=[curr_prog_id]
                )

                if self.use_merge and total_merges_tested < self.max_merge_invocations:
                    merges_due += 1

            except Exception as e:
                self.logger.log(f"Iteration {gepa_state.i+1}: Exception during optimization: {e}")
                self.logger.log(traceback.format_exc())
                continue
        
        gepa_state.save(self.run_dir)

        return gepa_state
