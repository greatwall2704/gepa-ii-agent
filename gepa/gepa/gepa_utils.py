from copy import deepcopy
import os
import random
import json
from typing import Any, Callable, Dict, List, Tuple, Union

from .entropy_utils import remove_dominated_programs

class GEPAState:
    program_candidates: List[Dict[str, str]]
    parent_program_for_candidate: List[List[Union[int, None]]]

    program_full_scores_val_set: List[float]

    program_at_pareto_front_valset: List[set[int]]

    prog_candidate_val_subscores: List[List[float]]

    list_of_named_predictors: List[str]
    named_predictor_id_to_update_next_for_program_candidate: List[int]

    i: int
    num_full_ds_evals: int

    total_num_evals: int

    num_metric_calls_by_discovery: List[int]

    running_linearized_gepa: bool

    rng1: random.Random

    full_program_trace: List

    per_program_tracked_scores: List[float]

    def __init__(
        self, 
        base_program: Dict[str, str],
        base_valset_eval_output: tuple[Any, List[float]],
        seed: int, 
        run_linearized_gepa: bool=False
    ):
        valset_base_score = sum(base_valset_eval_output[1]) / len(base_valset_eval_output[1])
        base_valset_pareto_front = base_valset_eval_output[1]

        self.program_candidates = [base_program]
        self.program_full_scores_val_set = [valset_base_score]
        
        self.per_program_tracked_scores = [valset_base_score]

        self.pareto_front_valset = base_valset_pareto_front
        self.parent_program_for_candidate = [[None]]
        self.program_at_pareto_front_valset = [{0} for _ in range(len(base_valset_pareto_front))]

        self.list_of_named_predictors = list(base_program.keys())
        self.named_predictor_id_to_update_next_for_program_candidate = [0]
        self.i = -1
        self.rng1 = random.Random(seed)

        self.prog_candidate_val_subscores = [base_valset_eval_output[1]]
        self.num_metric_calls_by_discovery = [0]

        self.running_linearized_gepa = run_linearized_gepa

        self.full_program_trace = []

    def is_consistent(self):
        assert len(self.program_candidates) == len(self.program_full_scores_val_set)
        assert len(self.program_candidates) == len(self.per_program_tracked_scores)
        assert len(self.program_candidates) == len(self.parent_program_for_candidate)
        assert len(self.program_candidates) == len(self.named_predictor_id_to_update_next_for_program_candidate)
        
        assert len(self.prog_candidate_val_subscores) == len(self.program_candidates)
        assert len(self.pareto_front_valset) == len(self.program_at_pareto_front_valset)
        assert len(self.program_candidates) == len(self.num_metric_calls_by_discovery)

        for prog_list in self.program_at_pareto_front_valset:
            for prog_idx in prog_list:
                assert prog_idx < len(self.program_candidates), "Program index in valset pareto front exceeds number of program candidates"

        return True

    def save(self, run_dir:str):
        # Save all the other state except programs as pickle
        with open(os.path.join(run_dir, "gepa_state.bin"), 'wb') as f:
            import pickle
            d = {k: v for k, v in self.__dict__.items()}
            pickle.dump(d, f)

    @staticmethod
    def load(run_dir: str) -> 'GEPAState':
        with open(os.path.join(run_dir, "gepa_state.bin"), 'rb') as f:
            import pickle
            d = pickle.load(f)
        state = GEPAState.__new__(GEPAState)
        state.__dict__.update(d)
        
        assert len(state.program_candidates) == len(state.program_full_scores_val_set)
        assert len(state.pareto_front_valset) == len(state.program_at_pareto_front_valset)

        assert len(state.program_candidates) == len(state.parent_program_for_candidate)
        assert len(state.program_candidates) == len(state.named_predictor_id_to_update_next_for_program_candidate)

        return state

    def update_state_with_new_program(
        self,
        parent_program_idx: List[int], 
        new_program: Dict[str, str], 
        valset_score: float,
        valset_outputs: Any, 
        valset_subscores: List[float], 
        run_dir: str, 
        num_metric_calls_by_discovery_of_new_program: int
    ):
        new_program_idx = len(self.program_candidates)
        self.program_candidates.append(new_program)
        self.num_metric_calls_by_discovery.append(num_metric_calls_by_discovery_of_new_program)
        # Find the highest predictor id from the parent programs
        max_predictor_id = max([self.named_predictor_id_to_update_next_for_program_candidate[p] for p in parent_program_idx])
        self.named_predictor_id_to_update_next_for_program_candidate.append(max_predictor_id)
        self.parent_program_for_candidate.append(parent_program_idx)

        self.prog_candidate_val_subscores.append(valset_subscores)
        self.program_full_scores_val_set.append(valset_score)
        for task_idx, (old_score, new_score) in enumerate(zip(self.pareto_front_valset, valset_subscores)):
            if new_score > old_score:
                self.pareto_front_valset[task_idx] = new_score
                self.program_at_pareto_front_valset[task_idx] = {new_program_idx}
                os.makedirs(os.path.join(run_dir, "generated_best_outputs_valset", f"task_{task_idx}"), exist_ok=True)
                with open(os.path.join(run_dir, "generated_best_outputs_valset", f"task_{task_idx}", f"iter_{self.i+1}_prog_{new_program_idx}.json"), 'w') as f:
                    json.dump(valset_outputs[task_idx], f, indent=4, default=json_default)
            elif new_score == old_score:
                self.program_at_pareto_front_valset[task_idx].add(new_program_idx)

        assert len(valset_subscores) == len(self.program_at_pareto_front_valset)

        self.per_program_tracked_scores = self.program_full_scores_val_set

        linear_pareto_front_program_idx = idxmax(self.per_program_tracked_scores)

        return new_program_idx, linear_pareto_front_program_idx

def json_default(x):
    """Default JSON encoder for objects that are not serializable by default."""
    try:
        return {**x}
    except:
        return repr(x)

def idxmax(lst: List[float]) -> int:
    """Return the index of the maximum value in a list."""
    max_val = max(lst)
    return lst.index(max_val)

def write_eval_output_to_directory(
    eval_out: Tuple[Any, List[float]], 
    output_dir: str
):
    for task_idx, score in enumerate(eval_out[1]):
        os.makedirs(os.path.join(output_dir, f"task_{task_idx}"), exist_ok=True)
        with open(os.path.join(output_dir, f"task_{task_idx}", f"iter_{0}_prog_0.json"), 'w') as f:
            json.dump(eval_out[1][task_idx], f, indent=4, default=json_default)

def initialize_wandb(wandb_api_key: str = None, run_dir: str = None):
    try:
        import wandb
        if wandb_api_key:
            wandb.login(key=wandb_api_key, verify=True)
        else:
            wandb.login()
    except ImportError:
        raise ImportError("wandb is not installed. Please install it or set use_wandb=False.")
    except Exception as e:
        raise RuntimeError(f"Error logging into wandb: {e}")
    
    wandb_run = wandb.init(
        project="gepa",
        dir=os.path.join(run_dir, "wandb"),
        name=run_dir,
    )
    return wandb_run

def initialize_gepa_state(
    gepa_state_to_use: Union[GEPAState, None],
    run_dir: str, 
    logger, 
    base_program: Dict[str, str],
    valset_evaluator: Callable[[Dict[str, str]], Tuple[Any, List[float]]],
    seed: int, 
    run_linearized_gepa: bool,
):
    if gepa_state_to_use is None:
        if os.path.exists(os.path.join(run_dir, "gepa_state.bin")) and os.path.exists(os.path.join(run_dir, "prog_candidates")):
            logger.log("Loading gepa state from run dir")
            gepa_state = GEPAState.load(run_dir)
        else:
            num_evals_run = 0

            valset_out = valset_evaluator(base_program)
            write_eval_output_to_directory(valset_out, os.path.join(run_dir, "generated_best_outputs_valset"))
            num_evals_run += len(valset_out[1])

            gepa_state = GEPAState(
                base_program, 
                valset_out,
                seed,
                run_linearized_gepa=run_linearized_gepa,
            )

            gepa_state.num_full_ds_evals = 1
            gepa_state.total_num_evals = num_evals_run

    else:
        gepa_state = gepa_state_to_use

    return gepa_state

def find_dominator_programs(pareto_front_programs, train_val_weighted_agg_scores_for_all_programs):
    train_val_pareto_front_programs = pareto_front_programs
    new_program_at_pareto_front_valset = remove_dominated_programs(train_val_pareto_front_programs, scores=train_val_weighted_agg_scores_for_all_programs)
    uniq_progs = []
    for front in new_program_at_pareto_front_valset:
        uniq_progs.extend(front)
    uniq_progs = set(uniq_progs)
    return list(uniq_progs)

def select_program_candidate_from_pareto_front(pareto_front_programs, train_val_weighted_agg_scores_for_all_programs, rng):
    train_val_pareto_front_programs = pareto_front_programs
    new_program_at_pareto_front_valset = remove_dominated_programs(train_val_pareto_front_programs, scores=train_val_weighted_agg_scores_for_all_programs)
    program_frequency_in_validation_pareto_front = {}
    for testcase_pareto_front in new_program_at_pareto_front_valset:
        for prog_idx in testcase_pareto_front:
            if prog_idx not in program_frequency_in_validation_pareto_front:
                program_frequency_in_validation_pareto_front[prog_idx] = 0
            program_frequency_in_validation_pareto_front[prog_idx] += 1
    
    sampling_list = [prog_idx for prog_idx, freq in program_frequency_in_validation_pareto_front.items() for _ in range(freq)]
    assert len(sampling_list) > 0
    curr_prog_id = rng.choice(sampling_list)
    return curr_prog_id

def log_detailed_metrics_after_discovering_new_program(logger, gepa_state: GEPAState, valset_score, new_program_idx, valset_subscores, new_instruction, use_wandb, linear_pareto_front_program_idx):
    best_prog_as_per_agg_score = idxmax(gepa_state.per_program_tracked_scores)
    best_prog_as_per_agg_score_valset = idxmax(gepa_state.program_full_scores_val_set)

    logger.log(f"Iteration {gepa_state.i+1}: Full valset score for new program: {valset_score}")
    logger.log(f"Iteration {gepa_state.i+1}: Full train_val score for new program: {gepa_state.per_program_tracked_scores[new_program_idx]}")
    logger.log(f"Iteration {gepa_state.i+1}: Individual valset scores for new program: {valset_subscores}")
    logger.log(f"Iteration {gepa_state.i+1}: New valset pareto front scores: {gepa_state.pareto_front_valset}")
    logger.log(f"Iteration {gepa_state.i+1}: Full valset pareto front score: {sum(gepa_state.pareto_front_valset)/len(gepa_state.pareto_front_valset)}")
    logger.log(f"Iteration {gepa_state.i+1}: Updated valset pareto front programs: {gepa_state.program_at_pareto_front_valset}")
    logger.log(f"Iteration {gepa_state.i+1}: Best valset aggregate score so far: {max(gepa_state.program_full_scores_val_set)}")
    logger.log(f"Iteration {gepa_state.i+1}: Best program as per aggregate score on train_val: {best_prog_as_per_agg_score}")
    logger.log(f"Iteration {gepa_state.i+1}: Best program as per aggregate score on valset: {best_prog_as_per_agg_score_valset}")
    logger.log(f"Iteration {gepa_state.i+1}: Best score on valset: {gepa_state.program_full_scores_val_set[best_prog_as_per_agg_score_valset]}")
    logger.log(f"Iteration {gepa_state.i+1}: Best score on train_val: {gepa_state.per_program_tracked_scores[best_prog_as_per_agg_score]}")
    logger.log(f"Iteration {gepa_state.i+1}: Linear pareto front program index: {linear_pareto_front_program_idx}")
    logger.log(f"Iteration {gepa_state.i+1}: New program candidate index: {new_program_idx}")

    wandb_logs = {
        "iteration": gepa_state.i+1,
        "full_valset_score": valset_score,
        "full_train_val_score": gepa_state.per_program_tracked_scores[new_program_idx],
        "new_instruction": new_instruction,
        "new_program_idx": new_program_idx,
        "valset_pareto_front_scores": gepa_state.pareto_front_valset,
        "individual_valset_score_new_program": valset_subscores,
        "valset_pareto_front_agg": sum(gepa_state.pareto_front_valset)/len(gepa_state.pareto_front_valset),
        "valset_pareto_front_programs": gepa_state.program_at_pareto_front_valset,
        "best_valset_agg_score": max(gepa_state.program_full_scores_val_set),
        "linear_pareto_front_program_idx": linear_pareto_front_program_idx,
        "best_program_as_per_agg_score": best_prog_as_per_agg_score,
        "best_program_as_per_agg_score_valset": best_prog_as_per_agg_score_valset,
        "best_score_on_valset": gepa_state.program_full_scores_val_set[best_prog_as_per_agg_score_valset],
        "best_score_on_train_val": gepa_state.per_program_tracked_scores[best_prog_as_per_agg_score],
    }

    if use_wandb:
        import wandb
        wandb.log(wandb_logs, step=gepa_state.i+1)
