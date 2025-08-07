import os
from typing import Any, Dict, Optional, Tuple, Callable
import dspy
import random

import dspy.teleprompt
import dspy.teleprompt.teleprompt

from typing import List
from collections import Counter

from gepa.gepa.gepa import GEPA

def idxmax(lst):
    """Return the index of the maximum value in a list."""
    max_val = max(lst)
    return lst.index(max_val)

class dspy_GEPA(dspy.teleprompt.teleprompt.Teleprompter):
    def __init__(
        self,
        named_predictor_to_feedback_fn_map: Dict[str, Callable],
        metric: Callable,
        logger,
        run_dir: str,
        run_linearized_gepa: bool=True,
        num_threads=None,
        num_iters=None,
        failure_score=0,
        perfect_score=1,
        teacher_lm: dspy.LM = None,
        use_wandb: bool = False,
        wandb_api_key: str = None,
        max_evals_per_trainval_instance=None,
        seed=0,
        skip_perfect_score=True,
        use_merge=False,
        max_merge_invocations=5,
        num_dspy_examples_per_gepa_step=3,
        max_metric_calls=None,
        add_format_failure_as_feedback: bool=False,
    ):
        # Exactly one of max_metric_calls, max_evals_per_trainval_instance or num_iters should be set
        assert (max_metric_calls is not None) + (max_evals_per_trainval_instance is not None) + (num_iters is not None) == 1, "Exactly one of max_metric_calls, max_evals_per_trainval_instance or num_iters should be set. You set max_metric_calls={}, max_evals_per_trainval_instance={}, num_iters={}".format(
            max_metric_calls, max_evals_per_trainval_instance, num_iters
        )   

        self.named_predictor_to_feedback_fn_map = named_predictor_to_feedback_fn_map
        self.metric_fn = metric
        self.logger = logger
        self.run_dir = run_dir
        self.run_linearized_gepa = run_linearized_gepa
        self.num_threads = num_threads
        
        self.failure_score = failure_score
        self.perfect_score = perfect_score
        self.teacher_lm = teacher_lm
        self.use_wandb = use_wandb
        self.wandb_api_key = wandb_api_key

        # Run constraints
        self.num_iters = num_iters
        self.max_evals_per_trainval_instance = max_evals_per_trainval_instance
        self.max_metric_calls = max_metric_calls

        self.seed = seed
        self.skip_perfect_score = skip_perfect_score
        self.use_merge = use_merge
        self.max_merge_invocations = max_merge_invocations

        self.valset_provided = None

        self.num_dspy_examples_per_gepa_step = num_dspy_examples_per_gepa_step

        self.add_format_failure_as_feedback = add_format_failure_as_feedback
        
        self.shuffled_trainset_ids = []
        self.epoch = -1
        self.id_freqs = Counter()
        self.rng = random.Random(seed)

        if self.num_threads is None:
            self.num_threads = os.cpu_count()

    def compile(
        self, student: dspy.Module, *, trainset: list[dspy.Example], teacher: Optional[dspy.Module] = None, valset: Optional[list[dspy.Example]] = None, **kwargs
    ) -> dspy.Module:
        if valset is not None:
            self.valset_provided = True

        assert trainset is not None, "Trainset must be provided"

        def create_dspy_prog_from_gepa_candidate(
            gepa_candidate: Dict[str, str]
        ) -> dspy.Module:
            new_prog = student.deepcopy()
            for idx, (name, pred) in enumerate(new_prog.named_predictors()):
                if name in gepa_candidate:
                    pred.signature = pred.signature.with_instructions(gepa_candidate[name])
            return new_prog

        def eval_and_get_outputs(inputs: List, proposed_program: Dict[str, str]) -> Tuple[Any, List[float]]:
            new_program = create_dspy_prog_from_gepa_candidate(proposed_program)
            
            evaluator = dspy.Evaluate(
                devset=inputs,
                metric=self.metric_fn,
                num_threads=self.num_threads,
                return_all_scores=True,
                return_outputs=True,
                failure_score=self.failure_score,
                provide_traceback=True,
                max_errors=len(inputs) * 100  # Allow for many errors
            )

            aggregate_score, outputs, per_example_scores = evaluator(new_program)

            return outputs, per_example_scores

        def capture_traces_and_eval(
            minibatch: List,
            curr_prog: Dict[str, str],
        ) -> Tuple[Any, List[float]]:
            """
            Capture traces and evaluate the current program on the minibatch.
            Returns a list of trajectories and their corresponding scores.
            """
            from dspy.teleprompt.bootstrap_finetune import bootstrap_trace_data, FailedPrediction
            new_prog = create_dspy_prog_from_gepa_candidate(curr_prog)
            
            trajectories = bootstrap_trace_data(
                program=new_prog,
                dataset=minibatch,
                metric=self.metric_fn,
                num_threads=self.num_threads,
                raise_on_error=False,
                capture_failed_parses=True,
                failure_score=self.failure_score,
                format_failure_score=self.failure_score,
            )

            subsample_scores = []
            for trajectory in trajectories:
                if isinstance(trajectory['prediction'], FailedPrediction):
                    subsample_scores.append(self.failure_score)
                else:
                    subsample_scores.append(trajectory['score'])
            
            return trajectories, subsample_scores

        def extract_reflection_content_from_trajectories(
            curr_prog: Dict[str, str],
            trajectories: Any,
            subsample_scores: List[float],
            predictor_names_to_update: List[str],
        ) -> Dict[str, List[Dict[str, str]]]:
            from dspy.teleprompt.bootstrap_finetune import FailedPrediction

            new_prog = create_dspy_prog_from_gepa_candidate(curr_prog)

            ret_d = {}
            for pred_name in predictor_names_to_update:
                feedback_func = self.named_predictor_to_feedback_fn_map[pred_name]
                module = None
                for m in new_prog.named_predictors():
                    if m[0] == pred_name:
                        module = m[1]
                        break
                assert module is not None

                ret = []
                for data in trajectories:
                    d = {}

                    # Trace is [dspy_module_invocation_idx -> Tuple[Predictor, PredictorInputs, Prediction]]
                    trace_instances_for_current_pred = [t for t in data["trace"] if t[0].signature.equals(module.signature)]

                    if not self.add_format_failure_as_feedback:
                        # If we are not adding format failure as feedback, we will only consider successful predictions
                        trace_instances_for_current_pred = [t for t in trace_instances_for_current_pred if not isinstance(t[2], FailedPrediction)]
                    
                    if len(trace_instances_for_current_pred) == 0:
                        # logger.log(f"Iteration {gepa_state.i+1}: No trace instances found for module {module.signature}. Skipping.")
                        continue

                    selected_trace_instance = None
                    for trace_instance in trace_instances_for_current_pred:
                        if isinstance(trace_instance[2], FailedPrediction):
                            selected_trace_instance = trace_instance

                    if selected_trace_instance is None:
                        # This means that all trace instances for current predictor are successful predictions
                        if isinstance(data['prediction'], FailedPrediction):
                            # This is coming from a different predictor, hence we don't have a good feedback for the current predictor
                            continue
                        selected_trace_instance = self.rng.choice(trace_instances_for_current_pred)

                    d['inputs'] = selected_trace_instance[1]
                    d['outputs'] = selected_trace_instance[2]

                    if isinstance(selected_trace_instance[2], FailedPrediction):
                        adapter = dspy.ChatAdapter()
                        structure_instruction = ""
                        for dd in adapter.format(module.signature, [], {}):
                            structure_instruction += dd["role"] + ": " + dd["content"] + "\n"
                        feedback_text = f"Your output text failed to parse. Please ensure that your output follows the structure:\n{structure_instruction}"
                        score = self.failure_score
                        d['feedback'] = feedback_text
                        d['score'] = score
                    else:
                        feedback_d = feedback_func(
                            predictor_output=d['outputs'], 
                            predictor_inputs=d['inputs'], 
                            module_inputs=data['example'],
                            module_outputs=data['prediction'],
                            captured_trace=data['trace'],
                        )

                        score, feedback_text = feedback_d["feedback_score"], feedback_d["feedback_text"]
                        d['feedback'] = feedback_text
                        d['score'] = score

                    ret.append(d)
                
                if len(ret) == 0:
                    self.logger.log(f"Iteration {gepa_state.i+1}: No valid predictions found for module {module.signature}. Skipping.")
                    raise Exception("No valid predictions found for module {module.signature}.")

                ret_d[pred_name] = ret
            return ret_d

        def reflect_and_propose_new_text_candidate(
            curr_prog: Dict[str, str],
            reflective_dataset: Dict[str, List[Dict[str, str]]],
            predictor_names_to_update: List[str]
        ) -> Dict[str, str]:
            from .instruction_proposal import ProposeNewInstructionModule

            new_instructions = {}

            for pred_name in predictor_names_to_update:
                base_instruction = curr_prog[pred_name]
                dataset_with_feedback = reflective_dataset[pred_name]

                instruction_propose_module = ProposeNewInstructionModule(
                    base_instruction=base_instruction,
                    instruction_lm=self.teacher_lm or dspy.settings.lm or student.get_lm(),
                    dataset_with_feedback=dataset_with_feedback
                )

                new_instruction = instruction_propose_module.compile()

                new_instructions[pred_name] = new_instruction
            
            return new_instructions
        
        gepa_obj = GEPA(
            logger=self.logger,
            run_dir=self.run_dir,
            run_linearized_gepa=self.run_linearized_gepa,
            num_iters=self.num_iters,
            perfect_score=self.perfect_score,
            use_wandb=self.use_wandb,
            wandb_api_key=self.wandb_api_key,
            max_evals_per_trainval_instance=self.max_evals_per_trainval_instance,
            seed=self.seed,
            skip_perfect_score=self.skip_perfect_score,
            use_merge=self.use_merge,
            max_merge_invocations=self.max_merge_invocations,
            num_examples_per_gepa_step=self.num_dspy_examples_per_gepa_step,
            max_metric_calls=self.max_metric_calls,
        )

        gepa_state = gepa_obj.gepa(
            base_program={name: pred.signature.instructions for name, pred in student.named_predictors()},
            trainset=trainset,
            eval_and_get_outputs=eval_and_get_outputs,
            capture_traces_and_eval=capture_traces_and_eval,
            extract_reflection_content_from_trajectories=extract_reflection_content_from_trajectories,
            reflect_and_propose_new_text_candidate=reflect_and_propose_new_text_candidate,
            valset=valset,
        )

        best_prog_idx = idxmax(gepa_state.per_program_tracked_scores)

        new_prog = student.deepcopy()
        for idx, (name, pred) in enumerate(new_prog.named_predictors()):
            if name in gepa_state.program_candidates[best_prog_idx]:
                pred.signature = pred.signature.with_instructions(gepa_state.program_candidates[best_prog_idx][name])
        
        return new_prog
