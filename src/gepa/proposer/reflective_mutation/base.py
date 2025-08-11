from typing import Protocol, List, Dict, Callable
from dataclasses import dataclass
from gepa.core.state import GEPAState
from gepa.core.adapter import Trajectory

class CandidateSelector(Protocol):
    def select_candidate_idx(self, state: GEPAState) -> int:
        ...

class ReflectionComponentSelector(Protocol):
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

class LanguageModel(Protocol):
    def __call__(self, prompt: str) -> str:
        ...

@dataclass
class Signature:
    prompt_template: str
    input_keys: List[str]
    output_keys: List[str]
    prompt_renderer: Callable[[Dict[str, str]], str]
    output_extractor: Callable[[str], Dict[str, str]]

    @classmethod
    def run(cls, lm: LanguageModel, input_dict: Dict[str, str]) -> Dict[str, str]:
        full_prompt = cls.prompt_renderer(input_dict)
        lm_out = lm(full_prompt).strip()
        return cls.output_extractor(lm_out)
