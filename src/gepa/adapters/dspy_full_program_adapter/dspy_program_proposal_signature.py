# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa


from gepa.proposer.reflective_mutation.base import Signature


class DSPyProgramProposalSignature(Signature):
    prompt_template = """Primer on DSPy:

### DSPy Signatures:
Signatures define LM tasks via typed input/output fields and instructions.

Simple Signatures: `"input1, ..., inputN -> output1, ..., outputM"`. Example: `"topic -> tweet"` defines a simple signature that takes a topic as input and generates a tweet as output.

Typed Signatures: A class that inherits from `dspy.Signature`. Class docstring can provide detailed LM instructions; `dspy.InputField`/`dspy.OutputField` with `desc` and pydantic types (e.g., `str`, `List[str]`, `Literal`).

Example:
```
from typing import List
class ArticleAnalysis(dspy.Signature):
    \"""Extract topic and three key points from a news article.\"""
    article_text = dspy.InputField(desc="Article content.")
    topic = dspy.OutputField(desc="Main topic phrase.")
    key_points: List[str] = dspy.OutputField(desc="Three key takeaways.")
```

### DSPy Modules: Define how a task defined by a signature is solved.
DSPy Modules are composable blocks like PyTorch layers, applying signatures to LMs.

#### Module I/O:
Typed by signature; inputs as kwargs matching input fields; outputs as `dspy.Prediction` with output fields.

#### Basic Modules
- `dspy.Predict(signature)`: Defines a single LM call for a signature providing the inputs and expect the LM to directly output all the output fields.
- `dspy.ChainOfThought(signature)`: Defines a single LM calls, where the LM receives the inputs and is expected to first generate a reasoning chain, and then output all the output fields.

Example:
```
program = dspy.Predict(ArticleAnalysis) # Or dspy.ChainOfThought(ArticleAnalysis)
# Can be called like:
# pred = program(article_text="LLMs on consumer hardware...")
# print(pred.topic, pred.key_points)
```

#### Custom Modules
For more complex tasks, the workflow can be defined by creating a custom module. Inherit `dspy.Module`; `__init__` initializes sub-modules; `forward` defines data flow, returning `dspy.Prediction`. forward receives inputs as kwargs matching input fields and returns dspy.Prediction with output fields.

Example: Summarize then extract takeaways.
```
class ArticleAnalysis2Stage(dspy.Module):
    def __init__(self):
        super().__init__()
        self.key_points_extractor = dspy.Predict("article_text -> key_points: List[str]")
        self.topic_extractor = dspy.Predict("key_points: List[str] -> topic")

    def forward(self, document):
        key_points_pred = self.key_points_extractor(document=document)
        topic_pred = self.topic_extractor(key_points=key_points_pred.key_points)
        return dspy.Prediction(topic=topic_pred.topic, key_points=key_points_pred.key_points)

program = ArticleAnalysis2Stage()
# Can be called same as above:
# pred = program(article_text="LLMs on consumer hardware...")
# print(pred.topic, pred.key_points)
```

I am trying to solve a task using the DSPy AI framework. Here's my current code:
```
<curr_program>
```

Here is the execution trace of the current code on some example inputs, their corresponding generated outputs, and the feedback on how the generated outputs could be better:
```
<dataset_with_feedback>
```

Propose a new version of the code that improves the performance of the current code. Ensure that the new code is a drop-in replacement for the current code, including creating a `program` object that will be used to solve the task. Respond with exactly one code block at the end, within ```. Do not include language marker in the code block."""
    input_keys = ["curr_program", "dataset_with_feedback"]
    output_keys = ["new_program"]

    @classmethod
    def prompt_renderer(cls, input_dict: dict[str, str]) -> str:
        def format_samples(samples):
            def render_value(value, level=3):
                # level controls markdown header depth (###, ####, etc.)
                if isinstance(value, dict):
                    s = ""
                    for k, v in value.items():
                        s += f"{'#' * level} {k}\n"
                        s += render_value(v, min(level + 1, 6))
                    if not value:
                        s += "\n"
                    return s
                elif isinstance(value, (list, tuple)):
                    s = ""
                    for i, item in enumerate(value):
                        s += f"{'#' * level} Item {i + 1}\n"
                        s += render_value(item, min(level + 1, 6))
                    if not value:
                        s += "\n"
                    return s
                else:
                    return f"{str(value).strip()}\n\n"

            def convert_sample_to_markdown(sample, examplenum):
                s = f"# Example {examplenum}\n"
                for key, val in sample.items():
                    s += f"## {key}\n"
                    s += render_value(val, level=3)
                return s

            return "\n\n".join(convert_sample_to_markdown(sample, i + 1) for i, sample in enumerate(samples))

        prompt = cls.prompt_template
        prompt = prompt.replace("<curr_program>", input_dict["curr_program"])
        prompt = prompt.replace("<dataset_with_feedback>", format_samples(input_dict["dataset_with_feedback"]))
        return prompt

    @classmethod
    def output_extractor(cls, lm_out: str) -> dict[str, str]:
        # Extract ``` blocks
        new_instruction = None
        if lm_out.count("```") >= 2:
            start = lm_out.find("```")
            end = lm_out.rfind("```")
            if start >= end:
                new_instruction = lm_out
            if start == -1 or end == -1:
                new_instruction = lm_out
            else:
                new_instruction = lm_out[start + 3 : end].strip()
        else:
            lm_out = lm_out.strip()
            if lm_out.startswith("```"):
                lm_out = lm_out[3:]
            if lm_out.endswith("```"):
                lm_out = lm_out[:-3]
            new_instruction = lm_out

        return {"new_program": new_instruction}
