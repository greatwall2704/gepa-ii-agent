import dspy
from dspy.teleprompt.bootstrap_finetune import FailedPrediction

prompt1 = """I provided an assistant with the following instructions to perform a task for me:
```
<curr_instructions>
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
<inputs_outputs_feedback>
```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Provide the new instructions within ``` blocks."""

def call_lm_and_extract_response(prompt, lm, current_instruction_doc, user_examples_and_feedback, reference_materials=None):
    full_prompt = prompt.replace("<curr_instructions>", current_instruction_doc)
    full_prompt = full_prompt.replace("<inputs_outputs_feedback>", user_examples_and_feedback)
    if reference_materials is not None:
        full_prompt = full_prompt.replace("<reference_materials>", reference_materials)
    
    lm_out = lm(full_prompt, max_tokens=16384)[0].strip()
    # Extract ``` blocks
    if lm_out.count("```") >= 2:
        start = lm_out.find("```")
        end = lm_out.rfind("```")
        if start >= end:
            return lm_out
        if start == -1 or end == -1:
            return lm_out
        else:
            return lm_out[start+3:end].strip()
    else:
        lm_out = lm_out.strip()
        if lm_out.startswith("```"):
            lm_out = lm_out[3:]
        if lm_out.endswith("```"):
            lm_out = lm_out[:-3]
        return lm_out

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

def ProposeNewInstructionModule(base_instruction, dataset_with_feedback, instruction_lm):
    instruction = base_instruction

    sample = dataset_with_feedback
    new_instruction = call_lm_and_extract_response(
        prompt1,
        instruction_lm,
        current_instruction_doc=instruction,
        user_examples_and_feedback=format_samples(sample)
    )

    return new_instruction