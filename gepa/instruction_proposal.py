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

class ProposeNewInstructionModule:
    def __init__(self, base_instruction, instruction_lm, dataset_with_feedback):
        self.base_instruction = base_instruction
        self.dataset_with_feedback = dataset_with_feedback

        self.kb_query = None
        self.kb_fetch = None

        self.instruction_lm = instruction_lm
    
    def format_samples(self, samples):
        def convert_sample_to_markdown(sample, examplenum):
            s = "# Example " + str(examplenum) + "\n"
            s += "## Inputs\n"
            contains_history = False
            history_key_name = None
            if 'inputs' not in sample:
                raise ValueError("Sample does not contain 'inputs' key." + str(sample))
            for input_key, input_val in sample['inputs'].items():
                if isinstance(input_val, dspy.History):
                    contains_history = True
                    assert history_key_name is None
                    history_key_name = input_key
            
            if contains_history:
                s += f"### Context\n"
                s += "```json\n"
                for i, message in enumerate(sample['inputs'][history_key_name].messages):
                    s += f"  {i}: {message}\n"
                s += "```\n\n"
            
            for input_key, input_val in sample['inputs'].items():
                if contains_history and input_key == history_key_name:
                    continue
                s += f"### {input_key}\n"
                s += str(input_val) + "\n\n"

            s += "## Generated Outputs\n"
            if isinstance(sample['outputs'], FailedPrediction):
                print("Adding raw response for failed prediction.")
                s += "Couldn't parse the output as per the expected output format. The model's raw response was:\n"
                s += "```\n"
                s += sample['outputs'].completion_text + "\n"
                s += "```\n\n"
            else:
                for output_key, output_val in sample['outputs'].items():
                    s += f"### {output_key}\n"
                    s += str(output_val) + "\n\n"

            s += "## Feedback\n"
            s += sample['feedback'] + "\n\n"

            return s
        
        return "\n\n".join([convert_sample_to_markdown(sample, i+1) for i, sample in enumerate(samples)])
    
    def compile(self):
        instruction = self.base_instruction

        sample = self.dataset_with_feedback
        new_instruction = call_lm_and_extract_response(
            prompt1,
            self.instruction_lm,
            current_instruction_doc=instruction,
            user_examples_and_feedback=self.format_samples(sample)
        )

        return new_instruction
