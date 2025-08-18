# GEPA for Math Problem Solving
**`AnyMaths`** Adapter is a GEPAAdapter for any dataset that contains mathematical word problems of varying complexity and structure. It is designed to handle a wide range of mathematical tasks, including arithmetic, algebra, and more.

Note: `Ollama` must be installed and configured to use this adapter. Instructions to download Ollama can be found in [this link](https://ollama.com/download). I recommend the Linux installation.

### Preparing the maths & reasoning dataset
In `src/gepa/examples/anymaths-bench/train_anymaths.py`, a sample function to prepare any maths dataset is provided via `init_dataset`. This function demonstrates how to load and preprocess the dataset for training and evaluation. Notably, it includes steps for data augmentation and splitting the dataset into training, validation, and test sets. Right now, it is more convenient to find datasets from Hugging Face dataset hub.

#### Best format for a custom dataset
If you have a custom dataset, it is best to follow the following schema:
```
{
    "question": ...,
    "solution": ...,
    "answer": ...
}
```
- `question` must be a string/text.
- `solution` must be a string/text.
- `answer` must be a purely numerical, no other text or units associated.

It is best to upload your custom dataset to the Hugging Face dataset hub to fully utilize `datasets.load_dataset`.

### Ollama-only
This is an Ollama-only adapter, meaning it is specifically designed to work with the Ollama platform via LiteLLM. This design choice is based on the fact that not everyone has access to other provider APIs requiring paid subscriptions.

### Preparing the seed prompt
The seed prompt is the initial instruction you provide to the model. It sets the context for the task at hand and this prompt evolves or changes over time toward maximizing the model's performance. Default failure score (i.e., score if the model outputs incorrectly) is zero.

Set the seed prompt in a separate directory under `prompt-templates`. Inside this directory is a file `instruction_prompt.txt` which contains the seed prompt.

### Specifying the reflection LM
The reflection LM is a language model used to generate reflections on the task at hand. You can specify the reflection LM by setting the `reflection_lm` argument when calling the `optimize` function.

### Running a sample `AnyMaths` training
To run a sample training session using the `AnyMaths` adapter, you can use the following command:

```bash
python src/gepa/examples/anymaths-bench/train_anymaths.py --model_name ... --api_base ... --max_litellm_workers ... --anymaths_dset_name ... --reflection_minibatch_size ... --budget ...
```
- `--model_name`: The model checkpoint to use for training (e.g., `"ollama/qwen3:4b"`).
- `--api_base`: The base URL for the Ollama API (e.g., `http://localhost:11434`).
- `--max_litellm_workers`: The maximum number of LiteLLM workers to use.
- `--anymaths_dset_name`: The name of the AnyMaths dataset to use for training.
- `--reflection_lm`: The name of the reflection language model to use (e.g., `"ollama/qwen3:8b"`).
- `--reflection_minibatch_size`: The size of the minibatch for the reflection LM (default is 3).
- `--budget`: The budget for the optimization process (default is 50).
- `--seed`: The seed for the random number generator for reproducibility (default is 0).


Example of a run:
```bash
python src/gepa/examples/anymaths-bench/train_anymaths.py --model_name "ollama/qwen3:4b" --api_base "http://localhost:11434" --max_litellm_workers 4 --anymaths_dset_name "openai/gsm8k" --reflection_lm "ollama/qwen3:8b" --reflection_minibatch_size 3 --budget 10 --seed 0
```

#### Sample output
- Initial prompt
```
You are an AI assistant that solves mathematical word problems. You will be given a question and you need to provide a step-by-step solution to the problem. Finally, you will provide the answer to the question.

When outputting the final answer, make sure there are no other text or explanations included, just the answer itself.

The following fields are what you need to include in your response:
- final_answer: The final answer to the question.
- solution_pad: The step-by-step solution to the problem.
```
- Final prompt (after 4 iterations):
```
You are an AI assistant that solves mathematical word problems. You will be given a question and you need to provide a step-by-step solution to the problem. Finally, you will provide the answer to the question.

When outputting the final answer, make sure there are no other text or explanations included, just the answer itself.

The following fields are what you need to include in your response:
- final_answer: The final answer to the question.
- solution_pad: The step-by-step solution to the problem.

Key guidelines:
1. **Break down the problem into clear steps** and calculate each part sequentially.
2. **Verify all arithmetic operations** (addition, subtraction, multiplication, division, percentages) for accuracy.
3. **Use proper notation** for intermediate steps (e.g., <<calculation>>).
4. **Avoid common errors**:
    - Correctly compute percentages (e.g., 10% of 90 = 9, not 9.5).
    - Ensure averages are calculated by summing all values and dividing by the count.
    - Match the final answer exactly to the correct sequence of operations.
5. **Double-check** that the final answer is derived directly from the problem's logic, not from intermediate miscalculations.

Example: If the problem involves multiple steps (e.g., percentages, averages, comparisons), ensure each step is logically connected and mathematically accurate.
```


### Contributor
This adapter was contributed by *Emmanuel G. Maminta* ([egmaminta](https://github.com/egmaminta)).