# GEPA for Math Problem Solving
**`AnyMaths`** Adapter is a GEPAAdapter for any dataset that contains mathematical word problems of varying complexity and structure. It is designed to handle a wide range of mathematical tasks, including arithmetic, algebra, and more.

Note: Just for this example, we will be using Ollama. Hence, one must ensure to have Ollama downloaded before proceeding through [this link](https://ollama.com/download). However, this is not necessary and that you may also use other provider APIs or LiteLLM-supported providers. This adapter can be used as a standalone without Ollama.

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

### Adapter Design
This adapter can work for any LiteLLM supported providers (e.g., OpenAI API, HuggingFace, Groq, vLLM, Ollama, etc.). For this instance, we opt to choose Ollama to show that **this adapter can work for local use where one has no access to expensive GPUs or paid APIs.** But, you may freely choose this adapter with any other LiteLLM-supported provider.

### Preparing the seed prompt
The seed prompt is the initial instruction you provide to the model. It sets the context for the task at hand and this prompt evolves or changes over time toward maximizing the model's performance. Default failure score (i.e., score if the model outputs incorrectly) is zero.

Set the seed prompt in a separate directory under `prompt-templates`. Inside this directory is a file `instruction_prompt.txt` which contains the seed prompt.

### Specifying the reflection LM
The reflection LM is a language model used to generate reflections on the task at hand. You can specify the reflection LM by setting the `reflection_lm` argument when calling the `optimize` function.

### Running an `AnyMaths` training
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
- `--budget`: The budget for the optimization process (default is 500).
- `--seed`: The seed for the random number generator for reproducibility (default is 0).


Example of a run:
```bash
python src/gepa/examples/anymaths-bench/train_anymaths.py --model_name "ollama/qwen3:4b" --api_base "http://localhost:11434" --max_litellm_workers 4 --anymaths_dset_name "openai/gsm8k" --reflection_lm "ollama/qwen3:8b" --reflection_minibatch_size 3 --budget 500 --seed 0
```

Once the training is completed, you may replace the optimal prompt found in `src/gepa/examples/anymaths-bench/prompt-templates/optimal_prompt.txt`.

#### Running the `eval_default.py`
`eval_default.py` is used to perform test split evaluation. Feel free to modify this script to fit your custom evaluation scheme. For this example, we evaluated it on the `openai/gsm8k` - `test` split.

Note: The model that was used in training must also be the same model used to perform evaluation.

#### Sample output
- Initial prompt
```
You are an AI assistant that solves mathematical word problems. You will be given a question and you need to provide a step-by-step solution to the problem. Finally, you will provide the answer to the question.

When outputting the final answer, make sure there are no other text or explanations included, just the answer itself.

The following fields are what you need to include in your response:
- final_answer: The final answer to the question.
- solution_pad: The step-by-step solution to the problem.
```
- Final prompt (after exhausting `budget=500`):
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


### Experiments
| Dataset - Split | Model | Accuracy (Before GEPA) $\uparrow$ | Accuracy (After GEPA) $\uparrow$ | GEPA Budget | Train-Val-Test Split Samples Used in GEPA Optimization |
| ------- | ------- | ---------------------- | --------------------- | ------------ | ------ |
| `openai/gsm8k` - `test` | `ollama/qwen3:4b` | 18% (238/1319) | 23% (300/1319) | 500 | 50-50-50 |

With just a `budget=500`, GEPA was able to improve the model's accuracy by 5% on the `openai/gsm8k` test split using the `Qwen3-4b` model.

**Notice of WIP**: More tests will be done soon on other models (preferrably, small language models first).

#### Initial prompt for `openai/gsm8k` - `test`:
```
You are an AI assistant that solves mathematical word problems. You will be given a question and you need to provide a step-by-step solution to the problem. Finally, you will provide the answer to the question.

When outputting the final answer, make sure there are no other text or explanations included, just the answer itself.

The following fields are what you need to include in your response:
- final_answer: The final answer to the question.
- solution_pad: The step-by-step solution to the problem.
```

#### Optimized prompt for `openai/gsm8k` - `test` after exhausted `budget=500`:
```
### Task Instruction: Solve Multi-Step Mathematical Problems with Precision and Contextual Understanding

You are tasked with solving problems that require careful parsing of contextual information, breaking down multi-step calculations, and ensuring accuracy in arithmetic and logical reasoning. Follow these steps to address diverse problem types (e.g., percentages, cost calculations, score determination, and distance computations):

---

#### **1. Parse the Problem**
- **Identify Key Values**: Extract numbers, percentages, fractions, and relationships (e.g., "40% of 60 students," "6 more than half of Ella\'s score").
- **Understand Relationships**: Determine if values are additive, multiplicative, or comparative (e.g., "round trips" imply doubling one-way distances, "cost per item" requires multiplication).
- **Clarify Ambiguities**: Resolve unclear phrasing (e.g., "half the score" refers to half the total items, not half the incorrect answers).

---

#### **2. Break Down the Problem**
- **Segment into Steps**: Divide the problem into smaller, manageable parts (e.g., calculate individual components before summing).
- **Apply Formulas**: Use appropriate mathematical operations (e.g., percentage = part/whole × 100, total cost = (item count × price)).
- **Account for Context**: Adjust calculations based on problem specifics (e.g., "round trip" requires doubling one-way distance, "score" may involve subtracting incorrect answers from total items).

---

#### **3. Perform Calculations**
- **Use Precise Arithmetic**:
    - For percentages: $ \\text{Percentage} \\times \\text{Total} $.
    - For fractions: $ \\frac{\\text{Numerator}}{\\text{Denominator}} \\times \\text{Value} $.
    - For multi-step operations: Follow order of operations (PEMDAS) and verify intermediate results.
- **Avoid Common Errors**:
    - Misinterpreting phrases like "half the score" (e.g., half of total items, not half of incorrect answers).
    - Confusing "round trips" (up + down) with single trips.
    - Incorrectly applying percentages to the wrong base (e.g., 40% of students vs. 40% of total score).

---

#### **4. Validate the Answer**
- **Check Logical Consistency**: Ensure results align with problem constraints (e.g., total students = sum of groups, total cost = sum of individual costs).
- **Verify Units and Formatting**: Confirm answers match required formats (e.g., boxed numbers, currency symbols, or percentage notation).
- **Cross-Validate with Examples**: Compare calculations against similar problems (e.g., "If 40% of 60 students = 24, then 60 - 24 = 36").

---

#### **5. Finalize the Response**
- **Present the Answer Clearly**: Use the exact format requested (e.g., `\boxed{36}` for numerical answers, `$77.00` for currency).
- **Include Step-by-Step Reasoning**: Explicitly show calculations (e.g., `18 * $2.50 = $45.00`, `8 * $4.00 = $32.00`).
- **Highlight Key Decisions**: Note critical choices (e.g., "Half of Ella's score = 36 items / 2 = 18 items").

---

### **Examples of Problem Types**
1. **Percentage Problems**:
    - *Input*: "40% of 60 students got below B."
    - *Solution*: $ 0.40 \\times 60 = 24 $, $ 60 - 24 = 36 $.
2. **Cost Calculations**:
    - *Input*: "18 knobs at $2.50 each and 8 pulls at $4.00 each."
    - *Solution*: $ 18 \\times 2.50 + 8 \\times 4.00 = 45 + 32 = 77 $.
3. **Score Determination**:
    - *Input*: "Ella got 4 incorrect answers; Marion got 6 more than half of Ella's score."
    - *Solution*: Total items = 40, Ella's correct = 36, half = 18, Marion = 18 + 6 = 24.
    
---

### **Key Niche Information**
- **Percentages**: Always apply to the total (e.g., 40% of 60 students = 24 students, not 40% of 40 items).
- **Round Trips**: Double one-way distances (e.g., 30,000 feet up + 30,000 feet down = 60,000 per trip).
- **Score Calculations**: Subtract incorrect answers from total items (e.g., 40 items - 4 incorrect = 36 correct).
- **Currency Formatting**: Use decimal points and symbols (e.g., `$77.00`, not `77`).

---

### **Final Output Format**\nAlways conclude with:
    `Final Answer: \boxed{<result>}`
For non-numeric answers, use:
    `Final Answer: <result>`

Ensure calculations are explicitly shown and errors are corrected based on problem context.
```

#### Observations on the structure of the derived optimal prompt
* Goal-oriented: The prompt starts by clearly stating the overall task, which is to solve multi-step math problems with precision.
* Chain-of-Thought: It breaks down the problem-solving process into a detailed, numbered sequence of five steps: **Parse**, **Break Down**, **Calculate**, **Validate**, and **Finalize**.
* Instruction Detail: Each step includes specific instructions on how to perform the task, such as identifying key values, applying formulas, and avoiding common errors.
* Few-shot Learning: The prompt provides concrete **examples** of different problem types (percentage, costs, scores) to show the model how to apply the instructions.
* Knowledge Base: It includes *key niche information* section that acts as a mini-rulebook, highlighting specific details and common pitfalls like "round trips" and currency formatting.
* Structured Output: The prompt ends by defining a strict **final output format** to ensure the model's answer is consistent and easy to read.

### Contributor
This adapter was contributed by *Emmanuel G. Maminta* ([egmaminta](https://github.com/egmaminta)).