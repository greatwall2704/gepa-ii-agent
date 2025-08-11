# GEPA: System Optimization through Reflective Text Evolution

> **Efficiently evolve and optimize any system controlled by text—AI prompts, code, or instructions—using reflective evolution.**

---

## Overview

**GEPA** (Genetic-Pareto) offers a novel, sample-efficient framework for **optimizing arbitrary systems composed of text components**—such as the prompts of AI systems, code snippets/code files in a project, or other textual specifications—against any desired evaluation metric. GEPA uses **language models (LLMs) to reflect on the system's own behavior and outcomes, leveraging textual feedback from both execution and evaluation traces to guide strategic improvements.** Through iterative text mutation, reflection, and Pareto-aware candidate selection, GEPA efficiently discovers robust, high-performing variants of your system, *with minimal rollouts or evaluation calls*. GEPA can co-evolve multiple text-components belonging to the same system.

GEPA can **optimize any modular system that exposes text parameters** (instructions, prompts, code blocks, etc.), extracting maximal signal from every costly system execution, and producing domain-specific improvements.

> **The easiest and most powerful way to use GEPA is within [DSPy](https://dspy.ai/), where the GEPA algorithm is directly available through the `dspy.GEPA` API. If you use DSPy for building your AI systems, you already have access to GEPA without special integration.**

This repository provides the official implementation of the GEPA algorithm as proposed in the paper titled "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning" ([https://arxiv.org/abs/2507.19457](https://arxiv.org/abs/2507.19457)).

---

## Key Principles

GEPA is built on three fundamental ideas:

### 1. **Textual Reflection Instead of Blind Mutation**
Instead of black-box mutation or random evolution, GEPA uses LLMs to **reflect in natural language on the trajectories and outcomes of candidate systems**. This enables targeted, interpretable updates: the LLMs can diagnose failure points, understand context, leverage their vast world-knowledge priors and propose nuanced textual edits grounded in observed behavior.

### 2. **Rich Text Feedback as Optimization Signal**
GEPA can leverage *any* available textual feedback: not just execution logs from the system itself, but also rich traces from evaluation metrics (e.g., test-case logs, compiler error messages, profiler traces, etc.). This feedback becomes the input for LLM-based reflection, enabling **credit assignment and domain-aware optimization even for complex, multi-component systems**.

### 3. **Pareto-based Evolution of System Candidates**
GEPA **tracks and samples candidates for mutation from a Pareto frontier of high-performing candidates across all evaluation instances**. This preserves solution diversity, accumulates complementary strategies, and avoids premature convergence—allowing GEPA to stochastically combine and evolve candidates that individually win on different instances.

---

## What Can GEPA Optimize?

- **Compound LLM programs:** Multi-stage and multi-module LLM systems with orchestrated control flow.
- **Agents or tools with text instructions:** Any system where text determines behavior (e.g., prompt sets, user guides, modular agent instructions).
- **Code:** GEPA can be leveraged to evolve critical code snippets against performance and correctness metrics.
- **Any system whose key behaviors are controlled by editable textual components.**

**GEPA is model- and metric-agnostic:** supply any callable system, any evaluation function, and (optionally) an LLM for reflection.

---

## How Does GEPA Work?

### Iterative Optimization Loop

GEPA starts with a seed candidate, consisting of default instantiation of all the components of the sysem and tracks the scores achieved by every proposed candidate on all validation set instances. Then, at each iteration:

1. **Candidate Selection:** Choose a candidate from the Pareto frontier.
2. **Rollout & Trace:** Execute the candidate on a sampled minibatch, capturing inputs, outputs, and optional traces.
3. **Module Selection:** Pick a text component within the candidate to update.
4. **Construct Feedback Dataset:** For the selected component, extract the relevant context, inputs, outcomes, and feedback into a record.
5. **LLM Reflection:** Submit this record to an LLM, prompting it to propose an improved version of the text component.
6. **Mutation & Evaluation:** Instantiate a new candidate using the improved text, evaluate on the same minibatch, and accept if strictly improved.
7. **Pareto Pool Update:** Add accepted candidates to the pool and recompute Pareto front.

Optionally, GEPA also periodically performs **system-aware merges or crossovers**—combining components from different lineages that have independently evolved strong strategies.

## Why Does GEPA Work Well?

- **Sample Efficiency through reflection:** GEPA makes maximal use of each rollout—LLMs learn from mistakes through language, giving richer updates than scalar reward-only methods.
- **Interpretability:** Improvements are surfaced as new text, often with explicit rationale. GEPA's optimization history is auditable: you can see exactly how each modification arose.
- **Robust Generalization:** Pareto-based tracking preserves diverse strategies and avoids loss of promising candidates due to greedy search.
- **Strong Empirical Results:** Across diverse benchmarks—including multi-hop QA, privacy-preserving rewriting, and code optimization—GEPA shows strong ability to optimize the AI system for the task (See [https://arxiv.org/abs/2507.19457](https://arxiv.org/abs/2507.19457)).

---

## Using GEPA

### The Easiest Path: [DSPy Integration](https://dspy.ai/)

We highly recommend using GEPA from within DSPy as [dspy.GEPA](https://dspy.ai/api/optimizers/GEPA/). If your AI system is built using [DSPy](https://dspy.ai/), GEPA is available as a plug-in optimizer. With DSPy, you don't need to implement any adapter—all modules, pipelines, and demonstrations are already compatible.

**See [DSPy Documentation](https://dspy.ai/) for how to invoke the GEPA optimizer out-of-the-box.**

### Using GEPA Directly

GEPA can be used to optimize any system consisting of textual components. Follow these steps:
 - **Implement `GEPAAdapter`:** In order to allow the GEPA optimizer to pair with your system and its environment, GEPA requires users to implement the `GEPAAdapter` interface defined in [src/gepa/gepa.py](src/gepa/gepa.py). `GEPAAdapter` requires 2 methods:
    - Evaluate: Given a candidate consisting of proposed text components, and a minibatch of inputs sampled from the train/val sets, evaluate and return execution scores, also capturing the system traces.
    - Extract Traces for Reflection: Given the execution traces obtained from executing a proposed candidate, and a named component being optimized, return the textual content from the traces relevant to the named component.
- **Prepare trainset and valset:** Lists of example inputs and task metadata.
- **Call `gepa.optimize`** with your adapter, metric, and system configuration.

> We are actively working on providing an easy to use interface to allow using GEPA for novel scenarios easier!

### Minimal Example

```python
from gepa import optimize

# Define your seed candidate (dict of component_name -> text)
seed_candidate = {
    "agent_react_prompt": "...",
    "agent_error_message": "...",
}

# Define your dataset and adapter (see above)
trainset = [...]  # List of training inputs
valset = [...]    # List of validation inputs

# Import or construct a compatible LanguageModel object
from your_lm_provider import YourLLM
my_lm = YourLLM(...)

# Your GEPAAdapter implementation (see docs)
my_adapter = MyCustomGEPAAdapter(...)

# Run optimization
result = optimize(
    seed_candidate=seed_candidate,
    trainset=trainset,
    valset=valset,
    adapter=my_adapter,
    reflection_lm=my_lm,
    num_iters=50,                # Or max_metric_calls=1000
)
```

**See the `src/gepa/gepa.py` and `GEPAAdapter` docstrings for all options and details.**

---

## How Does GEPA Propose Improvements?

**Prompt Meta-Prompt Example**: When updating a text component, GEPA submits a prompt like:

```
I provided an assistant with the following instructions to perform a task:
<curr_instructions>

Here are some task inputs, system outputs, and feedback:
<inputs_outputs_feedback>

Your task is to write a new instruction, learning from the failures and aimed at solving the user's task precisely and robustly. Provide the new instruction within ``` blocks.
```

This allows the LLM to:
- Synthesize high-level domain lessons.
- Integrate niche, factual, and strategic details from observed failures.
- Avoid mistakes common to previous versions.

For code modules, domain-specific transformation, or response rewriters, similar reflection is used, powered by evaluation traces and failure analysis.

---

## Main Features

- **Language-driven reflection for targeted improvement** (LLM proposes new text based on trace and feedback).
- **Instance-level Pareto-aware candidate management**—systematically explores, tracks, and combines diverse winning strategies.
- **Rich trace-level feedback**—uses any structured textual feedback from system or evaluator for LLM context.
- **Interpretable, modular, and extensible**—swappable module and candidate selection strategies, merge/crossover, and reward aggregation.
- **Full training and inference-time optimization support.**
- **Compatible with open, proprietary, or local LLMs.**

---

## Advanced: Customization and Extension

- **Adapters:** Implement the `GEPAAdapter` interface to plug GEPA into new systems (see `src/gepa/gepa.py`).
- **Candidate and Module Selectors:** Swap out candidate or module selection logic to bias toward different exploration/exploitation balances.
- **Reflection Prompts:** Customize the prompt templates or output extractors for new component types or domains.
- **Feedback Engineering:** Maximize impact by designing evaluation metrics and feedback traces that provide the most actionable information.

---

## Reference & Citation

GEPA is described in:

> **GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning**  
> Lakshya A. Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J. Ryan, Meng Jiang, Christopher Potts, Koushik Sen, Alexandros G. Dimakis, Ion Stoica, Dan Klein, Matei Zaharia, Omar Khattab  
> [arXiv:2507.19457](https://arxiv.org/abs/2507.19457)

If you use this repository, or the GEPA algorithm, kindly cite:
```
@misc{agrawal2025gepareflectivepromptevolution,
      title={GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning}, 
      author={Lakshya A Agrawal and Shangyin Tan and Dilara Soylu and Noah Ziems and Rishi Khare and Krista Opsahl-Ong and Arnav Singhvi and Herumb Shandilya and Michael J Ryan and Meng Jiang and Christopher Potts and Koushik Sen and Alexandros G. Dimakis and Ion Stoica and Dan Klein and Matei Zaharia and Omar Khattab},
      year={2025},
      eprint={2507.19457},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.19457}, 
}
```

---