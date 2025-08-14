<p align="center">
  <img src="https://raw.githubusercontent.com/gepa-ai/gepa/refs/heads/main/assets/gepa_logo_with_text.svg" alt="GEPA Logo" width="500">
</p>

<h1 align="center">GEPA: System Optimization through Reflective Text Evolution</h1>

<p align="center">
  <em>Optimize text components—AI prompts, code, or instructions—of any system using reflective text evolution.</em>
</p>

## Overview

**GEPA** (Genetic-Pareto) offers a novel, sample-efficient framework for **optimizing arbitrary systems composed of text components**—such as the prompts of AI systems, code snippets/code files in a project, or other textual specifications—against any desired evaluation metric. GEPA uses **language models (LLMs) to reflect on the system's own behavior and outcomes, leveraging textual feedback from both execution and evaluation traces to guide strategic improvements.** Through iterative text mutation, reflection, and Pareto-aware candidate selection, GEPA efficiently discovers robust, high-performing variants of your system, *with minimal rollouts or evaluation calls*. GEPA can co-evolve multiple text-components belonging to the same system.

GEPA can **optimize any modular system that exposes text parameters** (instructions, prompts, code blocks, etc.), extracting maximal signal from every costly system execution, and producing domain-specific improvements.

> **The easiest and most powerful way to use GEPA is within [DSPy](https://dspy.ai/), where the GEPA algorithm is directly available through the `dspy.GEPA` API. If you use DSPy for building your AI systems, you already have access to GEPA without special integration.**

This repository provides the official implementation of the GEPA algorithm as proposed in the paper titled "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning" ([https://arxiv.org/abs/2507.19457](https://arxiv.org/abs/2507.19457)). In order to reproduce experiments from the GEPA paper, we provide a separate, [reproduction artifact](https://github.com/gepa-ai/gepa-artifact).

## Using GEPA

### The Easiest Path: [DSPy Integration](https://dspy.ai/)

We highly recommend using GEPA from within DSPy as [dspy.GEPA](https://dspy.ai/api/optimizers/GEPA/). If your AI system is built using [DSPy](https://dspy.ai/), GEPA is available as a plug-in optimizer. `dspy.GEPA` tutorials are available at [dspy.GEPA Tutorials](https://dspy.ai/tutorials/gepa_ai_program/).

### Simple Prompt Optimization Example (without DSPy)
GEPA is built around a flexible [`GEPAAdapter`](src/gepa/core/adapter.py) abstraction that lets it plug into any system and optimize different types of text snippets. 

In this example, we'll use GEPA to optimize a system prompt for math problems from the AIME dataset. Run the following in an environment with `OPENAI_API_KEY`:
```python
import gepa

# train/val/test sets are like [{"input": <question>, 'additional_context': {'solution': <hint to be used for optimization>}, 'answer': <gold answer>)}]
trainset, valset, _ = gepa.examples.aime.init_dataset()

seed_prompt = {
    "system_prompt": "You are a helpful assistant. You are given a question and you need to answer it. The answer should be given at the end of your response in exactly the format '### <final answer>'"
}

# GEPA can work to optimize any framework and system. Here, we use a simple adapter that evolves the system prompt
adapter = gepa.adapters.default_adapter.DefaultAdapter(model="openai/gpt-4.1-mini") 

# Let's run GEPA optimization process.
gepa_result = gepa.optimize(
    seed_candidate=seed_prompt,
    trainset=trainset,
    valset=valset,
    adapter=adapter,
    max_metric_calls=150, # <-- set a small budget of just 150 rollouts.
    reflection_lm="openai/gpt-5", # <-- Use a strong model to reflect on mistakes and propose better prompts
)

print("GEPA Optimized Prompt:", gepa_result.best_candidate['system_prompt'])
```

The above example used a simple [`DefaultAdapter`](src/gepa/adapters/default_adapter.py) that evolves system prompts, where the tasks are presented as user messages.

### Using GEPA to optimize _your_ system

GEPA can be used to optimize any system consisting of textual components. Follow these steps:
 - **Implement `GEPAAdapter`:** In order to allow the GEPA optimizer to pair with your system and its environment, users can implement the `GEPAAdapter` interface defined in [src/gepa/core/adapter.py](src/gepa/core/adapter.py). `GEPAAdapter` requires 2 methods:
    - Evaluate: Given a candidate consisting of proposed text components, and a minibatch of inputs sampled from the train/val sets, evaluate and return execution scores, also capturing the system traces.
    - Extract Traces for Reflection: Given the execution traces obtained from executing a proposed candidate, and a named component being optimized, return the textual content from the traces relevant to the named component.
- **Prepare trainset and valset:** Lists of example inputs and task metadata.
- **Call `gepa.optimize`** with your adapter, metric, and system configuration.

> We are actively working on implementing adapters to integrate into many different frameworks. Please open an issue if there's a specific framework you would like to see supported!

## Key Principles

GEPA is built on three fundamental ideas:

### 1. **Textual Reflection Instead of Blind Mutation**
Instead of black-box mutation or random evolution, GEPA uses LLMs to **reflect in natural language on the trajectories and outcomes of candidate systems**. This enables targeted, interpretable updates: the LLMs can diagnose failure points, understand context, leverage their vast world-knowledge priors and propose nuanced textual edits grounded in observed behavior.

### 2. **Rich Text Feedback as Optimization Signal**
GEPA can leverage *any* available textual feedback: not just execution logs from the system itself, but also rich traces from evaluation metrics (e.g., test-case logs, compiler error messages, profiler traces, etc.). This feedback becomes the input for LLM-based reflection, enabling **credit assignment and domain-aware optimization even for complex, multi-component systems**.

### 3. **Pareto-based Evolution of System Candidates**
GEPA **tracks and samples candidates for mutation from a Pareto frontier of high-performing candidates across all evaluation instances**. This preserves solution diversity, accumulates complementary strategies, and avoids premature convergence—allowing GEPA to stochastically combine and evolve candidates that individually win on different instances.

## What Can GEPA Optimize?

- **Compound AI Systems:** Multi-stage and multi-module LLM systems with orchestrated control flow (for example, DSPy, LangChain, etc.)
- **Agents or tools with text instructions:** Any system where text determines behavior (e.g., prompt sets, user guides, modular agent instructions).
- **Code:** GEPA can be leveraged to evolve critical code snippets against performance and correctness metrics.
- **Any system whose key behaviors are controlled by editable textual components.**

**GEPA is model- and metric-agnostic:** supply any callable system, any evaluation function, and an LLM for reflection.

## Main Features

- **Language-driven reflection for targeted improvement** (LLM proposes new text based on trace and feedback).
- **Instance-level Pareto-aware candidate management**—systematically explores, tracks, and combines diverse winning strategies.
- **Rich trace-level feedback**—uses any structured textual feedback from system or evaluator for LLM context.
- **Interpretable, modular, and extensible**—swappable module and candidate selection strategies, merge/crossover, and reward aggregation.
- **Full training and inference-time optimization support.**
- **Compatible with open, proprietary, or local LLMs.**

# Contributions

We encourage the community and users to help us develop adapters to allow GEPA to be used for optimizing all kinds of systems leveraging textual components. Refer to [DSPy/GEPAAdapter](https://github.com/stanfordnlp/dspy/tree/main/dspy/teleprompt/gepa/gepa_utils.py) and [src/gepa/adapters/](src/gepa/adapters/) for example `GEPAAdapter` implementations.

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