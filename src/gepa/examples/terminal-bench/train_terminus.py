import argparse
import json
import os
from pathlib import Path

import litellm
from terminal_bench.agents.terminus_1 import AgentResult, Chat, FailureMode, Terminus
from terminal_bench.dataset.dataset import Dataset
from terminal_bench.terminal.tmux_session import TmuxSession

from gepa import optimize
from gepa.adapters.terminal_bench_adapter.terminal_bench_adapter import (
    TerminalBenchTask,
    TerminusAdapter,
)

INSTRUCTION_PROMPT_PATH = Path(__file__).parent / "prompt-templates/instruction_prompt.txt"
II_PROMPT_PATH = Path(__file__).parent / "prompts" / "ii_system_prompt.txt"


def _load_env_file(env_path: str | None):
    if not env_path:
        return
    p = Path(env_path)
    if not p.exists():
        print(f"Warning: env file not found at {env_path}")
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()


class TerminusWrapper(Terminus):
    def __init__(
        self,
        model_name: str,
        max_episodes: int = 50,
        api_base: str | None = None,
        **kwargs,
    ):
        self.PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompt-templates/terminus.txt"
        self.instruction_prompt = INSTRUCTION_PROMPT_PATH.read_text()
        super().__init__(model_name, max_episodes, api_base, **kwargs)

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ):
        chat = Chat(self._llm)

        initial_prompt = self.instruction_prompt + self._prompt_template.format(
            response_schema=self._response_schema,
            instruction=instruction,
            history="",
            terminal_state=session.capture_pane(),
        )

        self._run_agent_loop(initial_prompt, session, chat, logging_dir)

        return AgentResult(
            total_input_tokens=chat.total_input_tokens,
            total_output_tokens=chat.total_output_tokens,
            failure_mode=FailureMode.NONE,
            timestamped_markers=self._timestamped_markers,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--n_concurrent", type=int, default=6)
    parser.add_argument("--env_file", type=str, default=str(Path(__file__).parent / ".env"))
    args = parser.parse_args()

    # Load .env (OpenAI-compatible base/key, default models)
    _load_env_file(args.env_file)
    model_name = os.environ.get("TASK_MODEL", args.model_name)
    n_concurrent = int(os.environ.get("TB_N_CONCURRENT", str(args.n_concurrent)))

    # Seed prompt: your II system prompt (can customize here)
    try:
        initial_prompt_from_terminus = II_PROMPT_PATH.read_text(encoding="utf-8")
    except Exception:
        # Fallback to the default instruction if II prompt missing
        initial_prompt_from_terminus = INSTRUCTION_PROMPT_PATH.read_text(encoding="utf-8")

    terminal_bench_dataset = Dataset(name="terminal-bench-core", version="head")
    terminal_bench_dataset.sort_by_duration()

    terminal_bench_tasks = terminal_bench_dataset._tasks[::-1]

    trainset = [
        TerminalBenchTask(task_id=task.name, model_name=model_name) for task in terminal_bench_tasks[45:50]
    ]
    valset = [TerminalBenchTask(task_id=task.name, model_name=model_name) for task in terminal_bench_tasks[:3]]

    testset = [
        TerminalBenchTask(task_id=task.name, model_name=model_name)
        for task in terminal_bench_tasks[50:52]
        if task.name != "chem-rf"
    ]

    reflection_lm_name = os.environ.get("REFLECTION_MODEL", "openai/gpt-5")
    # Use a plain OpenAI-compatible call for reflection (avoid provider-specific params)
    reflection_lm = (
        lambda prompt: litellm.completion(
            model=reflection_lm_name,
            messages=[{"role": "user", "content": prompt}],
        )
        .choices[0]
        .message.content
    )

    adapter = TerminusAdapter(n_concurrent=n_concurrent, instruction_prompt_path=INSTRUCTION_PROMPT_PATH)
    testset_results_no_prompt = adapter.evaluate(testset, {"instruction_prompt": ""}, capture_traces=True)
    testset_results_before_opt = adapter.evaluate(
        testset,
        {"instruction_prompt": initial_prompt_from_terminus},
        capture_traces=True,
    )

    # Ensure output directory exists (configurable via env GEPA_TB_RUN_DIR)
    out_dir = Path(os.environ.get("GEPA_TB_RUN_DIR", "gepa_terminus"))
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "testset_results_no_prompt.json", "w") as f:
        json.dump(
            {
                "score": sum(trajectory["success"] for trajectory in testset_results_no_prompt.trajectories),
                "trajectories": testset_results_no_prompt.trajectories,
            },
            f,
            indent=4,
        )
    with open(out_dir / "testset_results_before_opt.json", "w") as f:
        json.dump(
            {
                "score": sum(trajectory["success"] for trajectory in testset_results_before_opt.trajectories),
                "trajectories": testset_results_before_opt.trajectories,
            },
            f,
            indent=4,
        )

    optimized_results = optimize(
        seed_candidate={"instruction_prompt": initial_prompt_from_terminus},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        use_wandb=True,
        max_metric_calls=int(os.environ.get("GEPA_TB_MAX_METRIC_CALLS", "400")),
        reflection_minibatch_size=3,
        perfect_score=1,
        skip_perfect_score=False,
        run_dir=str(out_dir),
    )

    testset_results_after_opt = adapter.evaluate(
        testset,
        {"instruction_prompt": optimized_results.best_candidate["instruction_prompt"]},
        capture_traces=True,
    )

    with open(out_dir / "optimized_results.json", "w") as f:
        json.dump(
            {
                "score": sum(trajectory["success"] for trajectory in testset_results_after_opt.trajectories),
                "trajectories": testset_results_after_opt.trajectories,
            },
            f,
            indent=4,
        )
