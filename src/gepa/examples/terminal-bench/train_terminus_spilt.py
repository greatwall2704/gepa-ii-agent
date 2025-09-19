import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import litellm
from terminal_bench.dataset.dataset import Dataset

from gepa import optimize
from gepa.adapters.terminal_bench_adapter.terminal_bench_adapter import (
    TerminalBenchTask,
    TerminusAdapter,
)

# Drop unsupported params for OpenAI-compatible backends
litellm.drop_params = True  # type: ignore[attr-defined]

# Resolve paths relative to this file
HERE = Path(__file__).parent
INSTRUCTION_PROMPT_PATH = HERE / "prompt-templates" / "instruction_prompt.txt"
II_PROMPT_PATH = HERE / "prompts" / "ii_system_prompt.txt"


def _load_env_file(env_path: Optional[str]) -> None:
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


def _parse_slice(s: str) -> slice:
    if ":" not in s:
        raise ValueError("Slice must be in 'start:end' form")
    a, b = s.split(":", 1)
    start = int(a) if a else None
    end = int(b) if b else None
    return slice(start, end)


@dataclass
class SplitConfig:
    train_slice: slice
    val_slice: slice
    test_slice: slice


def _select_tasks(dataset: Dataset, split: SplitConfig, model_name: str) -> Tuple[List[TerminalBenchTask], List[TerminalBenchTask], List[TerminalBenchTask]]:
    dataset.sort_by_duration()
    all_tasks = dataset._tasks[::-1]

    def make(tasks):
        return [
            TerminalBenchTask(task_id=t.name, model_name=model_name)
            for t in tasks
            if t.name != "chem-rf"
        ]

    train = make(all_tasks[split.train_slice])
    val = make(all_tasks[split.val_slice])
    test = make(all_tasks[split.test_slice])
    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/Val vs Test split runner for GEPA+Terminus")
    parser.add_argument("--env_file", type=str, default=str(HERE / ".env"))
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--reflection_model", type=str, default=None)
    parser.add_argument("--n_concurrent", type=int, default=6)
    parser.add_argument("--train_slice", type=str, default="45:50")
    parser.add_argument("--val_slice", type=str, default=":3")
    parser.add_argument("--test_slice", type=str, default="50:52")
    parser.add_argument("--run_dir", type=str, default="gepa_terminus_split")
    parser.add_argument("--max_metric_calls", type=int, default=None)
    parser.add_argument("--minibatch", type=int, default=3)
    parser.add_argument("--skip_perfect_tasks", action="store_true", default=True)
    args = parser.parse_args()

    _load_env_file(args.env_file)
    model_name = os.environ.get("TASK_MODEL", args.model_name)
    reflection_model = args.reflection_model or os.environ.get("REFLECTION_MODEL", model_name)
    n_concurrent = int(os.environ.get("TB_N_CONCURRENT", str(args.n_concurrent)))
    max_metric_calls_env = os.environ.get("GEPA_TB_MAX_METRIC_CALLS", "400")
    max_metric_calls = args.max_metric_calls if args.max_metric_calls is not None else int(max_metric_calls_env)

    try:
        seed_prompt = II_PROMPT_PATH.read_text(encoding="utf-8")
    except Exception:
        seed_prompt = INSTRUCTION_PROMPT_PATH.read_text(encoding="utf-8")

    out_dir = Path(args.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = Dataset(name="terminal-bench-core", version="head")
    split = SplitConfig(
        train_slice=_parse_slice(args.train_slice),
        val_slice=_parse_slice(args.val_slice),
        test_slice=_parse_slice(args.test_slice),
    )
    trainset, valset, testset = _select_tasks(ds, split, model_name)

    adapter = TerminusAdapter(
        n_concurrent=n_concurrent,
        instruction_prompt_path=INSTRUCTION_PROMPT_PATH,
    )

    def reflection_lm(prompt: str) -> str:
        resp = litellm.completion(
            model=reflection_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content  # type: ignore[attr-defined]

    # Baseline test
    base_empty = adapter.evaluate(testset, {"instruction_prompt": ""}, capture_traces=True)
    (out_dir / "test_baseline_empty.json").write_text(
        json.dumps(
            {
                "score": sum(t["success"] for t in base_empty.trajectories),
                "trajectories": base_empty.trajectories,
            },
            indent=4,
        ),
        encoding="utf-8",
    )

    base_seed = adapter.evaluate(testset, {"instruction_prompt": seed_prompt}, capture_traces=True)
    (out_dir / "test_baseline_seed.json").write_text(
        json.dumps(
            {
                "score": sum(t["success"] for t in base_seed.trajectories),
                "trajectories": base_seed.trajectories,
            },
            indent=4,
        ),
        encoding="utf-8",
    )

    # Filter train tasks already perfect with seed
    if args.skip_perfect_tasks:
        probe = adapter.evaluate(trainset, {"instruction_prompt": seed_prompt}, capture_traces=True)
        kept: List[TerminalBenchTask] = []
        for task, traj in zip(trainset, probe.trajectories):
            if not traj["success"]:
                kept.append(task)
        if kept:
            trainset = kept
        (out_dir / "train_filtered_summary.json").write_text(
            json.dumps(
                {
                    "kept": [t.task_id for t in trainset],
                },
                indent=4,
            ),
            encoding="utf-8",
        )

    result = optimize(
        seed_candidate={"instruction_prompt": seed_prompt},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        use_wandb=False,
        max_metric_calls=max_metric_calls,
        reflection_minibatch_size=args.minibatch,
        skip_perfect_score=True,
        run_dir=str(out_dir),
    )

    optimized_prompt = result.best_candidate["instruction_prompt"]
    (out_dir / "optimized_prompt.txt").write_text(optimized_prompt, encoding="utf-8")

    after = adapter.evaluate(testset, {"instruction_prompt": optimized_prompt}, capture_traces=True)
    (out_dir / "test_after_opt.json").write_text(
        json.dumps(
            {
                "score": sum(t["success"] for t in after.trajectories),
                "trajectories": after.trajectories,
            },
            indent=4,
        ),
        encoding="utf-8",
    )

    print("\n=== Comparison (Test set) ===")
    print("Empty prompt success count:", sum(t["success"] for t in base_empty.trajectories))
    print("Seed prompt success count:", sum(t["success"] for t in base_seed.trajectories))
    print("Optimized prompt success count:", sum(t["success"] for t in after.trajectories))


if __name__ == "__main__":
    main()
