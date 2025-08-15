import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import litellm
from pydantic import BaseModel
from terminal_bench.agents.terminus_1 import Terminus, CommandBatchResponse
from terminal_bench.agents.terminus_1 import AgentResult, FailureMode
from terminal_bench.agents.terminus_1 import Chat
from terminal_bench.terminal.tmux_session import TmuxSession
from terminal_bench.dataset.dataset import Dataset

from gepa import GEPAAdapter, optimize, EvaluationBatch


class TerminalBenchTask(BaseModel):
    task_id: str
    model_name: str


INSTRUCTION_PROMPT_PATH = (
    Path(__file__).parent / "prompt-templates/instruction_prompt.txt"
)


class TerminusWrapper(Terminus):
    def __init__(
        self,
        model_name: str,
        max_episodes: int = 50,
        api_base: str | None = None,
        **kwargs,
    ):
        self.PROMPT_TEMPLATE_PATH = (
            Path(__file__).parent / "prompt-templates/terminus.txt"
        )
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


def run_agent_tb(
    task_ids: str | list[str],
    run_id: str,
    model_name: str,
    instruction_prompt: str,
    dataset_name: str = "terminal-bench-core",
    dataset_version: str = "head",
    agent_import_path: str = "train_terminus:TerminusWrapper",
    n_concurrent: int = 6,
):
    """Run the replay agent for multiple task IDs using tb run command."""

    env = os.environ.copy()
    # write instruction prompt to file
    with open("prompt-templates/instruction_prompt.txt", "w") as f:
        f.write(instruction_prompt)

    cmd = [
        "tb",
        "run",
        "--dataset-name",
        dataset_name,
        "--dataset-version",
        dataset_version,
        "--agent-import-path",
        agent_import_path,
        "--model-name",
        model_name,
        "--run-id",
        run_id,
        "--n-concurrent",
        str(n_concurrent),
    ]
    if isinstance(task_ids, list):
        for task_id in task_ids:
            cmd.extend(["--task-id", task_id])
    else:
        cmd.extend(["--task-id", task_ids])

    print(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, env=env, cwd=Path(__file__).parent, check=True)
        print(f"Command completed successfully with return code: {result.returncode}")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code: {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"Error running command: {e}")
        return 1


def get_results(task_id: str, run_id: str) -> tuple[int, list]:

    def _read_episode_response(episode_dir: Path) -> CommandBatchResponse | None:
        """Helper method to read and parse response.json from an episode directory."""
        response_file = episode_dir / "response.json"
        if response_file.exists():
            try:
                response_content = response_file.read_text()
                return CommandBatchResponse.model_validate_json(response_content)
            except Exception:
                pass
        return None

    def _get_logging_dir(task_id: str, run_id: str):
        logging_dir_base = Path("runs") / run_id / task_id
        for dir in logging_dir_base.iterdir():
            if dir.is_dir() and dir.name.startswith(task_id):
                return dir
        raise ValueError(
            f"No logging directory found for task {task_id} and run {run_id}"
        )

    logging_dir = _get_logging_dir(task_id, run_id)
    result_json = logging_dir / "results.json"
    with open(result_json, "r") as f:
        result = json.load(f)
    if result.get("parser_results", None):
        score = sum(map(lambda x: x == "passed", result["parser_results"].values()))
    else:
        score = 0

    if result.get("is_resolved", None):
        success = True
    else:
        success = False

    failed_reason = result.get("failure_mode", "unknown")

    trajectory_path = logging_dir / "agent-logs"
    episode_dirs = []
    for dir in trajectory_path.iterdir():
        if dir.is_dir() and dir.name.startswith("episode-"):
            episode_dirs.append(dir)

    if episode_dirs:
        # Sort by episode number to get the last one
        episode_dirs.sort(key=lambda x: int(x.name.split("-")[1]))
        last_episode_dir = episode_dirs[-1]

    last_episode_dir_trajectory = last_episode_dir / "debug.json"
    with open(last_episode_dir_trajectory, "r") as f:
        trajectory = json.load(f)

        if "input" in trajectory and isinstance(trajectory["input"], list):
            messages = trajectory["input"]

        # Add the last assistant response using helper method
        parsed_response = _read_episode_response(last_episode_dir)

        if parsed_response:
            assistant_message = {
                "role": "assistant",
                "content": parsed_response.model_dump_json(),
            }
            messages.append(assistant_message)

    return success, score, failed_reason, messages


class TerminusAdapter(GEPAAdapter):

    def __init__(self, n_concurrent: int = 6):
        self.n_concurrent = n_concurrent

    def evaluate(
        self,
        batch: list[TerminalBenchTask],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        outputs = []
        scores = []
        trajectories = []
        example_run_id = "temp_gepa_run" + "_" + datetime.now().strftime("%Y%m%d%H%M%S")
        example_model_name = batch[0].model_name

        run_agent_tb(
            [task.task_id for task in batch],
            example_run_id,
            example_model_name,
            instruction_prompt=candidate["instruction_prompt"],
            n_concurrent=self.n_concurrent,
        )

        for example in batch:
            try:
                success, score, failed_reason, messages = get_results(
                    example.task_id, example_run_id
                )
            except Exception as e:
                print(f"Error running example {example.task_id} {example_run_id}: {e}")
                success = False
                score = 0
                failed_reason = str(e)
                messages = []

            outputs.append(
                f"Terminal Bench outputs are omitted. Please see runs/{example_run_id}/{example.task_id}/ for detailed logging."
            )
            scores.append(score)
            trajectories.append(
                {
                    "messages": messages,
                    "instruction_prompt": candidate["instruction_prompt"],
                    "failed_reason": failed_reason,
                    "success": success,
                }
            )
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ):
        reflective_dataset = {"instruction_prompt": []}
        for score, trajectory in zip(eval_batch.scores, eval_batch.trajectories):
            if trajectory["success"]:
                feedback = f"Successfully solved the task!"
            else:
                feedback = (
                    f"Failed to solve the task. Reason: {trajectory['failed_reason']}"
                )
            reflective_dataset["instruction_prompt"].append(
                {
                    "Message History": trajectory["messages"],
                    "Instruction Prompt": candidate["instruction_prompt"],
                    "Feedback": feedback,
                }
            )
        return reflective_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--n_concurrent", type=int, default=6)
    args = parser.parse_args()

    initial_prompt_from_terminus = """
You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task instruction and the output from previously executed commands. Your goal is to solve the task by providing batches of shell commands.

For each response:
1. Analyze the current state based on any terminal output provided
2. Determine the next set of commands needed to make progress
3. Decide if you need to see the output of these commands before proceeding

Don't include markdown formatting.

Note that you operate directly on the terminal from inside a tmux session. Use tmux keystrokes like `C-x` or `Escape` to interactively navigate the terminal. If you would like to execute a command that you have written you will need to append a newline character to the end of your command.

For example, if you write "ls -la" you will need to append a newline character to the end of your command like this: `ls -la\n`.

One thing to be very careful about is handling interactive sessions like less, vim, or git diff. In these cases, you should not wait for the output of the command. Instead, you should send the keystrokes to the terminal as if you were typing them.
"""

    terminal_bench_dataset = Dataset(name="terminal-bench-core", version="head")
    terminal_bench_dataset.sort_by_duration()

    terminal_bench_tasks = terminal_bench_dataset._tasks[::-1]
    trainset = [
        TerminalBenchTask(task_id=task.name, model_name=args.model_name)
        for task in terminal_bench_tasks[30:50]
    ]
    valset = [
        TerminalBenchTask(task_id=task.name, model_name=args.model_name)
        for task in terminal_bench_tasks[:30]
    ]

    reflection_lm_name = "openai/gpt-5"
    reflection_lm = (
        lambda prompt: litellm.completion(
            model=reflection_lm_name,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="medium",
        )
        .choices[0]
        .message.content
    )

    optimized_results = optimize(
        seed_candidate={"instruction_prompt": initial_prompt_from_terminus},
        trainset=trainset,
        valset=valset,
        adapter=TerminusAdapter(n_concurrent=args.n_concurrent),
        reflection_lm=reflection_lm,
        use_wandb=True,
        max_metric_calls=400,
        reflection_minibatch_size=3,
        perfect_score=1,
        skip_perfect_score=False,
    )

    print(optimized_results.to_dict())
    # save to json
    with open(
        f"optimized_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.json", "w"
    ) as f:
        json.dump(optimized_results.to_dict(), f, indent=4)
