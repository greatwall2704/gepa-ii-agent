import json
import os
from pathlib import Path

import pytest

# --- Constants and Setup ---
# Use pathlib for robust path handling relative to this test file.
# This assumes your test data will live in a sibling directory 'test_data'.
RECORDER_DIR = Path(__file__).parent

# --- Pytest Fixtures ---

@pytest.fixture(scope="module")
def recorder_dir() -> Path:
    """Provides the path to the recording directory and ensures it exists."""
    RECORDER_DIR.mkdir(parents=True, exist_ok=True)
    return RECORDER_DIR

@pytest.fixture(scope="module")
def mocked_lms(recorder_dir):
    """
    A pytest fixture to handle record/replay logic for LLM calls.

    In 'record' mode, it calls the actual LLM API and saves the results.
    In 'replay' mode (default), it loads results from a cached JSON file.
    """
    should_record = os.environ.get("RECORD_TESTS", "false").lower() == "true"
    cache_file = recorder_dir / "llm_cache.json"
    cache = {}

    def get_task_key(messages):
        """Creates a deterministic key from a list of message dicts."""
        # json.dumps with sort_keys=True provides a canonical representation.
        # The tuple prefix distinguishes it from reflection_lm calls.
        return str(("task_lm", json.dumps(messages, sort_keys=True)))

    def get_reflection_key(prompt):
        """Creates a deterministic key for the reflection prompt string."""
        return str(("reflection_lm", prompt))

    # --- Record Mode ---
    if should_record:
        # Lazy import litellm only when needed to avoid dependency in replay mode.
        import litellm

        print("\n--- Running in RECORD mode. Making live API calls. ---")

        def task_lm(messages):
            key = get_task_key(messages)
            if key not in cache:
                response = litellm.completion(
                    model="openai/gpt-4.1-nano",
                    messages=messages
                )
                cache[key] = response.choices[0].message.content.strip()
            return cache[key]

        def reflection_lm(prompt):
            key = get_reflection_key(prompt)
            if key not in cache:
                response = litellm.completion(
                    model="openai/gpt-4.1",
                    messages=[{"role": "user", "content": prompt}]
                )
                cache[key] = response.choices[0].message.content.strip()
            return cache[key]

        # Yield the live functions to the test, then save the cache on teardown.
        yield task_lm, reflection_lm

        print(f"--- Saving cache to {cache_file} ---")
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)

    # --- Replay Mode ---
    else:
        print("\n--- Running in REPLAY mode. Using cached API calls. ---")
        try:
            with open(cache_file) as f:
                cache = json.load(f)
        except FileNotFoundError:
            pytest.fail(
                f"Cache file not found: {cache_file}. "
                "Run with 'RECORD_TESTS=true pytest' to generate it."
            )

        def task_lm(messages):
            key = get_task_key(messages)
            if key not in cache:
                pytest.fail(f"Unseen input for task_lm in replay mode. Key: {key}")
            return cache[key]

        def reflection_lm(prompt):
            key = get_reflection_key(prompt)
            if key not in cache:
                pytest.fail(f"Unseen input for reflection_lm in replay mode. Key: {key}")
            return cache[key]

        yield task_lm, reflection_lm

# --- The Test Function ---

def test_aime_prompt_optimize(mocked_lms, recorder_dir):
    """
    Tests the GEPA optimization process using recorded/replayed LLM calls.
    """
    # Imports for the specific test logic
    import gepa
    from gepa.adapters.default_adapter.default_adapter import DefaultAdapter

    # 1. Setup: Unpack fixtures and load data
    task_lm, reflection_lm = mocked_lms
    adapter = DefaultAdapter(model=task_lm)

    print("Initializing AIME dataset...")
    trainset, valset, _ = gepa.examples.aime.init_dataset()
    trainset = trainset[:10]
    valset = valset[:10] #[3:8]

    seed_prompt = {
        "system_prompt": "You are a helpful assistant. You are given a question and you need to answer it. The answer should be given at the end of your response in exactly the format '### <final answer>'"
    }

    # 2. Execution: Run the core optimization logic
    print("Running GEPA optimization process...")
    gepa_result = gepa.optimize(
        seed_candidate=seed_prompt,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        max_metric_calls=30,
        reflection_lm=reflection_lm,
    )

    # 3. Assertion: Verify the result against the golden file
    optimized_prompt_file = recorder_dir / "optimized_prompt.txt"
    best_prompt = gepa_result.best_candidate["system_prompt"]

    # In record mode, we save the "golden" result
    if os.environ.get("RECORD_TESTS", "false").lower() == "true":
        print(f"--- Saving optimized prompt to {optimized_prompt_file} ---")
        with open(optimized_prompt_file, "w") as f:
            f.write(best_prompt)
        # Add a basic sanity check to ensure the process produced a valid output
        assert isinstance(best_prompt, str) and len(best_prompt) > 0

    # In replay mode, we assert against the golden result
    else:
        print(f"--- Asserting against golden file: {optimized_prompt_file} ---")
        with open(optimized_prompt_file) as f:
            expected_prompt = f.read()
        assert best_prompt == expected_prompt
