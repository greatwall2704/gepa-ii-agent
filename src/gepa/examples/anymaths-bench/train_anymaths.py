def init_dataset(anymaths_dset_name: str = "openai/gsm8k"):
    import random

    from datasets import load_dataset

    train_split = []
    test_split = []
    match anymaths_dset_name:
        case "openai/gsm8k":
            train_load_dataset = load_dataset(anymaths_dset_name, "main", split="train")
            for item in train_load_dataset:
                answer = item["answer"].split("####")[-1].strip()
                solution = item["answer"].split("####")[0].strip()
                question = item["question"]

                train_split.append({"input": question, "additional_context": {"solution": solution}, "answer": answer})

            random.Random(0).shuffle(train_split)

            test_load_dataset = load_dataset(anymaths_dset_name, "main", split="test")
            for item in test_load_dataset:
                answer = item["answer"].split("####")[-1].strip()
                solution = item["answer"].split("####")[0].strip()
                question = item["question"]

                test_split.append({"input": question, "answer": answer})

        case "MathArena/aime_2025":
            train_load_dataset = load_dataset("AI-MO/aimo-validation-aime", "default", split="train")
            for item in train_load_dataset:
                question = item["problem"]
                solution = item["solution"]
                answer = item["answer"]

                train_split.append({"input": question, "additional_context": {"solution": solution}, "answer": answer})

            random.Random(0).shuffle(train_split)

            test_load_dataset = load_dataset("MathArena/aime_2025", "default", split="train")
            for item in test_load_dataset:
                question = item["problem"]
                answer = item["answer"]

                test_split.append({"input": question, "answer": answer})

    trainset = train_split[: len(train_split) // 2]
    valset = train_split[len(train_split) // 2 :]
    testset = test_split * 5

    return trainset, valset, testset


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import litellm

    from gepa import optimize
    from gepa.adapters.anymaths_adapter import AnyMathsAdapter

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ollama/qwen3:4b")  # only Ollama models are supported
    parser.add_argument("--api_base", type=str, default="http://localhost:11434")
    parser.add_argument("--max_litellm_workers", type=int, default=10)
    parser.add_argument("--anymaths_dset_name", type=str, default="openai/gsm8k")
    parser.add_argument("--budget", type=int, default=5, help="The budget for the optimization process.")
    parser.add_argument(
        "--reflection_lm", type=str, default="ollama/qwen3:8b", help="The name of the reflection LM to use."
    )
    parser.add_argument(
        "--reflection_minibatch_size", type=int, default=3, help="The size of the minibatch for the reflection LM."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="The seed for the random number generator for reproducibility."
    )
    args = parser.parse_args()

    INSTRUCTION_PROMPT_PATH = Path(__file__).parent / "prompt-templates/instruction_prompt.txt"

    seed_instruction = INSTRUCTION_PROMPT_PATH.read_text()

    trainset, valset, testset = init_dataset(args.anymaths_dset_name)

    trainset = trainset[:50]  # Limit to 50 samples for demo purposes
    valset = valset[:50]  # Limit to 50 samples for demo purposes
    testset = testset[:50]  # Limit to 50 samples for demo purposes

    print(f"Train set size: {len(trainset)}")
    print(f"Validation set size: {len(valset)}")
    print(f"Test set size: {len(testset)}")

    reflection_lm_name = args.reflection_lm

    def reflection_lm(prompt: str):
        """Call the reflection language model with the given prompt and return its content string."""
        response = litellm.completion(
            model=reflection_lm_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    adapter_model = args.model_name
    api_base = args.api_base
    max_litellm_workers = args.max_litellm_workers
    budget = args.budget
    reflection_minibatch_size = args.reflection_minibatch_size
    seed = args.seed

    optimized_results = optimize(
        seed_candidate={"instruction_prompt": seed_instruction},
        trainset=trainset,
        valset=valset,
        adapter=AnyMathsAdapter(model=adapter_model, api_base=api_base, max_litellm_workers=max_litellm_workers),
        reflection_lm=reflection_lm,
        reflection_minibatch_size=reflection_minibatch_size,
        perfect_score=1,
        skip_perfect_score=False,
        use_wandb=False,
        num_iters=budget,
        seed=seed,
    )

    print(optimized_results.to_dict())
