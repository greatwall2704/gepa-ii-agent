from pydantic import BaseModel, Field


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
    testset = test_split

    return trainset, valset, testset


class AnyMathsStructuredOutput(BaseModel):
    final_answer: str = Field(
        ..., description="The final answer to the mathematical problem (i.e., no units, no other text)"
    )
    solution_pad: str = Field(..., description="The solution pad containing the step-by-step solution to the problem.")


if __name__ == "__main__":
    import ast
    from pathlib import Path

    import litellm
    from tqdm import tqdm

    # Evaluate qwen3:4b on test set
    trainset, valset, testset = init_dataset("openai/gsm8k")

    # Get seed prompt
    INSTRUCTION_PROMPT_PATH = Path(__file__).parent / "prompt-templates/optimal_prompt.txt"

    seed_instruction = INSTRUCTION_PROMPT_PATH.read_text()

    batched_testset = []
    batch_size = 8
    for i in range(0, len(testset), batch_size):
        batched_testset.append(testset[i : i + batch_size])

    total_score = 0.0
    with tqdm(total=len(testset), desc="Evaluating") as pbar:
        for batch in batched_testset:
            litellm_requests = []
            for item in batch:
                user_content = f"{item['input']}"
                messages = [{"role": "system", "content": seed_instruction}, {"role": "user", "content": user_content}]

                litellm_requests.append(messages)

            try:
                responses = litellm.batch_completion(
                    model="ollama/qwen3:4b",
                    messages=litellm_requests,
                    api_base="http://localhost:11434",
                    max_workers=4,
                    format=AnyMathsStructuredOutput.model_json_schema(),
                )
            except Exception as e:
                raise e

            for response, item in zip(responses, batch, strict=False):
                correct_output_format = True
                try:
                    assistant_response = ast.literal_eval(response.choices[0].message.content.strip())
                    assistant_final_answer = assistant_response["final_answer"]
                    ground_truth = item["answer"]
                    score = 1.0 if ground_truth in assistant_final_answer else 0.0
                    total_score += score
                except Exception:
                    correct_output_format = False
                    continue
            pbar.update(len(batch))
            pbar.set_postfix({"Score": f"{total_score} / {len(testset):.4f}"})

    print(f"Final score >> {total_score} / {len(testset):.4f}")
