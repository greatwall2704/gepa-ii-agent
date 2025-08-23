import json


def test_aime_prompt_optimize():
    print("Running test_aime_prompt_optimize...")
    recorder_dir = "tests/test_aime_prompt_optimization"
    import os

    should_record = os.environ.get("RECORD_TESTS", "false").lower() == "true"
    if should_record:
        from gepa.adapters.default_adapter.default_adapter import DefaultAdapter
        import litellm
        cache = {}
        def task_lm(messages):
            hash_k = str(('task_lm', hash(str(messages))))
            if hash_k not in cache:
                cache[hash_k] = litellm.completion(model="openai/gpt-4.1-mini", messages=messages).choices[0].message.content.strip()
            return cache[hash_k]

        def reflection_lm(prompt):
            hash_k = str(('reflection_lm', hash(prompt)))
            if hash_k not in cache:
                cache[hash_k] = litellm.completion(model="openai/gpt-4.1", messages=[{"role": "user", "content": prompt}]).choices[0].message.content.strip()
            return cache[hash_k]

        adapter = DefaultAdapter(model=task_lm)
    else:
        with open(os.path.join(recorder_dir, "cache.json"), "r") as f:
            cache = json.load(f)
        def task_lm(messages):
            hash_k = str(('task_lm', hash(str(messages))))
            return cache[hash_k]
        def reflection_lm(prompt):
            hash_k = str(('reflection_lm', hash(prompt)))
            return cache[hash_k]
        adapter = DefaultAdapter(model=task_lm)

    import gepa

    # Load AIME dataset
    trainset, valset, _ = gepa.examples.aime.init_dataset()
    trainset = trainset[:10]
    valset = valset[3:8]

    seed_prompt = {
        "system_prompt": "You are a helpful assistant. You are given a question and you need to answer it. The answer should be given at the end of your response in exactly the format '### <final answer>'"
    }

    print("Running GEPA optimization process...")
    # Let's run GEPA optimization process.
    gepa_result = gepa.optimize(
        seed_candidate=seed_prompt,
        trainset=trainset, valset=valset,
        adapter=adapter,
        max_metric_calls=30,
        reflection_lm=reflection_lm,
    )

    # print("GEPA Optimized Prompt:", gepa_result.best_candidate['system_prompt'])
    if should_record:
        # save cache
        with open(os.path.join(recorder_dir, "cache.json"), "w") as f:
            json.dump(cache, f)
        with open(os.path.join(recorder_dir, "optimized_prompt.txt"), "w") as f:
            f.write(gepa_result.best_candidate['system_prompt'])
    else:
        # Read the optimized prompt from the file
        with open(os.path.join(recorder_dir, "optimized_prompt.txt"), "r") as f:
            optimized_prompt = f.read()
        assert optimized_prompt == gepa_result.best_candidate['system_prompt']
