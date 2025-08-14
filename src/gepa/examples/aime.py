def init_dataset():
    from datasets import load_dataset
    import random

    train_split = [{"input": x['problem'], 'additional_context': {'solution': x['solution']}, 'answer': "### " + str(x['answer'])} for x in load_dataset("AI-MO/aimo-validation-aime")['train']]
    random.Random(0).shuffle(train_split)
    test_split = [{"input": x['problem'], 'answer': "### " + str(x['answer'])} for x in load_dataset("MathArena/aime_2025")['train']]

    trainset = train_split[:len(train_split)//2]
    valset = train_split[len(train_split)//2:]
    testset = test_split * 5

    return trainset, valset, testset
