"""
Collect TriviaQA answers + logprobs. 
"""

import json
from pathlib import Path
from itertools import islice
from datasets import load_dataset
from client import generate, judge

N = 100
SAVE_PATH = Path("data/results.json")

# Load previous progress
if SAVE_PATH.exists():
    with open(SAVE_PATH) as f:
        data = json.load(f)
    succ = data.get("succ", [])
    fail = data.get("fail", [])
    null = data.get("null", [])
else:
    succ, fail, null = [], [], []

completed = len(succ) + len(fail) + len(null)
print(f"Resuming from example {completed}")

dataset = load_dataset(
    "mandarjoshi/trivia_qa",
    "unfiltered",
    split="train",
    streaming=True,
)

# Skip already processed examples
dataset = islice(dataset, completed, N)

for question in dataset:
    msg = [{
        "role": "user",
        "content": question["question"],
    }]

    response = generate(msg)

    Q = question["question"]
    A = question["answer"]["aliases"]
    S = response["text"]

    result = judge(Q, A, S)

    response['question'] = Q
    response['answer'] = A

    if result == "correct":
        succ.append(response)
    elif result == "incorrect":
        fail.append(response)
    else:
        null.append(response)

    # Save checkpoint after every example
    with open(SAVE_PATH, "w") as f:
        json.dump(
            {
                "succ": succ,
                "fail": fail,
                "null": null,
            },
            f,
            indent=2,
        )

print("=" * 20)
print(
    f"N={N}, "
    f"Succ={len(succ)}, "
    f"Fail={len(fail)}, "
    f"Null={len(null)}"
)
print("=" * 20)