import math
import os
import threading
import time
import re

from openai import OpenAI

MODEL = "meta-llama/llama-3.1-8b-instruct"
_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

_usage_lock = threading.Lock()
USAGE = {"prompt": 0, "completion": 0, "calls": 0, "failures": 0}

def generate(messages, *, model=MODEL, temperature=0.7, max_tokens=160, retries=4):
    """Returns {text, tokens, surprisal, entropy} for one generation.

    surprisal[i] = -log p(token_i) in nats (the pseudo-NLL series);
    entropy[i]   = truncated Shannon entropy over the top-20 alternatives.
    """
    extra_body = {"provider": {"require_parameters": True}}
    if "qwen3.5" in model or "thinking" in model:
        extra_body["reasoning"] = {"enabled": False}

    last_err = None
    for attempt in range(retries):
        try:
            r = _client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                logprobs=True,
                top_logprobs=20,
                extra_body=extra_body,
            )
            choice = r.choices[0]
            if choice.logprobs is None or not choice.logprobs.content:
                raise RuntimeError("provider returned no logprobs")
            with _usage_lock:
                USAGE["prompt"] += r.usage.prompt_tokens
                USAGE["completion"] += r.usage.completion_tokens
                USAGE["calls"] += 1
            content = choice.logprobs.content
            return {
                "text": choice.message.content,
                "tokens": [t.token for t in content],
                "surprisal": [-t.logprob for t in content],
                "entropy": [
                    -sum(math.exp(tl.logprob) * tl.logprob for tl in t.top_logprobs)
                    for t in content
                ],
            }
        except Exception as e:  # noqa: BLE001 — transient provider errors dominate
            last_err = e
            time.sleep(1.5 * 2**attempt)
    with _usage_lock:
        USAGE["failures"] += 1
    raise RuntimeError(f"generate failed after {retries} attempts: {last_err}")

def _judge(_Q, A, S):
    response = S.lower()

    m = re.search(r"Answer:\s*(.+)$", response, re.MULTILINE)

    if m:
        final_answer = m.group(1).strip()
    else:
        final_answer = None

    if final_answer == "null":
        return "null"

    aliases = [alias.lower().strip() for alias in A]

    if final_answer in aliases:
        result = "correct"
    else:
        result = "incorrect"


def judge(question, answer, response, judge="meta-llama/llama-3.3-70b-instruct"):

    messages = [
        {
            "role": "system",
            "content": """
    You are grading factual QA answers.

    Return exactly one token:

    correct
    incorrect
    null

    Rules:

    - correct:
    - the response ultimately gives the correct answer.
    - reasoning before or after the answer does not matter.
    - self-corrections count. If the model first proposes a wrong answer but then changes to the correct one, return correct.
    - minor spelling differences and aliases count as correct.

    - incorrect:
    - the final answer is wrong.
    - the response clearly commits to an incorrect entity.

    - null:
    - the model refuses.
    - "I don't know."
    - "Not enough information."
    - no answer is attempted.

    Only judge the final answer, not intermediate reasoning.

    Reply with exactly one word.
    """
        },
        {
            "role": "user",
            "content": f"""
    Question:
    {question}

    Ground truth aliases:
    {", ".join(answer)}

    Model response:
    {response}
    """
        }
    ]

    r = _client.chat.completions.create(
        model=judge,
        messages=messages,
        temperature=0,
        max_tokens=256,
    )
    choice = r.choices[0]

    return choice.message.content