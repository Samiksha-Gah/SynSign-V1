# hf_zephyr_new_examples.py

import os
import time
import requests
import pandas as pd

# ─── CONFIG ──────────────────────────────────────────────────────────────────────

MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
API_URL   = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

# System‐role message (ChatML)
SYSTEM_PROMPT = """\
<|im_start|>system
You are an ASL gloss translator. Given exactly one ASL gloss sentence (all uppercase), output only its simple, present‐tense English translation—no extra commentary or new sentences.
<|im_end|>
"""

# Three few‐shot pairs, each as a separate user→assistant turn
FEW_SHOT_CHAT = [
    # Example 1
    (
        "<|im_start|>user\nGloss: FOOTBALL YOU LIKE\n<|im_end|>\n"
        "<|im_start|>assistant\nYou like football.\n<|im_end|>\n"
    ),
    # Example 2
    (
        "<|im_start|>user\nGloss: ME NAME WHAT\n<|im_end|>\n"
        "<|im_start|>assistant\nWhat is my name?\n<|im_end|>\n"
    ),
    # Example 3
    (
        "<|im_start|>user\nGloss: I HAVE DOG NAME R E X\n<|im_end|>\n"
        "<|im_start|>assistant\nI have a dog named Rex.\n<|im_end|>\n"
    ),
]

# The new glosses you asked to try
sample_glosses = [
    "HI MY NAME SAM",            # “Hi, my name is Sam.”
    "YOUR NAME WHAT ?",          # “What is your name?”
    "I LIKE PIZZA AND SUSHI",    # “I like pizza and sushi.”
    "YOU GO STORE TODAY ?",      # “Are you going to the store today?”
    "BATHROOM WHERE ?",          # “Where is the restroom?”
    "I NOT UNDERSTAND",          # “I don’t understand.”
    "JOHN MY FRIEND",            # “John is my friend.”
    "I LIVE NEW YORK",           # “I live in New York.”
    "SHE TEACHER",               # “She is a teacher.”
    "TIME WHAT ?",               # “What time is it?”
]

# ─── MAIN ─────────────────────────────────────────────────────────────────────────

def main():
    # 1) Ensure HF_API_TOKEN is set
    hf_token = os.getenv("HF_API_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_API_TOKEN is not set.\n"
            "Run:\n\n"
            "  export HF_API_TOKEN=\"hf_your_token_here\"\n\n"
            "and then rerun this script."
        )

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }

    results = []
    for gloss in sample_glosses:
        # 2) Build ChatML: system + few-shot examples + new user turn
        chatml = SYSTEM_PROMPT
        for pair in FEW_SHOT_CHAT:
            chatml += pair
        chatml += (
            "<|im_start|>user\n"
            f"Gloss: {gloss}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        # 3) Call Zephyr with a stop sequence to force a single sentence
        payload = {
            "inputs": chatml,
            "parameters": {
                "max_new_tokens": 64,
                "temperature": 0.1,          # must be > 0 for Zephyr
                "stop": ["<|im_end|>"]       # halt generation at end of assistant turn
            }
        }

        t0 = time.perf_counter()
        resp = requests.post(API_URL, headers=headers, json=payload)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000

        if resp.status_code != 200:
            raise RuntimeError(f"Error {resp.status_code}: {resp.text}")

        data = resp.json()
        # Zephyr returns a list of {"generated_text": "..."}
        if not isinstance(data, list) or "generated_text" not in data[0]:
            raise RuntimeError(f"Unexpected response format: {data}")

        full_output = data[0]["generated_text"]

        # 4) Extract only the assistant’s single translation:
        #    - Split at "<|im_start|>assistant" to get the answer block
        #    - Then split off at "<|im_end|>"
        if "<|im_start|>assistant" in full_output:
            answer_block = full_output.split("<|im_start|>assistant")[-1]
        else:
            answer_block = full_output
        answer = answer_block.split("<|im_end|>")[0].strip()
        answer = answer.split("\n")[0].strip()

        results.append({
            "Gloss":       gloss,
            "Translation": answer,
            "Time (ms)":   round(elapsed_ms, 2),
        })

    # 5) Print and save results
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv("zephyr_new_examples_results.csv", index=False)
    print("\nResults saved to zephyr_new_examples_results.csv")

if __name__ == "__main__":
    main()
