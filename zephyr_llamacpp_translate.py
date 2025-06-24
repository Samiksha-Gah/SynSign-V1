# zephyr_llamacpp_translate.py

import os
import time
import pandas as pd
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# ─── 1) DOWNLOAD (OR LOCATE) THE Q4_K_M GGUF ──────────────────────────────────────

# We explicitly request "zephyr-7b-beta.Q4_K_M.gguf" from TheBloke/zephyr-7B-beta-GGUF
gguf_path = hf_hub_download(
    repo_id="TheBloke/zephyr-7B-beta-GGUF",
    filename="zephyr-7b-beta.Q4_K_M.gguf",
    repo_type="model"
)
print("Quantized (Q4_K_M) model path:", gguf_path)

# ─── 2) LOAD THE MODEL VIA LLAMA.CPP ──────────────────────────────────────────────

print("Loading Zephyr-7B-β (Q4_K_M) into llama.cpp…")
llm = Llama(
    model_path=gguf_path,
    n_threads=os.cpu_count(),  # use all available CPU cores
)
print("Model loaded!\n")

# ─── 3) SET UP PROMPT TEMPLATE AND GLOSSES ───────────────────────────────────────

PROMPT_TEMPLATE = (
    "Translate the following ASL gloss into simple, present‐tense English.\n"
    "Gloss: {gloss}\n"
    "Answer:"
)

sample_glosses = [
    "HI MY NAME SAM",
    "YOUR NAME WHAT ?",
    "I LIKE PIZZA AND SUSHI",
    "YOU GO STORE TODAY ?",
    "BATHROOM WHERE ?",
    "I NOT UNDERSTAND",
    "JOHN MY FRIEND",
    "I LIVE NEW YORK",
    "SHE TEACHER",
    "TIME WHAT ?",
]

results = []

# ─── 4) RUN A LOOP OVER ALL GLOSSES ────────────────────────────────────────────────

for gloss in sample_glosses:
    prompt = PROMPT_TEMPLATE.format(gloss=gloss)

    t0 = time.perf_counter()
    resp = llm.create_completion(
        prompt,
        max_tokens=16,      # generate up to 32 tokens
        temperature=0.1,    # small randomness
        stop=["\n"],        # stop at newline
    )
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000

    # Extract just the first line of output (before any newline)
    text = resp["choices"][0]["text"].strip().split("\n")[0].strip()

    results.append({
        "Gloss":       gloss,
        "Translation": text,
        "Time (ms)":   round(elapsed_ms, 2),
    })

# ─── 5) DISPLAY & SAVE RESULTS ────────────────────────────────────────────────────

df = pd.DataFrame(results)
print("\n" + df.to_string(index=False))
df.to_csv("zephyr_llamacpp_translations.csv", index=False)
print("\nResults saved to zephyr_llamacpp_translations.csv")
