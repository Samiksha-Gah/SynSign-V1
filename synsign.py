# synsign.py

import os
import time
import requests
import cv2
import numpy as np
import torch
import pandas as pd
import mediapipe as mp

from pose_models.st_gcn import STGCN
from pose_models.fc import FC
from pose_models.network import Network

# ─── CONFIG: Zephyr API ──────────────────────────────────────────────────────────

MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
API_URL   = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

SYSTEM_PROMPT = """\
<|im_start|>system
You are an ASL gloss translator. Given exactly one ASL gloss sentence (all uppercase),
output only its simple, present‐tense English translation—no extra commentary or new sentences.
<|im_end|>
"""

FEW_SHOT_CHAT = [
    (
        "<|im_start|>user\nGloss: FOOTBALL YOU LIKE\n<|im_end|>\n"
        "<|im_start|>assistant\nYou like football.\n<|im_end|>\n"
    ),
    (
        "<|im_start|>user\nGloss: ME NAME WHAT\n<|im_end|>\n"
        "<|im_start|>assistant\nWhat is my name?\n<|im_end|>\n"
    ),
    (
        "<|im_start|>user\nGloss: I HAVE DOG NAME R E X\n<|im_end|>\n"
        "<|im_start|>assistant\nI have a dog named Rex.\n<|im_end|>\n"
    ),
]

# ─── CONFIG: Pose Model / ST-GCN ─────────────────────────────────────────────────

VIDEO_PATH       = os.path.join("videos", "tongue1.mp4")
MODEL_PATH       = "pose_models/stgcn_asl_citizen.pth"
MOTION_THRESHOLD = 0.3
WINDOW_SIZE      = 32
STRIDE           = 8
N_FEATURES       = 256
N_CLASSES        = 2731  # Make sure this matches your trained model

# Use a single gloss list file
GLOSS_LIST_FILE = "vocab/gloss_list.txt"

# ─── HELPER: Extract & normalize landmarks ──────────────────────────────────────

def extract_27_keypoints_normalized(results):
    """
    Given MediaPipe holistic results, pick 27 keypoints (pose, hands, face),
    center/scale by shoulders, and return a (27, 2) numpy array.
    """
    full_vec = np.zeros((543, 2))

    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark[:33]):
            full_vec[i] = [lm.x, lm.y]

    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark[:21]):
            full_vec[33 + i] = [lm.x, lm.y]

    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark[:21]):
            full_vec[54 + i] = [lm.x, lm.y]

    if results.face_landmarks:
        for i, lm in enumerate(results.face_landmarks.landmark[:468]):
            full_vec[75 + i] = [lm.x, lm.y]

    keypoints = [
        0, 2, 5, 11, 12, 13, 14,
        33, 37, 38, 41, 42, 45, 46,
        49, 50, 53, 54, 58, 59,
        62, 63, 66, 67, 70, 71, 74
    ]
    selected = full_vec[keypoints]

    shoulder_l = full_vec[11]
    shoulder_r = full_vec[12]
    center     = (shoulder_l + shoulder_r) / 2
    shoulder_dist = np.linalg.norm(shoulder_l - shoulder_r)
    scale = 1.0 / shoulder_dist if shoulder_dist > 0 else 1.0

    normalized = (selected - center) * scale
    return normalized  # shape (27, 2)

# ─── HELPER: Translate one gloss via Zephyr API ─────────────────────────────────

def translate_gloss(gloss: str, hf_token: str) -> (str, float):
    """
    Given a single ASL gloss (uppercase), call the Zephyr Hugging Face Inference API
    to translate it into simple, present‐tense English. Returns (translation, elapsed_ms).
    """
    chatml = SYSTEM_PROMPT
    for pair in FEW_SHOT_CHAT:
        chatml += pair
    chatml += (
        "<|im_start|>user\n"
        f"Gloss: {gloss}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": chatml,
        "parameters": {
            "max_new_tokens": 64,
            "temperature": 0.1,         # Zephyr requires > 0
            "stop": ["<|im_end|>"]
        }
    }

    t0 = time.perf_counter()
    resp = requests.post(API_URL, headers=headers, json=payload)
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000

    if resp.status_code != 200:
        raise RuntimeError(f"Zephyr API error {resp.status_code}: {resp.text}")

    data = resp.json()
    if not isinstance(data, list) or "generated_text" not in data[0]:
        raise RuntimeError(f"Unexpected Zephyr response format: {data}")

    full_output = data[0]["generated_text"]

    if "<|im_start|>assistant" in full_output:
        answer_block = full_output.split("<|im_start|>assistant")[-1]
    else:
        answer_block = full_output
    answer = answer_block.split("<|im_end|>")[0].strip()
    answer = answer.split("\n")[0].strip()

    return answer, round(elapsed_ms, 2)

# ─── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    hf_token = os.getenv("HF_API_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_API_TOKEN is not set.\n"
            "Run:\n\n"
            "  export HF_API_TOKEN=\"hf_your_token_here\"\n\n"
            "and then rerun this script."
        )

    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video file not found: {VIDEO_PATH}")

    if not os.path.exists(GLOSS_LIST_FILE):
        raise FileNotFoundError(f"Cannot find gloss list file: {GLOSS_LIST_FILE}")

    # Load single gloss list
    with open(GLOSS_LIST_FILE, "r") as f:
        gloss_list = [line.strip() for line in f]

    # Load the ST-GCN pose model
    graph_args = {
        'num_nodes': 27,
        'center': 0,
        'inward_edges': [
            [2, 0], [1, 0], [0, 3], [0, 4], [3, 5],
            [4, 6], [5, 7], [6, 17], [7, 8], [7, 9],
            [9, 10], [7, 11], [11, 12], [7, 13], [13, 14],
            [7, 15], [15, 16], [17, 18], [17, 19], [19, 20],
            [17, 21], [21, 22], [17, 23], [23, 24], [17, 25], [25, 26]
        ]
    }
    encoder = STGCN(
        in_channels=2,
        graph_args=graph_args,
        edge_importance_weighting=True,
        n_out_features=N_FEATURES
    )
    decoder = FC(
        n_features=N_FEATURES,
        num_class=N_CLASSES,
        dropout_ratio=0.05
    )
    model = Network(encoder=encoder, decoder=decoder)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()

    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Extract landmarks from the video
    cap = cv2.VideoCapture(VIDEO_PATH)
    pose_sequence = []

    print("Extracting landmarks from video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        vec = extract_27_keypoints_normalized(results)
        pose_sequence.append(vec)

    cap.release()
    holistic.close()
    num_frames = len(pose_sequence)
    print(f"Extracted keypoints for {num_frames} frames.")

    # Sliding window inference + translation
    print("\nRunning ST-GCN on sliding windows and translating detected glosses:\n")

    seen_glosses = set()

    for start in range(0, num_frames - WINDOW_SIZE + 1, STRIDE):
        window = pose_sequence[start : start + WINDOW_SIZE]
        window_np = np.array(window)

        diffs = np.linalg.norm(np.diff(window_np, axis=0), axis=(1, 2))
        motion_score = np.sum(diffs)
        if motion_score < MOTION_THRESHOLD:
            continue

        input_tensor = torch.tensor(window_np).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_tensor.float())
            probs  = torch.softmax(logits, dim=1).squeeze()
            pred_idx = torch.argmax(probs).item()

        if pred_idx < 0 or pred_idx >= len(gloss_list):
            detected_gloss = "<out_of_bounds>"
        else:
            detected_gloss = gloss_list[pred_idx]

        if detected_gloss not in seen_glosses:
            seen_glosses.add(detected_gloss)

            print(f"Window {start}-{start+WINDOW_SIZE} | Motion: {motion_score:.3f}")
            print(f"  Detected gloss: {detected_gloss}")

            try:
                translation, elapsed_ms = translate_gloss(detected_gloss, hf_token)
                print(f"  → Translation: \"{translation}\"  [{elapsed_ms:.1f} ms]\n")
            except Exception as e:
                print(f"  [Error translating gloss \"{detected_gloss}\"]: {e}\n")

    print("Done processing video.")

if __name__ == "__main__":
    main()
