# synchron_sign.py

import os
import time
import threading
import requests
import cv2
import numpy as np
import torch
import mediapipe as mp
import warnings

# ─── Suppress noisy Protobuf deprecation warnings ───────────────────────────────
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="SymbolDatabase.GetPrototype() is deprecated.*",
)

# ─── GUI + VirtualCam imports ───────────────────────────────────────────────────
import PySimpleGUI as sg
import pyvirtualcam

# ─── ST-GCN model imports ───────────────────────────────────────────────────────
from pose_models.st_gcn import STGCN
from pose_models.fc import FC
from pose_models.network import Network

# ─── CONFIG: Hugging Face Zephyr API ─────────────────────────────────────────────
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
API_URL   = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

SYSTEM_PROMPT = """\
<|im_start|>system
You are an ASL gloss translator. Given exactly one ASL gloss sentence (all uppercase),
output only its simple, present-tense English translation—no extra commentary or new sentences.
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

# ─── CONFIG: Pose Model / ST-GCN ──────────────────────────────────────────────────
VIDEO_PATH       = 0  # 0 = first webcam; or replace with "videos/tongue1.mp4"
MODEL_PATH       = "pose_models/stgcn_asl_citizen.pth"
MOTION_THRESHOLD = 0.7   # raised from 0.3 to reduce spurious motion
WINDOW_SIZE      = 32
STRIDE           = 8
N_FEATURES       = 256
N_CLASSES        = 2731  # must match your ST‐GCN vocab length

GLOSS_LIST_FILE = "vocab/gloss_list.txt"

# ─── GLOBAL STATE ────────────────────────────────────────────────────────────────
gloss_buffer        = []
last_flushed_index  = 0
first_gloss_time    = None
last_gloss_time     = None

partial_enabled     = True
current_overlay_text = ""
overlay_lock        = threading.Lock()

partial_timer       = None
final_timer         = None
clear_overlay_timer = None

shutting_down       = False  # Prevent spawning new timers during exit


# ─── HELPER #1: Extract & normalize 27 keypoints from MediaPipe results ─────────
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


# ─── HELPER #2: Call Zephyr API to translate a single gloss ─────────────────────
def translate_gloss(gloss: str, hf_token: str) -> (str, float):
    """
    Given one ASL gloss (uppercase), call Hugging Face's Zephyr API to translate
    to simple present-tense English. Returns (translation, elapsed_ms).
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
            "temperature": 0.1,
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


# ─── HELPER #3: Word-wrap & overlay text on a frame ─────────────────────────────
def wrap_text(text: str, max_width: int, font_scale=0.7, thickness=2) -> list:
    """
    Break `text` into lines that fit within `max_width` pixels (Hershey Simplex).
    Returns list of lines.
    """
    words = text.split()
    lines = []
    current_line = ""
    for w in words:
        test_line = current_line + (" " if current_line else "") + w
        (w_width, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        if w_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = w
    if current_line:
        lines.append(current_line)
    return lines


def overlay_text_on_frame(frame: np.ndarray, text: str):
    """
    Draw a translucent black rectangle at the bottom of `frame`,
    then draw `text` in white, word-wrapped.
    """
    h, w, _ = frame.shape
    bar_height = 60

    # 1) Draw translucent black rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 2) Word-wrap & draw text
    lines = wrap_text(text, max_width=w - 20, font_scale=0.7, thickness=2)
    y0 = h - 10
    for line in reversed(lines):
        (line_width, line_height), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x = 10
        y = y0 - 5
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y0 = y - line_height - 2


# ─── HELPER #4: Schedule & clear overlay text after 2.5 s ────────────────────────
def schedule_clear_overlay(delay: float):
    """
    Cancel any existing clear timer and schedule a new one after `delay` seconds.
    """
    global clear_overlay_timer
    if clear_overlay_timer:
        clear_overlay_timer.cancel()
    clear_overlay_timer = threading.Timer(delay, clear_overlay)
    clear_overlay_timer.start()


def clear_overlay():
    """
    Clear the overlay text (called after 2.5 s).
    """
    global current_overlay_text
    with overlay_lock:
        current_overlay_text = ""


# ─── PARTIAL-FLUSH THREAD: every 1 s if ≥ 3 unflushed gloss tokens ───────────────
def partial_flush_worker():
    global partial_timer, last_flushed_index, current_overlay_text

    if shutting_down:
        return

    # Reschedule next partial in 1 s
    partial_timer = threading.Timer(1.0, partial_flush_worker)
    partial_timer.start()

    if not partial_enabled:
        return

    num_unflushed = len(gloss_buffer) - last_flushed_index
    if num_unflushed < 3:  # require at least 3 new gloss tokens
        return

    unflushed = gloss_buffer[last_flushed_index:]
    if len(unflushed) > 25:
        unflushed = unflushed[-25:]
    gloss_str = " ".join(unflushed)
    print(f"[DEBUG] Partial flush: sending gloss_seq='{gloss_str}' to translate_gloss()")

    try:
        translation, elapsed_ms = translate_gloss(gloss_str, hf_token)
        print(f"[DEBUG] Partial translation result: '{translation}' ({elapsed_ms:.1f} ms)")
    except Exception as e:
        print(f"[DEBUG] Partial translate_gloss failed: {e}")
        return

    with overlay_lock:
        current_overlay_text = translation
    schedule_clear_overlay(2.5)


# ─── FINAL-FLUSH THREAD: every 0.2 s if ≥ 3 s of no new gloss ────────────────────
def final_flush_worker():
    global final_timer

    if shutting_down:
        return

    final_timer = threading.Timer(0.2, final_flush_worker)
    final_timer.start()

    if not gloss_buffer or first_gloss_time is None:
        return

    if time.time() - first_gloss_time >= 3.0:
        print("[DEBUG] 3 seconds passed with no new gloss → final flush")
        do_final_flush()


def do_final_flush():
    """
    Immediately translate all unflushed glosses, overlay result, reset buffer,
    then schedule clearing after 2.5 s and restart partial thread if not shutting_down.
    """
    global last_flushed_index, first_gloss_time, last_gloss_time, current_overlay_text, partial_timer

    # Cancel pending partial + clear timers
    if partial_timer:
        partial_timer.cancel()
    if clear_overlay_timer:
        clear_overlay_timer.cancel()

    unflushed = gloss_buffer[last_flushed_index:]
    if len(unflushed) > 25:
        unflushed = unflushed[-25:]
    gloss_str = " ".join(unflushed)
    print(f"[DEBUG] Final flush: sending gloss_seq='{gloss_str}' to translate_gloss()")

    try:
        translation, elapsed_ms = translate_gloss(gloss_str, hf_token)
        print(f"[DEBUG] Final translation result: '{translation}' ({elapsed_ms:.1f} ms)")
    except Exception as e:
        print(f"[DEBUG] Final translate_gloss failed: {e}")
        return

    with overlay_lock:
        current_overlay_text = translation

    # Mark all glosses as flushed and clear buffer
    last_flushed_index = len(gloss_buffer)
    gloss_buffer.clear()
    first_gloss_time = None
    last_gloss_time = None

    schedule_clear_overlay(2.5)

    # Restart partial thread if not shutting_down
    if not shutting_down:
        partial_timer = threading.Timer(1.0, partial_flush_worker)
        partial_timer.start()


# ─── MAIN APPLICATION ────────────────────────────────────────────────────────────
def main():
    global hf_token, gloss_buffer, first_gloss_time, last_gloss_time, partial_enabled
    global partial_timer, final_timer, shutting_down

    print("[INFO] Starting synchron_sign.py")

    # 1) Check HF_API_TOKEN
    hf_token = os.getenv("HF_API_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_API_TOKEN is not set.\n"
            "Run:\n\n"
            "  export HF_API_TOKEN=\"hf_your_token_here\"\n\n"
            "and then rerun this script."
        )
    print("[INFO] HF_API_TOKEN found")

    # 2) Verify ST-GCN model & gloss list exist
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"ST-GCN model not found: {MODEL_PATH}")
    if not os.path.exists(GLOSS_LIST_FILE):
        raise FileNotFoundError(f"Gloss list not found: {GLOSS_LIST_FILE}")
    print(f"[INFO] Found ST-GCN model at '{MODEL_PATH}' and gloss list at '{GLOSS_LIST_FILE}'")

    # 3) Load gloss list
    with open(GLOSS_LIST_FILE, "r") as f:
        gloss_list_local = [line.strip() for line in f]
    print(f"[INFO] Loaded {len(gloss_list_local)} gloss tokens into memory")

    # 4) Load ST-GCN pose model
    print("[INFO] Loading ST-GCN pose model (this may take a moment)...")
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
    pose_model = Network(encoder=encoder, decoder=decoder)
    pose_model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    pose_model.eval()
    print("[INFO] ST-GCN model loaded and set to eval()")

    # 5) Initialize MediaPipe Holistic
    print("[INFO] Initializing MediaPipe Holistic...")
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    print("[INFO] MediaPipe Holistic ready")

    # 6) Build PySimpleGUI window
    sg.theme("DarkGrey3")
    layout = [
        [sg.Image(filename="", key="-IMAGE-")],
        [sg.Text("Status: Waiting to start…", key="-STATUS-")],
        [
            sg.Button("Start", size=(6, 1)),
            sg.Button("Done", size=(6, 1)),
            sg.Button("Stop", size=(6, 1)),
            sg.Checkbox("Partial Captions", default=True, key="-PARTIAL-")
        ],
        [sg.Text("Tip: Press Spacebar or Done to force a flush.")]
    ]
    print("[INFO] Creating PySimpleGUI window...")
    window = sg.Window(
        "SynSign – Real-time ASL→English",
        layout,
        finalize=True,
        return_keyboard_events=True
    )
    print("[INFO] Window created; waiting for user to click Start")

    # 7) Open webcam
    print("[INFO] Attempting to open webcam...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        sg.popup_error("Error: Could not open webcam. Make sure you’ve granted camera permission.")
        print("[ERROR] Failed to open webcam. Exiting.")
        shutting_down = True
        window.close()
        return
    print("[INFO] Webcam opened successfully")

    # 8) Attempt to open virtual camera (warn if unavailable)
    virtual_cam = None
    try:
        print("[INFO] Attempting to open virtual camera (pyvirtualcam)...")
        virtual_cam = pyvirtualcam.Camera(width=640, height=480, fps=30)
        print("[INFO] Virtual camera opened")
    except Exception as e:
        print(f"[WARNING] Could not open virtual camera → continuing without it.\n  ({e})")
        virtual_cam = None

    # 9) Start the GUI event loop
    running = False
    pose_sequence = []

    # Debounce state:
    last_seen_gloss = None
    stable_count = 0
    last_appended = None

    frame_count = 0

    while True:
        event, values = window.read(timeout=10)
        if event == sg.WIN_CLOSED or event == "Stop":
            print("[INFO] Stop clicked or window closed → shutting down")
            shutting_down = True
            break

        if event == "Start":
            running = True
            window["-STATUS-"].update("Running…")
            print("[INFO] Start clicked → entering main capture loop")

            # Start the partial & final worker threads
            partial_timer = threading.Timer(1.0, partial_flush_worker)
            partial_timer.start()
            final_timer = threading.Timer(0.2, final_flush_worker)
            final_timer.start()

        if event == "Done" or event in (" ", "Space:32"):
            print("[DEBUG] Done/Space pressed → calling do_final_flush()")
            do_final_flush()

        partial_enabled = values["-PARTIAL-"]

        if not running:
            continue

        # 10) Capture one frame
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Webcam frame read failed; breaking loop")
            break
        frame = cv2.resize(frame, (640, 480))
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"[DEBUG] Captured {frame_count} frames so far")

        # 11) Run MediaPipe on this frame
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        vec = extract_27_keypoints_normalized(results)
        pose_sequence.append(vec)

        # 12) Once we have ≥ WINDOW_SIZE frames, compute motion & run ST-GCN
        if len(pose_sequence) >= WINDOW_SIZE:
            window_np = np.array(pose_sequence[-WINDOW_SIZE:])
            diffs = np.linalg.norm(np.diff(window_np, axis=0), axis=(1, 2))
            motion_score = np.sum(diffs)
            # Print motion score occasionally
            if frame_count % 30 == 0:
                print(f"[DEBUG] Motion score at frame {frame_count}: {motion_score:.3f}")

            if motion_score >= MOTION_THRESHOLD:
                print(f"[DEBUG] motion_score {motion_score:.3f} ≥ MOTION_THRESHOLD → run ST-GCN")
                input_tensor = torch.tensor(window_np).permute(2, 0, 1).unsqueeze(0)
                with torch.no_grad():
                    logits = pose_model(input_tensor.float())
                    probs  = torch.softmax(logits, dim=1).squeeze()
                    pred_idx = torch.argmax(probs).item()

                if 0 <= pred_idx < len(gloss_list_local):
                    detected_gloss = gloss_list_local[pred_idx]
                else:
                    detected_gloss = "<out_of_bounds>"

                # Debounce: only append after seeing same gloss STABLE_THRESHOLD times
                if detected_gloss == last_seen_gloss:
                    stable_count += 1
                else:
                    stable_count = 1
                last_seen_gloss = detected_gloss

                print(f"[DEBUG] ST-GCN predicted index={pred_idx}, gloss='{detected_gloss}', stable_count={stable_count}")

                if stable_count >= 2:  # STABLE_THRESHOLD = 2
                    if detected_gloss != last_appended:
                        now = time.time()
                        gloss_buffer.append(detected_gloss)
                        last_appended = detected_gloss
                        if first_gloss_time is None:
                            first_gloss_time = now
                        last_gloss_time = now
                        print(f"[INFO] Appended new gloss → '{detected_gloss}' (buffer size now {len(gloss_buffer)})")
                    stable_count = 0

            # Keep the last (WINDOW_SIZE - STRIDE) frames to overlap
            overlap = WINDOW_SIZE - STRIDE
            pose_sequence = pose_sequence[-overlap:]

        # 13) If there’s overlay text, draw it
        with overlay_lock:
            text_to_show = current_overlay_text
        if text_to_show:
            overlay_text_on_frame(frame, text_to_show)

        # 14) Update the GUI image element
        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["-IMAGE-"].update(data=imgbytes)

        # 15) Send to virtual cam if available
        if virtual_cam is not None:
            try:
                virtual_cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            except Exception as e:
                print(f"[WARNING] virtual_cam.send() failed: {e}")
        # sleep just a tiny bit to reduce CPU
        time.sleep(1 / 30)

    # ─── Cleanup on exit ───────────────────────────────────────────────────────────
    print("[INFO] Cleaning up timers, camera, and window")
    shutting_down = True
    if partial_timer:
        partial_timer.cancel()
    if final_timer:
        final_timer.cancel()
    if clear_overlay_timer:
        clear_overlay_timer.cancel()

    cap.release()
    if virtual_cam is not None:
        virtual_cam.close()
    window.close()
    print("[INFO] synchron_sign.py exit complete")


if __name__ == "__main__":
    main()
