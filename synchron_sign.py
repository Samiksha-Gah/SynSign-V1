# ─── synchron_sign.py ────────────────────────────────────────────────────────────
import os

# ─── QUIET ALL Python warnings ───────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")   # drop every UserWarning, DeprecationWarning, etc.

# ─── SILENCE Mediapipe/TF log spam ────────────────────────────────────────────────
import logging
logging.getLogger("mediapipe").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ─── NOW import everything else ───────────────────────────────────────────────────
import time
import threading
import requests
import cv2
import numpy as np
import torch
import mediapipe as mp

import PySimpleGUI as sg
import pyvirtualcam

from pose_models.st_gcn import STGCN
from pose_models.fc import FC
from pose_models.network import Network

# ─── CONFIG: Pose Model / ST-GCN ──────────────────────────────────────────────────
VIDEO_PATH       = 0  # 0 = first webcam; or replace with "videos/tongue1.mp4"
MODEL_PATH       = "pose_models/stgcn_asl_citizen.pth"
MOTION_THRESHOLD = 0.5
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

shutting_down       = False


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


# ─── HELPER #2: Word-wrap & overlay text on a frame ─────────────────────────────
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


# ─── HELPER #3: Schedule & clear overlay text after 2.5 s ────────────────────────
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


# ─── MAIN APPLICATION ────────────────────────────────────────────────────────────
def main():
    global gloss_buffer, partial_timer, final_timer, shutting_down

    print("[INFO] Starting synchron_sign.py")

    # 1) Verify ST-GCN model & gloss list exist
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"ST-GCN model not found: {MODEL_PATH}")
    if not os.path.exists(GLOSS_LIST_FILE):
        raise FileNotFoundError(f"Gloss list not found: {GLOSS_LIST_FILE}")
    print(f"[INFO] Found ST-GCN model at '{MODEL_PATH}' and gloss list at '{GLOSS_LIST_FILE}'")

    # 2) Load gloss list
    with open(GLOSS_LIST_FILE, "r") as f:
        gloss_list_local = [line.strip() for line in f]
    print(f"[INFO] Loaded {len(gloss_list_local)} gloss tokens into memory")

    # 3) Load ST-GCN pose model
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

    # 4) Initialize MediaPipe Holistic
    print("[INFO] Initializing MediaPipe Holistic...")
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    print("[INFO] MediaPipe Holistic ready")

    # 5) Build PySimpleGUI window
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

    # 6) Open webcam
    print("[INFO] Attempting to open webcam...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        sg.popup_error("Error: Could not open webcam. Make sure you’ve granted camera permission.")
        print("[ERROR] Failed to open webcam. Exiting.")
        shutting_down = True
        window.close()
        return
    print("[INFO] Webcam opened successfully")

    # 7) Attempt to open virtual camera (warn if unavailable)
    virtual_cam = None
    try:
        print("[INFO] Attempting to open virtual camera (pyvirtualcam)...")
        virtual_cam = pyvirtualcam.Camera(width=640, height=480, fps=30)
        print("[INFO] Virtual camera opened")
    except Exception as e:
        print(f"[WARNING] Could not open virtual camera → continuing without it.\n  ({e})")
        virtual_cam = None

    # 8) Start the GUI event loop
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

        if event == "Done" or event in (" ", "Space:32"):
            print("[DEBUG] Done/Space pressed → (no translations in this “gloss‐only” build)")
            # If you only want to show glosses, skip any API calls here.

        partial_enabled = values["-PARTIAL-"]

        if not running:
            continue

        # 9) Capture one frame
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Webcam frame read failed; breaking loop")
            break
        frame = cv2.resize(frame, (640, 480))
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"[DEBUG] Captured {frame_count} frames so far")

        # 10) Run MediaPipe on this frame
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        vec = extract_27_keypoints_normalized(results)
        pose_sequence.append(vec)

        # 11) Once we have ≥ WINDOW_SIZE frames, compute motion & run ST-GCN
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

        # 12) Draw the most recent gloss as overlay (if any)
        with overlay_lock:
            if gloss_buffer:
                overlay_text_on_frame(frame, gloss_buffer[-1])  # show only latest gloss

        # 13) Update the GUI image element
        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["-IMAGE-"].update(data=imgbytes)

        # 14) Send to virtual cam if available
        if virtual_cam is not None:
            try:
                virtual_cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            except Exception as e:
                print(f"[WARNING] virtual_cam.send() failed: {e}")
        # sleep just a tiny bit to reduce CPU
        time.sleep(1 / 30)

    # ─── Cleanup on exit ───────────────────────────────────────────────────────────
    print("[INFO] Cleaning up camera and window")
    shutting_down = True

    cap.release()
    if virtual_cam is not None:
        virtual_cam.close()
    window.close()
    print("[INFO] synchron_sign.py exit complete")


if __name__ == "__main__":
    main()
