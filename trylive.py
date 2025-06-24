import os
import cv2
import numpy as np
import torch
from collections import deque
import mediapipe as mp

# --- IMPORT YOUR MODEL CLASSES (adjust paths if needed) ---
from pose_models.st_gcn import STGCN
from pose_models.fc import FC
from pose_models.network import Network

# --- CONFIGURATION FOR FULL DEBUGGING ---
MODEL_PATH        = "pose_models/stgcn_asl_citizen.pth"
MOTION_THRESHOLD  = 0.3       # small cheek‐twist motion for “APPLE” should cross this
WINDOW_SIZE       = 32        # length of sliding window
STRIDE            = 8         # run inference every 8 frames (~0.25 s)
N_FEATURES        = 256
N_CLASSES         = 2731      # must match your trained model’s output size
CONFIDENCE_CUTOFF = 0.6       # allow “APPLE” even at ~0.55–0.6 confidence
VOTES_REQUIRED    = 2         # need two consecutive windows of same gloss
COOLDOWN_FRAMES   = 16        # after commit, wait 16 frames (~0.5 s)

# --- Load gloss_list.txt (one gloss per line) ---
with open("vocab/gloss_list.txt", "r") as f:
    gloss_list = [line.strip() for line in f]

if len(gloss_list) != N_CLASSES:
    print(f"⚠️  WARNING: gloss_list.txt has {len(gloss_list)} lines but N_CLASSES = {N_CLASSES}.")
    print("   Any pred_idx ≥ len(gloss_list) will be treated as “<OOB>.”\n")

# Helper to get gloss name or "<OOB>"
def gloss_name(idx):
    if idx is None:
        return "None"
    if 0 <= idx < len(gloss_list):
        return gloss_list[idx]
    return "<OOB>"

# --- Function to extract & normalize 27 keypoints exactly as during training ---
def extract_27_keypoints_normalized(results):
    full_vec = np.zeros((543, 2), dtype=np.float32)

    # Pose landmarks (first 33)
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark[:33]):
            full_vec[i] = [lm.x, lm.y]
    # Left hand (21)
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark[:21]):
            full_vec[33 + i] = [lm.x, lm.y]
    # Right hand (21)
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark[:21]):
            full_vec[54 + i] = [lm.x, lm.y]
    # Face (468)
    if results.face_landmarks:
        for i, lm in enumerate(results.face_landmarks.landmark[:468]):
            full_vec[75 + i] = [lm.x, lm.y]

    keypoints = [
        0, 2, 5, 11, 12, 13, 14,
        33, 37, 38, 41, 42, 45, 46,
        49, 50, 53, 54, 58, 59,
        62, 63, 66, 67, 70, 71, 74
    ]
    selected = full_vec[keypoints]  # shape = (27, 2)

    # Center on midpoint of shoulders
    shoulder_l = full_vec[11]
    shoulder_r = full_vec[12]
    center = (shoulder_l + shoulder_r) / 2.0

    # Scale by inverse shoulder distance
    shoulder_dist = np.linalg.norm(shoulder_l - shoulder_r)
    scale = 1.0 / shoulder_dist if (shoulder_dist > 1e-6) else 1.0

    normalized = (selected - center) * scale
    return normalized.astype(np.float32)  # (27, 2)

# --- Build & load the ST-GCN model ---
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
print("Model loaded successfully. Ready to run inference.\n")

# --- Initialize MediaPipe Holistic ---
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Initialize Webcam (macOS: no CAP_DSHOW) ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Check camera permissions or try a different index.")

# (Optional) fix resolution for performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- Buffers & State Variables ---
pose_buffer = deque(maxlen=WINDOW_SIZE)
frame_count = 0

# Last “committed” gloss index & confidence
last_displayed_idx        = None
last_displayed_confidence = 0.0

# Temporary candidate & its vote count
temp_candidate_idx        = None
temp_candidate_confidence = 0.0
consecutive_votes         = 0

# Frame index of last commit (for cooldown)
last_commit_frame = -COOLDOWN_FRAMES

print("Webcam started. Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab a frame. Exiting.")
        break

    frame_count += 1
    # Mirror for natural feel
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    # Extract normalized (27×2) vector
    vec = extract_27_keypoints_normalized(results)
    pose_buffer.append(vec)

    motion_score = 0.0
    current_idx = None
    current_confidence = 0.0

    # Only run inference every STRIDE frames, once buffer has WINDOW_SIZE frames
    if len(pose_buffer) == WINDOW_SIZE and (frame_count % STRIDE) == 0:
        window_np = np.stack(pose_buffer, axis=0)          # shape = (32, 27, 2)
        diffs = np.linalg.norm(np.diff(window_np, axis=0), axis=(1, 2))
        motion_score = float(np.sum(diffs))

        if motion_score >= MOTION_THRESHOLD:
            print(f"[Frame {frame_count:4d}] ▶ Inference triggered (motion_score = {motion_score:.3f} ≥ {MOTION_THRESHOLD})")

            # Build input tensor (1, 2, 32, 27)
            input_tensor = (
                torch.tensor(window_np)
                .permute(2, 0, 1)    # (coords=2, time=32, nodes=27)
                .unsqueeze(0)        # (batch=1, 2, 32, 27)
                .float()
            )
            with torch.no_grad():
                logits = model(input_tensor)            # (1, N_CLASSES)
                probs = torch.softmax(logits, dim=1)[0]  # (N_CLASSES,)
                top_confidence, top_idx = torch.max(probs, dim=0)
                top_confidence = float(top_confidence.item())
                top_idx = int(top_idx.item())

                gloss = gloss_name(top_idx)
                print(f"    ‣ Raw model output: pred_idx = {top_idx} → “{gloss}”   confidence = {top_confidence:.3f}")

                # Accept as candidate if ≥ CONFIDENCE_CUTOFF and in range
                if top_confidence >= CONFIDENCE_CUTOFF and top_idx < len(gloss_list):
                    current_idx = top_idx
                    current_confidence = top_confidence
                    print(f"    ‣ Candidate accepted (≥ {CONFIDENCE_CUTOFF}): “{gloss}”  (votes = 1)")
                else:
                    # Below cutoff or OOB → treat as no-change
                    current_idx = last_displayed_idx
                    current_confidence = last_displayed_confidence
                    reason = "confidence < cutoff" if top_confidence < CONFIDENCE_CUTOFF else "idx out of range"
                    print(f"    ‣ Candidate rejected ({reason}); reusing last_displayed_idx = {gloss_name(last_displayed_idx)}")
        else:
            print(f"[Frame {frame_count:4d}] ✱ Motion too low ({motion_score:.3f} < {MOTION_THRESHOLD}), skipping inference")

        # Voting logic (if current_idx is not None)
        if current_idx is not None:
            if current_idx == temp_candidate_idx:
                consecutive_votes += 1
                print(f"    ‣ Temp candidate “{gloss_name(current_idx)}” now has {consecutive_votes} consecutive vote(s)")
            else:
                temp_candidate_idx = current_idx
                temp_candidate_confidence = current_confidence
                consecutive_votes = 1
                print(f"    ‣ New temp candidate set: idx = {current_idx} (“{gloss_name(current_idx)}”), conf = {current_confidence:.3f} (votes = 1)")

            # If enough votes, attempt commit (subject to cooldown)
            if consecutive_votes >= VOTES_REQUIRED:
                frames_since = frame_count - last_commit_frame
                if frames_since >= COOLDOWN_FRAMES:
                    if temp_candidate_idx != last_displayed_idx:
                        last_displayed_idx = temp_candidate_idx
                        last_displayed_confidence = temp_candidate_confidence
                        last_commit_frame = frame_count
                        print(f"    ✅ Committed new gloss: idx = {last_displayed_idx} → “{gloss_name(last_displayed_idx)}”, conf = {last_displayed_confidence:.3f}, at frame {frame_count}")
                    else:
                        print(f"    ⚠️  Temp candidate matches already displayed (idx = {temp_candidate_idx} → “{gloss_name(temp_candidate_idx)}”); no commit needed")
                    # Reset voting after commit (or redundant)
                    consecutive_votes = 0
                    temp_candidate_idx = None
                    temp_candidate_confidence = 0.0
                else:
                    left = COOLDOWN_FRAMES - frames_since
                    print(f"    ⏱ In cooldown for {left} more frames; delaying commit of idx = {temp_candidate_idx} (“{gloss_name(temp_candidate_idx)}”)")
        else:
            print(f"    ‣ No valid candidate this round (current_idx is None)")

    # Decide what to display on screen
    if last_displayed_idx is not None and 0 <= last_displayed_idx < len(gloss_list):
        display_text = gloss_list[last_displayed_idx]
    else:
        display_text = "---"

    # Draw translucent box behind text for readability
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (350, 90), (0, 0, 0), thickness=-1)
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Show motion_score and committed gloss on screen
    cv2.putText(
        frame,
        f"Motion: {motion_score:.3f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Predicted: {display_text}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0) if display_text != "---" else (0, 0, 255),
        2
    )

    cv2.imshow("Live ST-GCN Prediction (DEBUG)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nUser pressed 'q'. Exiting.")
        break

# Cleanup
cap.release()
holistic.close()
cv2.destroyAllWindows()
