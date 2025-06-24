import os
import cv2
import numpy as np
import torch
from pose_models.st_gcn import STGCN
from pose_models.fc import FC
from pose_models.network import Network
import mediapipe as mp

# --- Config ---
VIDEO_PATH = os.path.join("videos", "tongue1.mp4")
MODEL_PATH = "pose_models/stgcn_asl_citizen.pth"
MOTION_THRESHOLD = 0.3
WINDOW_SIZE = 32
STRIDE = 8
N_FEATURES = 256
N_CLASSES = 2731  # Make sure this matches your model

# --- Check file exists ---
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Video file not found: {VIDEO_PATH}")

# --- Load gloss lists ---
with open("vocab/gloss_list_alphabetical.txt", "r") as f:
    alphabetical = [line.strip() for line in f]

with open("vocab/gloss_list_train_order.txt", "r") as f:
    train_order = [line.strip() for line in f]

# --- Extract & normalize exactly like training ---
def extract_27_keypoints_normalized(results):
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

    keypoints = [0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46, 49,
                 50, 53, 54, 58, 59, 62, 63, 66, 67, 70, 71, 74]
    selected = full_vec[keypoints]

    shoulder_l = full_vec[11]
    shoulder_r = full_vec[12]
    center = (shoulder_l + shoulder_r) / 2
    shoulder_dist = np.linalg.norm(shoulder_l - shoulder_r)
    scale = 1.0 / shoulder_dist if shoulder_dist > 0 else 1.0

    normalized = (selected - center) * scale
    return normalized  # shape (27, 2)

# --- Load model ---
graph_args = {
    'num_nodes': 27,
    'center': 0,
    'inward_edges': [[2, 0], [1, 0], [0, 3], [0, 4], [3, 5],
                     [4, 6], [5, 7], [6, 17], [7, 8], [7, 9],
                     [9, 10], [7, 11], [11, 12], [7, 13], [13, 14],
                     [7, 15], [15, 16], [17, 18], [17, 19], [19, 20],
                     [17, 21], [21, 22], [17, 23], [23, 24], [17, 25], [25, 26]]
}
encoder = STGCN(in_channels=2, graph_args=graph_args, edge_importance_weighting=True, n_out_features=N_FEATURES)
decoder = FC(n_features=N_FEATURES, num_class=N_CLASSES, dropout_ratio=0.05)
model = Network(encoder=encoder, decoder=decoder)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# --- Init MediaPipe ---
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# --- Extract landmarks from video ---
cap = cv2.VideoCapture(VIDEO_PATH)
pose_sequence = []

print("Extracting landmarks...")
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
print(f"Extracted {len(pose_sequence)} frames.")

# --- Sliding window inference ---
print("Running ST-GCN on sliding windows:")
for start in range(0, len(pose_sequence) - WINDOW_SIZE + 1, STRIDE):
    window = pose_sequence[start:start + WINDOW_SIZE]
    window = np.array(window)
    diffs = np.linalg.norm(np.diff(window, axis=0), axis=(1, 2))
    motion_score = np.sum(diffs)

    if motion_score < MOTION_THRESHOLD:
        continue

    input_tensor = torch.tensor(window).permute(2, 0, 1).unsqueeze(0)  # (1, 2, T, 27)
    with torch.no_grad():
        logits = model(input_tensor.float())
        probs = torch.softmax(logits, dim=1).squeeze()
        pred_idx = torch.argmax(probs).item()

        gloss_train = train_order[pred_idx] if pred_idx < len(train_order) else "<out of bounds>"
        gloss_alpha = alphabetical[pred_idx] if pred_idx < len(alphabetical) else "<out of bounds>"

        print(f"\nWindow {start}-{start+WINDOW_SIZE} | Motion: {motion_score:.3f}")
        print(f"  Predicted index: {pred_idx}")
        print(f"  gloss_list_train_order[{pred_idx}]: {gloss_train}")
        print(f"  gloss_list_alphabetical[{pred_idx}]: {gloss_alpha}")
