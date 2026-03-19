import cv2
import mediapipe as mp
import os
import csv
import math

# ========== USER INPUT ==========
INPUT_FOLDER = r"D:\mudra\Bharatanatyam-Mudra-Dataset-master\Bharatanatyam-Mudra-Dataset-master\Alapadmam(1)"
OUTPUT_FOLDER = r"D:\mudra\folder_csv"
CSV_NAME = "Alapadmam.csv"
MUDRA_LABEL = "Alapadmam"
# ================================

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

OUTPUT_CSV = os.path.join(OUTPUT_FOLDER, CSV_NAME)

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.3
)

def normalize_hand(landmarks):
    wrist = landmarks[0]
    norm = []

    scale = math.dist(
        [wrist.x, wrist.y],
        [landmarks[9].x, landmarks[9].y]
    )
    if scale == 0:
        scale = 1e-6

    for lm in landmarks:
        norm.extend([
            (lm.x - wrist.x) / scale,
            (lm.y - wrist.y) / scale,
            lm.z / scale
        ])
    return norm

# CSV header
header = []
for h in ["H1", "H2"]:
    for i in range(21):
        header += [f"{h}_x{i}", f"{h}_y{i}", f"{h}_z{i}"]
header.append("label")

total = 0
written = 0

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for img_name in os.listdir(INPUT_FOLDER):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        total += 1
        img_path = os.path.join(INPUT_FOLDER, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print("❌ Cannot read:", img_name)
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = mp_hands.process(img_rgb)

        if not result.multi_hand_landmarks:
            print("❌ No hand:", img_name)
            continue

        hand1 = [0.0] * 63
        hand2 = [0.0] * 63
        hands = result.multi_hand_landmarks

        if len(hands) >= 1:
            hand1 = normalize_hand(hands[0].landmark)
        if len(hands) == 2:
            hand2 = normalize_hand(hands[1].landmark)

        writer.writerow(hand1 + hand2 + [MUDRA_LABEL])
        written += 1

mp_hands.close()

print("\n✅ PROCESS FINISHED")
print("📸 Total images found :", total)
print("✍ Rows written       :", written)
print("📄 CSV saved at       :", OUTPUT_CSV)
