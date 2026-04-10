"""
main_ml.py
──────────
NatyaNetra — Mudra Challenge  (ML version)

Identical to the original main.py except the rule-based classifier is replaced
by the trained MLP.  Every other behaviour (serial comms, display, scoring) is
unchanged.

Usage:
    python main_ml.py --model mudra_model.pt
"""

import argparse
import cv2
import time
from collections import deque
import serial
from gesture_classifier_ml import classify_mudra, classify_mudra_with_probs, load_model
from gesture_classifier_ml import classify_mudra, load_model

# ── Configuration ─────────────────────────────────────────────────────────────

SERIAL_PORT = "COM4"
BAUD_RATE   = 9600
CAMERA_INDEX = 0

MUDRAS = ["Pathaka(1)", "Tripathaka(1)", "Mushti(1)", "Ardhachandran(1)", "Hamsasyam(1)", "Alapadmam(1)"]

MUDRA_ID = {
    "Unknown": 0, "Pathaka(1)": 1, "Tripathaka(1)": 2, "Mushti(1)": 3,
    "Ardhachandran(1)": 4, "Hamsasyam(1)": 5, "Alapadmam(1)": 6,
}

# ── Serial ────────────────────────────────────────────────────────────────────

ser = None

def init_serial():
    global ser
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
        time.sleep(2)
        print(f"Connected to MSPM0 on {SERIAL_PORT}")
        return True
    except Exception:
        print("Serial unavailable — running in visual-only mode")
        return False


def send_packet(target: str, detected: str):
    if detected == "Unknown" or not ser:
        return
    packet = bytes([MUDRA_ID[target], MUDRA_ID.get(detected, 0)])
    try:
        ser.write(packet)
        ser.flush()
        print(f"Sent: {target}({MUDRA_ID[target]}) vs {detected}({MUDRA_ID.get(detected, 0)})")
    except Exception:
        pass


def read_score() -> str:
    if not ser or ser.in_waiting == 0:
        return "0/0"
    try:
        data = ser.read(ser.in_waiting).decode("utf-8", errors="ignore").strip()
        if "SCORE:" in data:
            return data.replace("SCORE:", "").strip()
    except Exception:
        pass
    return "0/0"


# ── Main ──────────────────────────────────────────────────────────────────────

def main(model_path: str):
    # Load the trained model (once, before the video loop)
    load_model(model_path, device="cpu")

    init_serial()

    print(f"\nNatyaNetra — Mudra Challenge  [ML mode]")
    print(f"Available mudras: {', '.join(MUDRAS)}\n")

    target = ""
    while target not in MUDRAS:
        target = input("Choose mudra: ").strip().capitalize()

    print(f"\nTarget: {target}")
    print("Press Q or ESC to quit\n")

    import mediapipe as mp
    mp_hands        = mp.solutions.hands
    mp_drawing      = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    cap             = cv2.VideoCapture(CAMERA_INDEX)
    gesture_history = deque(maxlen=15)
    stable_gesture  = "Unknown"
    last_state      = None
    correct_count = 0
    total_attempts = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image     = cv2.flip(frame, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results   = hands.process(rgb_image)

        detected = "Unknown"
        if results.multi_hand_landmarks:
            # ML classification — one call replaces all the rule logic
            detected, probs = classify_mudra_with_probs(results.multi_hand_landmarks[0].landmark)
            print(probs)

            mp_drawing.draw_landmarks(
                image,
                results.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

        # Stabilise gesture over recent frames
        gesture_history.append(detected)
        if gesture_history.count(detected) >= 8:
            stable_gesture = detected

        is_correct = (stable_gesture == target)

        if stable_gesture == "Unknown":
            status, color = "Show hand", (200, 200, 200)
        elif is_correct:
            status, color = "CORRECT!", (0, 255, 0)
        else:
            status, color = "Try Again", (0, 0, 255)

        # Send serial packet only on state transition
        if last_state != is_correct and stable_gesture != "Unknown":
            send_packet(target, stable_gesture)
            total_attempts += 1
            if is_correct:
                correct_count += 1
            last_state = is_correct

     

        # Overlay
        cv2.putText(image, f"Target:   {target}",         (20,  50), cv2.FONT_HERSHEY_SIMPLEX, 1,   (255, 255,   0), 2)
        cv2.putText(image, f"Detected: {stable_gesture}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,   (255, 255, 255), 2)
        cv2.putText(image, status,                         (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,           2)
        
        cv2.putText(image, "[ML]",                         (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
        cv2.putText(image, f"Score: {correct_count}/{total_attempts}", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
        cv2.imshow("Mudra Challenge", image)

        if cv2.waitKey(5) & 0xFF in [ord("q"), 27]:
            break

    if ser:
        ser.close()
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NatyaNetra Mudra Challenge — ML mode")
    parser.add_argument("--model", type=str, default="mudra_model.pt",
                        help="Path to trained .pt checkpoint")
    args = parser.parse_args()
    main(args.model)
