"""
Main runner for AI Object & Scene Narrator
- Webcam capture
- YOLOv8 detection (ultralytics)
- Tesseract OCR
- pyttsx3 TTS
- On-demand and auto modes
"""

import time
import cv2
from ultralytics import YOLO
from .config import (
    MODEL_NAME, CONF_THRESHOLD, NARRATION_COOLDOWN, MAX_OBJECTS_TO_SAY,
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT
)
from .utils import ocr_read, speak, make_scene_sentence, init_tts


def main():
    print("Loading YOLO model (this may download weights if missing)...")
    model = YOLO(MODEL_NAME)
    init_tts()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Failed to open camera. Exiting.")
        return

    last_narration = 0
    mode_auto = False

    print("Controls: SPACE - narrate once | a - toggle auto-mode | q - quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed, skipping...")
            time.sleep(0.1)
            continue

        # YOLO inference (smaller frame for speed)
        small = cv2.resize(frame, (640, 360))
        results = model(small, conf=CONF_THRESHOLD)[0]

        detected = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            score = float(box.conf[0])
            if score < CONF_THRESHOLD:
                continue
            name = model.names[cls_id]
            detected.append(name)

            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            h_small, w_small = small.shape[:2]
            h, w = frame.shape[:2]
            sx, sy = w / w_small, h / h_small
            x1, x2 = int(x1 * sx), int(x2 * sx)
            y1, y2 = int(y1 * sy), int(y2 * sy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {score:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # OCR
        try:
            ocr_text = ocr_read(small)
        except Exception:
            ocr_text = ""

        cv2.putText(frame, f"Mode: {'AUTO' if mode_auto else 'ON-DEMAND'}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow('Narrator Preview', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            mode_auto = not mode_auto
            print('Auto-mode:', mode_auto)
        elif key == 32:  # SPACE
            msg = make_scene_sentence(detected, ocr_text, max_objects=MAX_OBJECTS_TO_SAY)
            speak(msg)

        if mode_auto:
            now = time.time()
            if now - last_narration >= NARRATION_COOLDOWN:
                msg = make_scene_sentence(detected, ocr_text, max_objects=MAX_OBJECTS_TO_SAY)
                speak(msg)
                last_narration = now

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
