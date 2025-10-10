
# === Real-time YOLO + Eye Closure + Head Pose Alert ===
# This cell implements real-time detection using YOLO + MediaPipe as explained.
# Run this cell after installing the requirements.
# Requirements: ultralytics, mediapipe, opencv-python, simpleaudio

import time
import math
import sys
import threading
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

try:
    import simpleaudio as sa
    SIMPLEAUDIO_AVAILABLE = True
except Exception:
    SIMPLEAUDIO_AVAILABLE = False
    try:
        import winsound
        WINSOUND_AVAILABLE = True
    except Exception:
        WINSOUND_AVAILABLE = False

YOLO_MODEL_PATH = 'yolov8n.pt'
WEBCAM_INDEX = 0
EAR_THRESHOLD = 0.20
EAR_CONSEC_FRAMES = 15
HEAD_PITCH_THRESHOLD = 15.0
ALERT_COOLDOWN = 3.0

ALERT_WAVE = None
if SIMPLEAUDIO_AVAILABLE:
    fr = 44100
    t = np.linspace(0, 0.5, int(fr * 0.5), False)
    tone = np.sin(440 * 2 * np.pi * t) * 0.3
    audio = (tone * (2**15 - 1)).astype(np.int16)
    ALERT_WAVE = sa.AudioData(audio.tobytes(), sample_rate=fr, num_channels=1, sample_width=2)

last_alert_time = 0
def play_alert():
    global last_alert_time
    now = time.time()
    if now - last_alert_time < ALERT_COOLDOWN:
        return
    last_alert_time = now
    if SIMPLEAUDIO_AVAILABLE and ALERT_WAVE is not None:
        try:
            sa.play_buffer(ALERT_WAVE._data, 1, 2, ALERT_WAVE.sample_rate)
            return
        except Exception:
            pass
    if WINSOUND_AVAILABLE:
        try:
            winsound.Beep(1000, 500)
            return
        except Exception:
            pass
    print("ALERT: condition triggered")

def eye_aspect_ratio(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def estimate_head_pitch(landmarks, image_shape):
    h, w = image_shape[:2]
    try:
        nose = np.array([landmarks[1][0] * w, landmarks[1][1] * h])
        chin = np.array([landmarks[152][0] * w, landmarks[152][1] * h])
        v = chin - nose
        angle = math.degrees(math.atan2(v[1], v[0]))
        pitch = abs(90 - abs(angle))
        return pitch
    except Exception:
        return 0.0

if MP_AVAILABLE:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                        refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
else:
    face_mesh = None

yolo_model = None
if YOLO_AVAILABLE:
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
    except Exception:
        try:
            yolo_model = YOLO('yolov8n.pt')
        except Exception:
            yolo_model = None

def main():
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print('Cannot open webcam')
        return

    ear_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        if yolo_model is not None:
            try:
                results = yolo_model(frame, verbose=False)
                r = results[0]
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{conf:.2f}', (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            except Exception:
                pass

        if face_mesh is not None:
            try:
                mp_results = face_mesh.process(frame_rgb)
                if mp_results.multi_face_landmarks:
                    face_landmarks = mp_results.multi_face_landmarks[0]
                    lm = [(p.x, p.y, p.z) for p in face_landmarks.landmark]
                    left_eye_idx = [33, 160, 158, 133, 153, 144]
                    right_eye_idx = [362, 385, 387, 263, 373, 380]
                    left_eye = [(int(lm[i][0] * w), int(lm[i][1] * h)) for i in left_eye_idx]
                    right_eye = [(int(lm[i][0] * w), int(lm[i][1] * h)) for i in right_eye_idx]
                    left_ear = eye_aspect_ratio(left_eye)
                    right_ear = eye_aspect_ratio(right_eye)
                    ear = (left_ear + right_ear) / 2.0
                    for (x, y) in left_eye + right_eye:
                        cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
                    pitch = estimate_head_pitch(lm, frame.shape)
                    cv2.putText(frame, f'EAR: {ear:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                    cv2.putText(frame, f'Pitch: {pitch:.1f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                    if ear < EAR_THRESHOLD:
                        ear_counter += 1
                    else:
                        ear_counter = 0
                    if ear_counter >= EAR_CONSEC_FRAMES:
                        cv2.putText(frame, 'ALERT: Eyes closed!', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        threading.Thread(target=play_alert, daemon=True).start()
                    if pitch > HEAD_PITCH_THRESHOLD:
                        cv2.putText(frame, 'ALERT: Head down!', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        threading.Thread(target=play_alert, daemon=True).start()
                else:
                    ear_counter = 0
            except Exception as e:
                pass

        cv2.imshow('YOLO + Eye Alert', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
