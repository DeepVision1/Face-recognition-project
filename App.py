# ========================================
# DEEPVISION CAR APP - COMPLETE INTEGRATION
# Face + Voice + Drowsiness Detection System
# ========================================
import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from numpy.linalg import norm
import sqlite3
import pickle
import time
import glob
import os
import math
import threading

import librosa
import tensorflow as tf
import sounddevice as sd

from ultralytics import YOLO
import mediapipe as mp
import winsound

# ========================================
# CONFIGURATION
# ========================================
FACE_THRESHOLD = 0.5
FACE_COOLDOWN = 30
VOICE_THRESHOLD = 0.7
VOICE_DURATION = 3
VOICE_SR = 16000
EAR_THRESHOLD = 0.20
EAR_CONSEC_FRAMES = 15
HEAD_PITCH_THRESHOLD = 15.0
ALERT_COOLDOWN = 3.0

# States
STATE_FACE_DETECTION = 0
STATE_FACE_RECOGNIZED = 1
STATE_VOICE_VERIFICATION = 2
STATE_AUTHENTICATED = 3
STATE_MONITORING = 4


# ========================================
# SPLASH SCREEN
# ========================================
def update_splash(window, message):
    splash = 255 * np.ones((400, 700, 3), dtype=np.uint8)
    cv2.putText(splash, message, (50, 200), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imshow(window, splash)
    cv2.waitKey(1)


# ========================================
# DATABASE SETUP
# ========================================
conn = sqlite3.connect("Database/faces.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    Face_embedding BLOB,
    Voice_embedding BLOB
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS recognition_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    similarity REAL,
    verification_type TEXT,
    FOREIGN KEY(person_id) REFERENCES faces(id)
)
""")
conn.commit()


# ========================================
# LOAD MODELS
# ========================================
window_name = "Loading..."
update_splash(window_name, "Loading Face Models...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval()

update_splash(window_name, "Loading Voice Model...")

siamese_model = tf.keras.models.load_model("Models/Voice_verification_model2.h5", compile=False)
embedding_model = siamese_model.layers[3]
print("✅ Voice model loaded successfully")

update_splash(window_name, "Loading Drowsiness Models...")

yolo_model = YOLO("Models/yolov8n.pt")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ========================================
# AUDIO ALERT
# ========================================
last_alert_time = 0

def play_alert():
    global last_alert_time
    now = time.time()
    if now - last_alert_time < ALERT_COOLDOWN:
        return
    last_alert_time = now
    winsound.Beep(1000, 500)
    print("🚨 DROWSINESS ALERT!")


# ========================================
# VOICE FUNCTIONS
# ========================================
def record_voice(duration=VOICE_DURATION, sr=VOICE_SR):
    print(f"🎤 Recording for {duration} seconds...")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    return recording.flatten()


def Voice_embedding(model, y, sr=VOICE_SR, target_sec=VOICE_DURATION, n_mfcc=40):
    target_len = sr * target_sec
    if len(y) > target_len:
        start = (len(y) - target_len) // 2
        y = y[start:start+target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = np.mean(mfcc.T, axis=0)
    mfcc = np.expand_dims(mfcc, -1)
    mfcc = np.expand_dims(mfcc, 0).astype(np.float32)
    
    emb = model.predict(mfcc, verbose=0)
    return tf.math.l2_normalize(emb, axis=1)


# ========================================
# FACE FUNCTIONS
# ========================================
def Face_embedding(face_tensor):
    if face_tensor.ndim == 3:
        face_tensor = face_tensor.unsqueeze(0)
    return facenet(face_tensor).detach().numpy()[0]


def Face_cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)

def Voice_cosine_similarity(a, b):
    return -cosine_loss(a, b).numpy()


# ========================================
# DROWSINESS FUNCTIONS
# ========================================
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
    except:
        return 0.0


# ========================================
# LOAD REFERENCE FACE DATA
# ========================================
update_splash(window_name, "Loading Reference Faces...")

folder_path = "persons"
ref_images = glob.glob(os.path.join(folder_path, "*.jpg")) + \
            glob.glob(os.path.join(folder_path, "*.png"))
ref_display_images = []

for img_path in ref_images:
    name = os.path.splitext(os.path.basename(img_path))[0]
    cursor.execute("SELECT id FROM faces WHERE name=?", (name,))
    if cursor.fetchone() is None:
        img = Image.open(img_path).convert("RGB")
        ref_face = mtcnn(img)
        
        if ref_face is not None:
            embedding = Face_embedding(ref_face)
            cursor.execute(
                "INSERT INTO faces (name, Face_embedding) VALUES (?, ?)",
                (name, pickle.dumps(embedding))
            )
            conn.commit()
            print(f"✅ Added face for: {name}")
        else:
            print(f"⚠️ No face detected in {img_path}")
    
    disp_img = cv2.imread(img_path)
    if disp_img is not None:
        disp_img = cv2.resize(disp_img, (258, 396))
        ref_display_images.append(disp_img)


# ========================================
# LOAD AND EMBED VOICE DATA
# ========================================
update_splash(window_name, "Loading Voice Samples...")


voices_base_path = "voices"

if os.path.exists(voices_base_path):
    voice_folders = [f for f in os.listdir(voices_base_path) 
                    if os.path.isdir(os.path.join(voices_base_path, f))]
    
    print(f"\n📁 Found {len(voice_folders)} voice folders")
    
    for person_folder in voice_folders:
        person_name = person_folder
        folder_path = os.path.join(voices_base_path, person_folder)
        
        cursor.execute("SELECT id, Voice_embedding FROM faces WHERE name=?", (person_name,))
        result = cursor.fetchone()
        
        if result is None:
            print(f"⚠️ Person '{person_name}' not found in database. Add face first!")
            continue
        
        person_id, existing_voice = result
        
        if existing_voice is not None:
            print(f"✅ Voice already exists for: {person_name}")
            continue
        
        audio_files = glob.glob(os.path.join(folder_path, "*.wav"))
        
        if not audio_files:
            print(f"⚠️ No .wav files found for: {person_name}")
            continue
        
        print(f"🎤 Processing voice for: {person_name}")
        print(f"   Found {len(audio_files)} audio files")
        
        voice_embeddings = []
        
        for audio_file in audio_files:
            try:
                audio_data, sr = librosa.load(audio_file, sr=VOICE_SR)
                emb = Voice_embedding(embedding_model, audio_data, sr=sr)
                voice_embeddings.append(emb.numpy())
            except Exception as e:
                print(f"Error processing {os.path.basename(audio_file)}: {e}")
                continue
        
        if voice_embeddings:
            avg_embedding = np.mean(voice_embeddings, axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
            
            cursor.execute("UPDATE faces SET Voice_embedding=? WHERE id=?",(pickle.dumps(avg_embedding), person_id))
            conn.commit()
            
            print(f"   ✅ Saved voice embedding (averaged {len(voice_embeddings)} samples)")
        else:
            print(f"   ❌ No valid voice embeddings for: {person_name}")
else:
    print(f"⚠️ Voices folder not found: {voices_base_path}")



# ========================================
# LOAD DATABASE REFERENCES
# ========================================
cursor.execute("SELECT id, name, Face_embedding, Voice_embedding FROM faces")
rows = cursor.fetchall()

ref_ids = []
ref_names = []
ref_face_embeddings = []
ref_voice_embeddings = []

for r in rows:
    ref_ids.append(r[0])
    ref_names.append(r[1])
    ref_face_embeddings.append(pickle.loads(r[2]))
    
    if r[3] is not None:
        ref_voice_embeddings.append(pickle.loads(r[3]))
    else:
        ref_voice_embeddings.append(None)

print(f"\n📋 Final database status - {len(ref_names)} persons:")
for i, name in enumerate(ref_names):
    face_status = "✅"
    voice_status = "✅" if ref_voice_embeddings[i] is not None else "❌"
    print(f"   {i+1}. {name} - Face: {face_status} | Voice: {voice_status}")

# Validate database
if len(ref_face_embeddings) == 0:
    print("\n❌ ERROR: No faces found in database!")
    print("   Please add face images to the 'persons' folder first.")
    conn.close()
    cv2.destroyAllWindows()
    exit(1)


# ========================================
# UI SETUP
# ========================================
update_splash(window_name, "Starting Camera...")

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread("DeepVision_Background.jpg")
cv2.destroyWindow(window_name)


# ========================================
# STATE MACHINE
# ========================================
current_state = STATE_FACE_DETECTION
recognized_person_id = None
recognized_person_name = None
recognized_person_idx = None
voice_verification_attempted = False

frame_count = 0
skip_frames = 5
scale = 0.5
boxes, probs, faces = None, None, None
last_seen = {}

ear_counter = 0
drowsy_alert_active = False

print("\n" + "="*60)
print("🚀 DEEPVISION SYSTEM STARTED")
print("="*60)
print("Controls:")
print("  V - Verify voice (when face recognized)")
print("  R - Reset to face detection")
print("  Q - Quit")
print("="*60 + "\n")

# ========================================
# MAIN LOOP
# ========================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read from camera")
        break
    
    frame_count += 1
    
    # ========================================
    # STATE: FACE DETECTION
    # ========================================
    if current_state == STATE_FACE_DETECTION:
        if frame_count % skip_frames == 0:
            small_rgb = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                None, fx=scale, fy=scale
            )
            boxes, probs = mtcnn.detect(small_rgb)
            faces = mtcnn(small_rgb)
            if boxes is not None:
                boxes /= scale
        
        if boxes is not None and faces is not None and len(ref_face_embeddings) > 0:
            for box, prob, face_tensor in zip(boxes, probs, faces):
                if face_tensor is not None:
                    try:
                        emb = Face_embedding(face_tensor)
                        
                        sims = [Face_cosine_similarity(ref_emb, emb) 
                                for ref_emb in ref_face_embeddings]
                        
                        if len(sims) == 0:
                            continue
                        
                        best_idx = int(np.argmax(sims))
                        sim = sims[best_idx]
                        
                        x1, y1, x2, y2 = [int(v) for v in box]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        label = f"{ref_names[best_idx]}: {sim:.2f}"
                        color = (0, 255, 0) if sim > FACE_THRESHOLD else (0, 0, 255)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        if sim > FACE_THRESHOLD:
                            recognized_person_id = ref_ids[best_idx]
                            recognized_person_name = ref_names[best_idx]
                            recognized_person_idx = best_idx
                            
                            if best_idx < len(ref_display_images):
                                imgBackground[304:304 + 396, 713:713 + 258] = \
                                    ref_display_images[best_idx]
                            
                            now = time.time()
                            if recognized_person_id not in last_seen or (now - last_seen[recognized_person_id]) > FACE_COOLDOWN:
                                cursor.execute("""INSERT INTO recognition_log (person_id, similarity, verification_type) VALUES (?, ?, ?)""",
                                (recognized_person_id, float(sim), 'face'))
                                conn.commit()
                                last_seen[recognized_person_id] = now
                            
                            current_state = STATE_FACE_RECOGNIZED
                            voice_verification_attempted = False
                            print(f"\n✅ Face Recognized: {recognized_person_name}")
                            
                            if ref_voice_embeddings[best_idx] is None:
                                print(f"⚠️ No voice data for {recognized_person_name}")
                                print("   Skipping to monitoring...")
                                current_state = STATE_AUTHENTICATED
                            else:
                                print("🎤 Press 'V' for voice verification")
                    
                    except Exception as e:
                        print(f"⚠️ Error processing face: {e}")
                        continue
        
        elif len(ref_face_embeddings) == 0:
            cv2.putText(frame, "No reference faces in database!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(frame, "STATE: Face Detection", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    # ========================================
    # STATE: FACE RECOGNIZED
    # ========================================
    elif current_state == STATE_FACE_RECOGNIZED:
        cv2.putText(frame, f"Face: {recognized_person_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'V' to verify voice", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # ========================================
    # STATE: VOICE VERIFICATION
    # ========================================
    elif current_state == STATE_VOICE_VERIFICATION:
        if not voice_verification_attempted:
            cv2.putText(frame, "Recording Voice...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(frame, "Please speak now!", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            resized_frame = cv2.resize(frame, (529, 397))
            imgBackground[303:700, 93:622] = resized_frame
            cv2.imshow("DeepVision System", imgBackground)
            cv2.waitKey(100)
            


            audio = record_voice()
            test_voice_emb = Voice_embedding(embedding_model, audio)
            
            ref_voice_emb = ref_voice_embeddings[recognized_person_idx]
            
            if ref_voice_emb is None:
                print(f"⚠️ No voice reference for {recognized_person_name}")
                current_state = STATE_AUTHENTICATED
            else:
                voice_sim = Voice_cosine_similarity(
                    test_voice_emb, 
                    ref_voice_emb.reshape(1, -1)
                )
                
                print(f"Voice Similarity: {voice_sim:.4f} (threshold: {VOICE_THRESHOLD})")
                
                if voice_sim > VOICE_THRESHOLD:
                    print(f"✅ Voice Verified!")
                    
                    cursor.execute(
                        """INSERT INTO recognition_log 
                            (person_id, similarity, verification_type) 
                            VALUES (?, ?, ?)""",
                        (recognized_person_id, float(voice_sim), 'voice')
                    )
                    conn.commit()
                    
                    current_state = STATE_AUTHENTICATED
                else:
                    print(f"❌ Voice Mismatch! Returning to face detection...")
                    current_state = STATE_FACE_DETECTION
                    recognized_person_id = None
                    recognized_person_name = None
            
            voice_verification_attempted = True
    
    # ========================================
    # STATE: AUTHENTICATED
    # ========================================
    elif current_state == STATE_AUTHENTICATED:
        cv2.putText(frame, f"Authenticated: {recognized_person_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Starting Monitoring...", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        current_state = STATE_MONITORING
        ear_counter = 0
        print(f"🔒 Authenticated: {recognized_person_name}")
        print("👁️ Starting drowsiness monitoring...")
    
    # ========================================
    # STATE: MONITORING
    # ========================================
    elif current_state == STATE_MONITORING:
        if face_mesh is None:
            cv2.putText(frame, f"Monitoring: {recognized_person_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "MediaPipe not available", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            
            try:
                mp_results = face_mesh.process(frame_rgb)
                
                if mp_results.multi_face_landmarks:
                    face_landmarks = mp_results.multi_face_landmarks[0]
                    lm = [(p.x, p.y, p.z) for p in face_landmarks.landmark]
                    
                    left_eye_idx = [33, 160, 158, 133, 153, 144]
                    right_eye_idx = [362, 385, 387, 263, 373, 380]
                    
                    left_eye = [(int(lm[i][0] * w), int(lm[i][1] * h)) 
                                for i in left_eye_idx]
                    right_eye = [(int(lm[i][0] * w), int(lm[i][1] * h)) 
                                for i in right_eye_idx]
                    
                    left_ear = eye_aspect_ratio(left_eye)
                    right_ear = eye_aspect_ratio(right_eye)
                    ear = (left_ear + right_ear) / 2.0
                    
                    for (x, y) in left_eye + right_eye:
                        cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
                    
                    pitch = estimate_head_pitch(lm, frame.shape)
                    
                    cv2.putText(frame, f"Driver: {recognized_person_name}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    if ear < EAR_THRESHOLD:
                        ear_counter += 1
                    else:
                        ear_counter = 0
                    
                    if ear_counter >= EAR_CONSEC_FRAMES:
                        cv2.putText(frame, "ALERT: Eyes Closed!", (10, 130),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                        threading.Thread(target=play_alert, daemon=True).start()
                    
                    if pitch > HEAD_PITCH_THRESHOLD:
                        cv2.putText(frame, "ALERT: Head Down!", (10, 170),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                        threading.Thread(target=play_alert, daemon=True).start()
                
                else:
                    ear_counter = 0
                    cv2.putText(frame, "Face Lost - Redetecting...", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
            except Exception as e:
                print(f"MediaPipe error: {e}")
    
    # Display
    resized_frame = cv2.resize(frame, (529, 397))
    imgBackground[303:700, 93:622] = resized_frame
    cv2.imshow("DeepVision System", imgBackground)
    
    # Keyboard
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('r'):
        print("\n🔄 Reset - Returning to Face Detection")
        current_state = STATE_FACE_DETECTION
        recognized_person_id = None
        recognized_person_name = None
        recognized_person_idx = None
        ear_counter = 0
    elif key == ord('v') or key == ord('V'):
        if current_state == STATE_FACE_RECOGNIZED:
            current_state = STATE_VOICE_VERIFICATION
            print(f"\n🎤 Starting voice verification...")

# Cleanup
cap.release()
conn.close()
cv2.destroyAllWindows()
print("\n✅ System Shutdown Complete")