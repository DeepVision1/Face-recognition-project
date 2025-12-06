import gradio as gr
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
import librosa
import tensorflow as tf
import mediapipe as mp
import sounddevice as sd
import scipy.signal as signal
import soundfile as sf

# ========================================
# CONFIGURATION
# ========================================
FACE_THRESHOLD = 0.6
VOICE_THRESHOLD = 0.8
VOICE_DURATION = 3
VOICE_SR = 16000
EAR_THRESHOLD = 0.20
EAR_CONSEC_FRAMES = 15
HEAD_CONSEC_FRAMES = 15
HEAD_PITCH_THRESHOLD = 15.0

# States
STATE_FACE_DETECTION = 0
STATE_FACE_RECOGNIZED = 1
STATE_VOICE_VERIFICATION = 2
STATE_AUTHENTICATED = 3
STATE_MONITORING = 4

# ========================================
# GLOBAL VARIABLES
# ========================================
class SystemState:
    def __init__(self):
        self.current_state = STATE_FACE_DETECTION
        self.recognized_person_id = None
        self.recognized_person_name = None
        self.recognized_person_idx = None
        self.ear_counter = 0
        self.head_counter = 0
        self.frame_count = 0
        self.last_alert_time = 0
        self.status_message = "System Initializing..."
        self.video_running = False
        self.cap = None
        
state = SystemState()

# ========================================
# DATABASE SETUP
# ========================================
def setup_database():
    os.makedirs("Database", exist_ok=True)
    conn = sqlite3.connect("Database/faces.db", check_same_thread=False)
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
    return conn, cursor

# ========================================
# LOAD MODELS
# ========================================
def load_models():
    print("Loading models...")
    
    # Face models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    # Voice model
    siamese_model = tf.keras.models.load_model("Models/Voice_verification_model5.h5", compile=False)
    embedding_model = siamese_model.layers[3]
    
    # Drowsiness models
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    print("✅ All models loaded successfully")
    return mtcnn, facenet, embedding_model, face_mesh, device

# ========================================
# HELPER FUNCTIONS
# ========================================
def Face_embedding(face_tensor, facenet, device):
    if face_tensor.ndim == 3:
        face_tensor = face_tensor.unsqueeze(0)
    face_tensor = face_tensor.to(device)
    return facenet(face_tensor).detach().cpu().numpy()[0]

def Face_cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

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

cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)

def Voice_cosine_similarity(a, b):
    return -cosine_loss(a, b).numpy()

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
# LOAD REFERENCE DATA
# ========================================
def load_reference_data(conn, cursor, mtcnn, facenet, embedding_model, device):
    # Load face images
    folder_path = "persons"
    if os.path.exists(folder_path):
        ref_images = glob.glob(os.path.join(folder_path, "*.jpg")) + \
                    glob.glob(os.path.join(folder_path, "*.png"))
        
        for img_path in ref_images:
            name = os.path.splitext(os.path.basename(img_path))[0]
            cursor.execute("SELECT id FROM faces WHERE name=?", (name,))
            if cursor.fetchone() is None:
                img = Image.open(img_path).convert("RGB")
                ref_face = mtcnn(img)
                
                if ref_face is not None:
                    embedding = Face_embedding(ref_face, facenet, device)
                    cursor.execute(
                        "INSERT INTO faces (name, Face_embedding) VALUES (?, ?)",
                        (name, pickle.dumps(embedding))
                    )
                    conn.commit()
                    print(f"✅ Added face for: {name}")
    
    # Load voice data
    voices_base_path = "voices"
    if os.path.exists(voices_base_path):
        voice_folders = [f for f in os.listdir(voices_base_path) 
                        if os.path.isdir(os.path.join(voices_base_path, f))]
        
        for person_folder in voice_folders:
            person_name = person_folder
            folder_path = os.path.join(voices_base_path, person_folder)
            
            cursor.execute("SELECT id, Voice_embedding FROM faces WHERE name=?", (person_name,))
            result = cursor.fetchone()
            
            if result is None or result[1] is not None:
                continue
            
            person_id = result[0]
            audio_files = glob.glob(os.path.join(folder_path, "*.wav"))
            
            if audio_files:
                voice_embeddings = []
                for audio_file in audio_files:
                    try:
                        audio_data, sr = librosa.load(audio_file, sr=VOICE_SR)
                        emb = Voice_embedding(embedding_model, audio_data, sr=sr)
                        voice_embeddings.append(emb.numpy())
                    except Exception as e:
                        continue
                
                if voice_embeddings:
                    avg_embedding = np.mean(voice_embeddings, axis=0)
                    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
                    
                    cursor.execute("UPDATE faces SET Voice_embedding=? WHERE id=?",
                                (pickle.dumps(avg_embedding), person_id))
                    conn.commit()
                    print(f"✅ Added voice for: {person_name}")
    
    # Load database references
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
    
    return ref_ids, ref_names, ref_face_embeddings, ref_voice_embeddings

# ========================================
# INITIALIZE SYSTEM
# ========================================
print("Initializing system...")
conn, cursor = setup_database()
mtcnn, facenet, embedding_model, face_mesh, device = load_models()
ref_ids, ref_names, ref_face_embeddings, ref_voice_embeddings = load_reference_data(
    conn, cursor, mtcnn, facenet, embedding_model, device
)

print(f"\n📋 Loaded {len(ref_names)} persons from database")

# ========================================
# MAIN PROCESSING FUNCTIONS
# ========================================
def process_face_detection(frame):
    """Process frame for face detection and recognition"""
    
    # Detect faces
    boxes, probs = mtcnn.detect(frame)
    faces = mtcnn(frame)
    
    if boxes is not None and faces is not None and len(ref_face_embeddings) > 0:
        for box, prob, face_tensor in zip(boxes, probs, faces):
            if face_tensor is not None:
                try:
                    emb = Face_embedding(face_tensor, facenet, device)
                    
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
                        state.recognized_person_id = ref_ids[best_idx]
                        state.recognized_person_name = ref_names[best_idx]
                        state.recognized_person_idx = best_idx
                        state.current_state = STATE_FACE_RECOGNIZED
                        state.status_message = f"✅ Face Recognized: {ref_names[best_idx]}"
                        
                        cursor.execute(
                            "INSERT INTO recognition_log (person_id, similarity, verification_type) VALUES (?, ?, ?)",
                            (state.recognized_person_id, float(sim), 'face')
                        )
                        conn.commit()
                        
                        return frame, state.status_message, True
                
                except Exception as e:
                    print(f"Error: {e}")
                    continue
    
    cv2.putText(frame, "Looking for faces...", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    return frame, "🔍 Searching for faces...", False

def process_voice_verification(audio_data):
    """Process voice for verification"""
    if audio_data is None or len(audio_data) == 0:
        return "❌ No audio received", False
    
    audio_data = audio_data.astype(float)
    
    #Band-pass filter (to remove noise outside voice range) ---
    lowcut = 60.0
    highcut = 6000.0

    nyquist = 0.5 * VOICE_SR
    b, a = signal.butter(3, [lowcut / nyquist, highcut / nyquist], btype='band')
    filtered_audio = signal.lfilter(b, a, audio_data)

    #clean_audio = nr.reduce_noise(y=audio_data, sr=VOICE_SR)

    # Silence detection ---
    energy = np.mean(filtered_audio**2)
    silence_threshold = 1e-4

    test_voice_emb = Voice_embedding(embedding_model, audio_data, sr=VOICE_SR)    #filter audio
    
    ref_voice_emb = ref_voice_embeddings[state.recognized_person_idx]
    
    if ref_voice_emb is None:
        state.current_state = STATE_FACE_DETECTION
        state.recognized_person_id = None
        state.recognized_person_name = None
        return "⚠️ No voice reference available. Proceeding to monitoring...", True
    
    voice_sim = Voice_cosine_similarity(test_voice_emb, ref_voice_emb.reshape(1, -1))
    
    if (voice_sim > VOICE_THRESHOLD) and (energy > silence_threshold):
        cursor.execute(
            "INSERT INTO recognition_log (person_id, similarity, verification_type) VALUES (?, ?, ?)",
            (state.recognized_person_id, float(voice_sim), 'voice')
        )
        conn.commit()
        
        state.current_state = STATE_MONITORING
        return f"✅ Voice Verified! Similarity: {voice_sim:.4f}", True
    
    elif (energy < silence_threshold):
        state.current_state = STATE_FACE_DETECTION
        return "🤫 No voice detected (silence)", False
    
    else:
        state.current_state = STATE_FACE_DETECTION
        state.recognized_person_id = None
        state.recognized_person_name = None
        return f"❌ Voice Mismatch! Similarity: {voice_sim:.4f}", False

def process_monitoring(frame):
    """Process frame for drowsiness monitoring"""
    h, w = frame.shape[:2]
    
    mp_results = face_mesh.process(frame)
    
    alert_messages = []
    
    if mp_results.multi_face_landmarks:
        face_landmarks = mp_results.multi_face_landmarks[0]
        lm = [(p.x, p.y, p.z) for p in face_landmarks.landmark]
        
        # Eye landmarks
        left_eye_idx = [33, 160, 158, 133, 153, 144]
        right_eye_idx = [362, 385, 387, 263, 373, 380]
        
        left_eye = [(int(lm[i][0] * w), int(lm[i][1] * h)) for i in left_eye_idx]
        right_eye = [(int(lm[i][0] * w), int(lm[i][1] * h)) for i in right_eye_idx]
        
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        # Draw eye landmarks
        # for (x, y) in left_eye + right_eye:
        #     cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
        
        # Head pitch
        pitch = estimate_head_pitch(lm, frame.shape)
        
        # Display info
        cv2.putText(frame, f"Driver: {state.recognized_person_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Check drowsiness
        if ear < EAR_THRESHOLD:
            state.ear_counter += 1
        else:
            state.ear_counter = 0
        
        if pitch > HEAD_PITCH_THRESHOLD:
            state.head_counter += 1
        else:
            state.head_counter = 0
        
        if state.ear_counter >= EAR_CONSEC_FRAMES:
            cv2.putText(frame, "ALERT: Eyes Closed!", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            alert_messages.append("🚨 DROWSINESS DETECTED: Eyes closed!")
        
        if state.head_counter >= HEAD_CONSEC_FRAMES:
            cv2.putText(frame, "ALERT: Head Down!", (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            alert_messages.append("🚨 DROWSINESS DETECTED: Head down!")
        
        status = " | ".join(alert_messages) if alert_messages else "✅ Driver alert and attentive"
        
    else:
        cv2.putText(frame, "Face Lost", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        status = "⚠️ Face not detected in frame"
    
    return frame, status

# ========================================
# VIDEO STREAM GENERATOR
# ========================================
def video_stream():
    """Generate video frames with automatic processing"""
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    voice_recording = None
    recording_start = None
    
    while state.video_running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        state.frame_count += 1
        
        # FACE DETECTION STATE
        if state.current_state == STATE_FACE_DETECTION:
            frame, status, recognized = process_face_detection(frame)
            state.status_message = status
            
            # Auto-transition to voice verification
            if recognized:
                state.current_state = STATE_VOICE_VERIFICATION
                state.status_message = f"🎤 Recording voice for {VOICE_DURATION} seconds..."
                recording_start = time.time()
                
                # Start recording in background
                print(f"🎤 Starting voice recording for {state.recognized_person_name}...")
                voice_recording = sd.rec(int(VOICE_DURATION * VOICE_SR), 
                                        samplerate=VOICE_SR, 
                                        channels=1,
                                        dtype='float32')
        
        # VOICE VERIFICATION STATE
        elif state.current_state == STATE_VOICE_VERIFICATION:
            cv2.putText(frame, f"Recording Voice... {state.recognized_person_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            elapsed = time.time() - recording_start if recording_start else 0
            remaining = max(0, VOICE_DURATION - elapsed)
            cv2.putText(frame, f"Speak now! {remaining:.1f}s", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            state.status_message = f"🎤 Recording... {remaining:.1f}s remaining"
            
            # Check if recording is complete
            if elapsed >= VOICE_DURATION and voice_recording is not None:
                sd.wait()  # Wait for recording to finish
                audio_data = voice_recording.flatten()
                
                print(f"✅ Recording complete. Processing voice...")
                message, verified = process_voice_verification(audio_data)
                state.status_message = message
                print(message)
                
                voice_recording = None
                recording_start = None
        
        # MONITORING STATE
        elif state.current_state == STATE_MONITORING:
            frame, status = process_monitoring(frame)
            state.status_message = status
        
        yield frame, state.status_message
        time.sleep(0.03)  # ~30 FPS
    
    cap.release()

# ========================================
# GRADIO INTERFACE FUNCTIONS
# ========================================
def start_video():
    """Start video processing"""
    state.video_running = True
    state.current_state = STATE_FACE_DETECTION
    return "✅ Video started. Looking for faces..."

def stop_video():
    """Stop video processing"""
    state.video_running = False
    return "⏹️ Video stopped."

def reset_system():
    """Reset system to initial state"""
    state.current_state = STATE_FACE_DETECTION
    state.recognized_person_id = None
    state.recognized_person_name = None
    state.recognized_person_idx = None
    state.ear_counter = 0
    state.status_message = "System reset. Ready for face detection."
    return "✅ System reset successfully!"

def get_logs():
    """Get recent recognition logs"""
    cursor.execute("""
        SELECT r.timestamp, f.name, r.similarity, r.verification_type
        FROM recognition_log r
        JOIN faces f ON r.person_id = f.id
        ORDER BY r.timestamp DESC
        LIMIT 10
    """)
    
    logs = cursor.fetchall()
    if not logs:
        return "No logs yet"
    
    log_text = "Recent Recognition Events:\n" + "="*60 + "\n"
    for log in logs:
        log_text += f"{log[0]} | {log[1]} | {log[3].upper()} | Similarity: {log[2]:.4f}\n"
    
    return log_text

# ========================================
# CREATE GRADIO INTERFACE
# ========================================
with gr.Blocks(title="DeepVision Car Authentication System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚗 DeepVision Car Authentication & Drowsiness Detection System
    
    **Automatic Multi-modal Biometric Authentication with Real-time Driver Monitoring**
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            video_output = gr.Image(label="Live Camera Feed", streaming=True)
            status_output = gr.Textbox(label="System Status", lines=2, interactive=False)
            
            with gr.Row():
                start_btn = gr.Button("▶️ Start System", variant="primary", size="lg")
                stop_btn = gr.Button("⏹️ Stop", variant="stop", size="lg")
                reset_btn = gr.Button("🔄 Reset", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### System Information")
            
            gr.Markdown(f"""
            **Registered Users:** {len(ref_names)}
            
            **Detection Thresholds:**
            - Face Similarity: {FACE_THRESHOLD}
            - Voice Similarity: {VOICE_THRESHOLD}
            - Eye Aspect Ratio: {EAR_THRESHOLD}
            - Head Pitch: {HEAD_PITCH_THRESHOLD}°
            """)
            
            with gr.Accordion("Recognition Logs", open=False):
                logs_btn = gr.Button("📋 Refresh Logs")
                logs_output = gr.Textbox(label="Recent Events", lines=12)
            
            reset_status = gr.Textbox(label="Action Status", lines=1)
    
    gr.Markdown("""
    ### 🔄 Automatic Workflow:
    
    1. **▶️ Click "Start System"** - Camera activates and begins face detection
    2. **👤 Face Detection** - System automatically recognizes registered faces
    3. **🎤 Voice Recording** - Automatically records for 3 seconds when face is detected
    4. **✅ Authentication** - Voice is verified automatically
    5. **👁️ Monitoring** - Drowsiness detection starts immediately after authentication
    
    ### ⚠️ Alerts:
    - 🚨 **Eyes Closed** - Triggered after 15 consecutive frames with closed eyes
    - 🚨 **Head Down** - Triggered when head pitch exceeds threshold
    
    ### 📁 Required Setup:
    - Face images in `persons/` folder (e.g., `persons/John.jpg`)
    - Voice samples in `voices/John/*.wav`
    - All models in `Models/` folder
    
    **No manual intervention needed - the system handles everything automatically!**
    """)
    
    # Event handlers
    start_btn.click(
        fn=start_video,
        outputs=[reset_status]
    ).then(
        fn=video_stream,
        outputs=[video_output, status_output]
    )
    
    stop_btn.click(
        fn=stop_video,
        outputs=[reset_status]
    )
    
    reset_btn.click(
        fn=reset_system,
        outputs=[reset_status]
    )
    
    logs_btn.click(
        fn=get_logs,
        outputs=[logs_output]
    )

# ========================================
# LAUNCH
# ========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 DEEPVISION GRADIO INTERFACE - AUTOMATIC MODE")
    print("="*60)
    print(f"Loaded {len(ref_names)} registered users")
    for i, name in enumerate(ref_names):
        has_voice = "✅" if ref_voice_embeddings[i] is not None else "❌"
        print(f"  {i+1}. {name} - Voice: {has_voice}")
    print("="*60)
    print("\n⚡ Auto-detection enabled:")
    print("  - Face detection → Voice recording (3s) → Authentication")
    print("  - No button clicks needed after starting!")
    print("="*60 + "\n")

    dummy_audio = np.zeros(int(VOICE_SR * 1.0), dtype=float)                 #1 sec of silence
    _ = Voice_embedding(embedding_model, dummy_audio, sr=VOICE_SR)           #Fixed Voice Model Lag
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )