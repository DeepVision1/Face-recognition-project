# -------------------------------
# 1. Imports
# -------------------------------
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

import librosa
import tensorflow as tf
import sounddevice as sd


# -------------------------------
# 1. Splash Screen
# -------------------------------
def update_splash(window, message):
    splash = 255 * np.ones((400, 700, 3), dtype=np.uint8)  # white screen
    cv2.putText(splash, message, (50, 200), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imshow(window, splash)
    cv2.waitKey(1)

# -------------------------------
# 2. Database Setup
# -------------------------------
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
    FOREIGN KEY(person_id) REFERENCES faces(id)
)
""")
conn.commit()

# -------------------------------
# 4. Load Models
# -------------------------------
window_name = "Loading..."
update_splash(window_name, "Loading Models...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval()

siamese_model = tf.keras.models.load_model("Models/Voice_verification_model.h5", compile=False)
embedding_model = siamese_model.layers[3]

# -------------------------------
# 5. Record_Voice
# -------------------------------
def record_voice(duration=3, sr=16000):
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    return recording.flatten()

# -------------------------------
# 6. Get Embedding
# -------------------------------
def Face_embedding(face_tensor):
    if face_tensor.ndim == 3:
        face_tensor = face_tensor.unsqueeze(0)
    return facenet(face_tensor).detach().numpy()[0]

def Voice_embedding(model, y, sr=16000, target_sec=3, n_mfcc=40):
    target_len = sr * target_sec
    if len(y) > target_len:
        start = (len(y) - target_len) // 2
        y = y[start:start+target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))
    
    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = np.mean(mfcc.T, axis=0)   # shape (40,)
    
    # reshape for Conv1D input
    mfcc = np.expand_dims(mfcc, -1)   # (40,1)
    mfcc = np.expand_dims(mfcc, 0).astype(np.float32)  # (1,40,1)

    emb = model.predict(mfcc, verbose=0)
    return tf.math.l2_normalize(emb, axis=1)

# -------------------------------
# 7. Cosine similarity
# -------------------------------
def Face_cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

cosine = tf.keras.losses.CosineSimilarity(axis=1)

# -------------------------------
# 8. Database Insertion
# -------------------------------
update_splash(window_name, "Preparing Reference Face...")

folder_path= "persons"
ref_images = glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.png"))
ref_display_images = []

# Insert embeddings into DB if not already stored
for img_path in ref_images:
    name = os.path.splitext(os.path.basename(img_path))[0]
    cursor.execute("SELECT id FROM faces WHERE name=?", (name,))
    if cursor.fetchone() is None:  # only insert if not exists
        img = Image.open(img_path).convert("RGB")
        ref_face = mtcnn(img)

        if ref_face is not None:
            embedding = Face_embedding(ref_face)
            cursor.execute("INSERT INTO faces (name, Face_embedding) VALUES (?, ?)",(name, pickle.dumps(embedding)))
            conn.commit()
        else:
            raise ValueError(f"No face detected in {img_path}")

    # load overlay image
    disp_img = cv2.imread(img_path)
    disp_img = cv2.resize(disp_img, (258, 396))
    ref_display_images.append(disp_img)

# Load embeddings back from DB
cursor.execute("SELECT id, name, Face_embedding FROM faces")
rows = cursor.fetchall()

ref_ids = []
ref_names = []
ref_embeddings = []
for r in rows:
    ref_ids.append(r[0])
    ref_names.append(r[1])
    ref_embeddings.append(pickle.loads(r[2]))
# -------------------------------
# 9. Background & Overlays
# -------------------------------
update_splash(window_name, "Starting Camera...")

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread("DeepVision_Background.jpg")

cv2.destroyWindow(window_name)

# -------------------------------
# 10. Main Loop
# -------------------------------
frame_count = 0
skip_frames = 5
scale = 0.5
boxes, probs, faces = None, None, None
threshold = 0.5
last_seen = {}  # person_id -> timestamp
cooldown = 30    # seconds to wait before re-logging same person


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % skip_frames == 0:
        small_rgb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None,fx=scale, fy=scale)
        boxes, probs = mtcnn.detect(small_rgb)
        faces = mtcnn(small_rgb)
        if boxes is not None:
            boxes /= scale

    if boxes is not None and faces is not None:
        for box, prob, face_tensor in zip(boxes, probs, faces):
            if face_tensor is not None:

                emb = Face_embedding(face_tensor)

                sims = [Face_cosine_similarity(ref_emb, emb) for ref_emb in ref_embeddings]
                best_idx = int(np.argmax(sims))
                sim = sims[best_idx]

                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"Conf:{prob:.2f}  Sim:{sim:.2f} {ref_names[best_idx]}"
                color = (0, 255, 0) if sim > threshold else (0, 0, 255)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if sim > threshold:
                    best_id = ref_ids[best_idx]
                    now = time.time()

                    # Only log if cooldown has passed
                    if best_id not in last_seen or (now - last_seen[best_id]) > cooldown:
                        cursor.execute("INSERT INTO recognition_log (person_id, similarity) VALUES (?, ?)",
                                    (best_id, float(sim)))
                        conn.commit()
                        last_seen[best_id] = now  # update last seen time
                    imgBackground[304:304 + 396, 713:713 + 258] = ref_display_images[best_idx]


    resized_frame = cv2.resize(frame, (529, 397))
    imgBackground[303:700, 93:622] = resized_frame
    cv2.imshow("Face Recognition (press q to quit)", imgBackground)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
conn.close()
cv2.destroyAllWindows()
