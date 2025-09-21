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


# ------------------ Splash Screen ------------------ #
def update_splash(window, message):
    splash = 255 * np.ones((400, 700, 3), dtype=np.uint8)  # white screen
    cv2.putText(splash, message, (50, 200), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imshow(window, splash)
    cv2.waitKey(1)

# ------------------ Database Setup ------------------ #
conn = sqlite3.connect("faces.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    embedding BLOB
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

# ------------------ Device Setup ------------------ #
window_name = "Loading..."
update_splash(window_name, "Loading Models...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# Cosine similarity function
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# ------------------ Reference Image ------------------ #
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
            if ref_face.ndim == 3:
                ref_face = ref_face.unsqueeze(0)
            embedding = facenet(ref_face).detach().numpy()[0]
            cursor.execute("INSERT INTO faces (name, embedding) VALUES (?, ?)",(name, pickle.dumps(embedding)))
            conn.commit()
        else:
            raise ValueError(f"No face detected in {img_path}")

    # load overlay image
    disp_img = cv2.imread(img_path)
    disp_img = cv2.resize(disp_img, (414, 633))
    ref_display_images.append(disp_img)

# Load embeddings back from DB
cursor.execute("SELECT id, name, embedding FROM faces")
rows = cursor.fetchall()

ref_ids = []
ref_names = []
ref_embeddings = []
for r in rows:
    ref_ids.append(r[0])
    ref_names.append(r[1])
    ref_embeddings.append(pickle.loads(r[2]))

# ------------------ Background & Overlays ------------------ #
imgBackground = cv2.imread("background.png")

# ------------------ Webcam Init ------------------ #
update_splash(window_name, "Starting Camera...")

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

cv2.destroyWindow(window_name)

# ------------------ Main Loop ------------------ #
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
                face_tensor = face_tensor.unsqueeze(0)
                emb = facenet(face_tensor).detach().numpy()[0]

                sims = [cosine_similarity(ref_emb, emb) for ref_emb in ref_embeddings]
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

                    imgBackground[44:44 + 633, 808:808 + 414] = ref_display_images[best_idx]

    imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Face Recognition (press q to quit)", imgBackground)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
conn.close()
cv2.destroyAllWindows()
