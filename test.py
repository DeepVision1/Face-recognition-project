import os
import glob
import random
import numpy as np
import librosa
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# --- Load Siamese model ---
siamese_model = tf.keras.models.load_model("Models/Voice_verification_model5.h5", compile=False)
embedding_model = siamese_model.layers[3]

# --- Voice embedding function ---
def Voice_embedding(model, y, sr=16000, target_sec=3.0, n_mfcc=40):
    target_len = int(sr * target_sec)
    if len(y) > target_len:
        start = (len(y) - target_len) // 2
        y = y[start:start + target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = np.mean(mfcc.T, axis=0)
    mfcc = np.expand_dims(mfcc, -1)
    mfcc = np.expand_dims(mfcc, 0).astype(np.float32)
    emb = model.predict(mfcc, verbose=0)
    emb = tf.math.l2_normalize(emb, axis=1)
    return emb.numpy().flatten()

# --- Load all voice files ---
voices_base_path = "voices"
people = [p for p in os.listdir(voices_base_path) if os.path.isdir(os.path.join(voices_base_path, p))]
voice_data = {}

for person in people:
    folder = os.path.join(voices_base_path, person)
    wavs = glob.glob(os.path.join(folder, "*.wav"))
    embeddings = []
    for w in wavs:
        y, sr = librosa.load(w, sr=16000)
        emb = Voice_embedding(embedding_model, y, sr)
        embeddings.append(emb)
    voice_data[person] = embeddings
    print(f"Loaded {len(embeddings)} samples for {person}")

# --- Generate pairs ---
def make_pairs(data, num_pairs=500):
    persons = list(data.keys())
    X1, X2, y = [], [], []
    sp1, sp2 = [], []

    # Positive pairs
    for _ in range(num_pairs // 2):
        person = random.choice(persons)
        if len(data[person]) < 2:
            continue
        a, b = random.sample(data[person], 2)
        X1.append(a)
        X2.append(b)
        y.append(1)
        sp1.append(person)
        sp2.append(person)

    # Negative pairs
    for _ in range(num_pairs // 2):
        p1, p2 = random.sample(persons, 2)
        a = random.choice(data[p1])
        b = random.choice(data[p2])
        X1.append(a)
        X2.append(b)
        y.append(0)
        sp1.append(p1)
        sp2.append(p2)

    return np.array(X1), np.array(X2), np.array(y), sp1, sp2

# --- Create pairs ---
X1, X2, y_true, sp1, sp2 = make_pairs(voice_data, num_pairs=1000)

# --- Compute cosine similarities ---
similarities = np.array([
    cosine_similarity([a], [b])[0][0] for a, b in zip(X1, X2)
])

# --- Find best threshold ---
fpr, tpr, thr = roc_curve(y_true, similarities)
best_idx = np.argmax(tpr - fpr)
best_thr = thr[best_idx]

# --- Evaluate ---
y_pred = (similarities >= best_thr).astype(int)
acc = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, similarities)

print("\n🎯 Model Evaluation (Embedding-based)")
print("---------------------------------")
print(f"Best Threshold : {best_thr:.3f}")
print(f"Accuracy       : {acc*100:.2f}%")
print(f"ROC AUC        : {auc:.3f}")

# --- Find and print wrong predictions ---
wrong_indices = np.where(y_true != y_pred)[0]

print(f"\n⚠️  Found {len(wrong_indices)} wrong predictions:\n")

for i in wrong_indices:
    true_label = y_true[i]
    pred_label = y_pred[i]
    sim = similarities[i]
    status = "False Positive" if pred_label == 1 else "False Negative"
    print(f"[{i:03d}] {sp1[i]} vs {sp2[i]} | True={true_label} | Pred={pred_label} | Sim={sim:.3f} --> {status}")

# --- Save wrong predictions to CSV ---
wrong_df = pd.DataFrame({
    "Index": wrong_indices,
    "Speaker1": [sp1[i] for i in wrong_indices],
    "Speaker2": [sp2[i] for i in wrong_indices],
    "True_Label": [y_true[i] for i in wrong_indices],
    "Pred_Label": [y_pred[i] for i in wrong_indices],
    "Similarity": [similarities[i] for i in wrong_indices],
    "Status": ["False Positive" if y_pred[i] == 1 else "False Negative" for i in wrong_indices]
})

wrong_df.to_csv("wrong_predictions.csv", index=False)
print("\n💾 Saved all wrong predictions to 'wrong_predictions.csv'")
