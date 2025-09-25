import os
import numpy as np
import librosa
import tensorflow as tf
import sounddevice as sd
import time

# -------------------------------
# 1. Load the Siamese model
# -------------------------------
siamese_model = tf.keras.models.load_model("Voice_verification_model.h5", compile=False)

# Extract the embedding submodel (the part that generates embeddings from one input)
# usually input shape is (40,1) for MFCCs or (n_mels, T, 1) for spectrograms
embedding_model = siamese_model.layers[3]  # <- adjust if index is different
print(f"Using embedding submodel: {embedding_model.name}")

# -------------------------------
# 2. Get embedding for one audio clip
# -------------------------------
def get_embedding(model, y, sr=16000, target_sec=3, n_mfcc=40):
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
# 3. Record audio (live testing)
# -------------------------------
def record_voice(duration=3, sr=16000):
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    return recording.flatten()

# -------------------------------
# 4. Build reference embeddings
# -------------------------------
train_root = "voices"   # folder with subfolders per speaker
reference_embeddings = {}

for speaker in os.listdir(train_root):
    spk_dir = os.path.join(train_root, speaker)
    if not os.path.isdir(spk_dir):
        continue
    
    files = [os.path.join(spk_dir, f) for f in os.listdir(spk_dir) if f.endswith(".wav")]
    embs = []
    for f in files:
        y, _ = librosa.load(f, sr=16000)
        emb = get_embedding(embedding_model, y)
        embs.append(emb)
    reference_embeddings[speaker] = tf.concat(embs, axis=0)

# Cosine similarity
cosine = tf.keras.losses.CosineSimilarity(axis=1)

# -------------------------------
# 5. Live recognition loop
# -------------------------------
print("🎤 Starting live speaker recognition... Press Ctrl+C to stop.")
try:
    while True:
        print("\nRecording 3-second clip...")
        y = record_voice(duration=3)
        test_emb = get_embedding(embedding_model, y)

        best_sim = -1e9
        best_speaker = None

        for speaker, ref_embs in reference_embeddings.items():
            sims = -cosine(test_emb, ref_embs)  # cosine similarity (negative because TF CosineSimilarity returns negative values)
            sim_val = tf.reduce_max(sims)       # best match among references
            if sim_val > best_sim:
                best_sim = sim_val
                best_speaker = speaker

        print(f"✅ Predicted Speaker: {best_speaker}, Similarity: {best_sim.numpy():.4f}")
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n🛑 Live recognition stopped.")