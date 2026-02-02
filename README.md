# DeepVision — Face + Voice Authentication Suite

DeepVision is a **finished, end‑to‑end biometric authentication project** that combines face recognition, voice verification, and real‑time attention monitoring into a single interactive system. The goal is to provide a practical, demo‑ready pipeline that can be run locally or backed by a cloud database.

## Highlights

- **Multi‑modal authentication**: face recognition + voice verification for stronger identity checks.
- **Real‑time monitoring**: eye aspect ratio (EAR) and head‑pitch cues for attention/drowsiness signals.
- **Gradio UI**: a ready‑to‑run interface for demos and validation.
- **Two storage backends**:
  - **Local SQLite** for offline/demo usage.
  - **Azure SQL** for cloud deployments.

## Project structure

```
.
├── mygradio2.py           # Main application (SQLite-backed)
├── mygradio.py            # Azure SQL variant (pyodbc)
├── Models/                # Voice verification models (.h5)
├── persons/               # Reference face images (name-based)
├── voices/                # Reference voice samples (per-person folders)
├── Database/              # SQLite database (faces.db)
├── test.py                # Voice model evaluation/threshold script
└── requirements*.txt      # Python dependencies
```

## How the system works (at a glance)

1. **Face detection + embedding**  
   MTCNN detects faces and FaceNet generates embeddings.
2. **Face match**  
   Embeddings are compared against the database via cosine similarity.
3. **Voice verification**  
   A Siamese model produces voice embeddings for similarity checks.
4. **Attention monitoring**  
   Eye aspect ratio and head pitch provide basic drowsiness cues.

## Setup (local, finished workflow)

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add reference data**
   - Put face images in `persons/` named like `person_name.jpg`.
   - Put voice samples in `voices/person_name/*.wav`.

3. **Run the main app**
   ```bash
   python mygradio2.py
   ```

This launches the full Gradio UI and automatically seeds the local database (`Database/faces.db`).

## Azure SQL deployment

Use `mygradio.py` when you want a cloud-backed deployment:

```bash
python mygradio.py
```

> Note: You must provide valid Azure SQL credentials and have ODBC drivers installed.

## Model notes

- The system expects a trained voice model at:
  ```
  Models/Voice_verification_model5.h5
  ```
- Additional model versions are stored in `Models/` for experimentation.

## Evaluation & validation

Use `test.py` to evaluate the voice model, compute a similarity threshold, and export misclassifications:

```bash
python test.py
```

## Status

✅ **Complete** — This repository represents a finished, working project with a runnable UI, local/cloud database support, and evaluation utilities.

---

If you are new to the code, start with **`mygradio2.py`** to see the full, working pipeline from capture → recognition → verification → monitoring.
