from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from insightface.app import FaceAnalysis

import numpy as np
import cv2
import os
import pickle
import zipfile
import base64
import threading
from typing import List, Dict, Optional

# ---------------- CONFIG ----------------
KNOWN_DIR = "LabourPhoto"
ZIP_NAME = "LabourPhoto.zip"
EMBEDDING_FILE = "faces.pkl"
THRESHOLD = 0.45

# ---------------- APP ----------------
app = FastAPI(title="Face Recognition System", version="4.0")

face_app = None
known_faces: Dict[str, List[np.ndarray]] = {}

total_images = 0

# ---------------- TRAINING STATUS ----------------
training_status = {
    "running": False,
    "progress": 0,
    "total": 0,
    "done": 0
}

# ---------------- MODEL ----------------
def load_model():
    global face_app
    face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    print("[OK] Model Loaded")


# ---------------- NORMALIZE ----------------
def normalize(v):
    return v / np.linalg.norm(v)


# ---------------- LOAD DATA ----------------
def load_faces():
    global known_faces, total_images

    if os.path.exists(EMBEDDING_FILE):
        with open(EMBEDDING_FILE, "rb") as f:
            data = pickle.load(f)
            known_faces = data.get("faces", {})
            total_images = data.get("image_count", 0)
    else:
        known_faces = {}
        total_images = 0

    print(f"[OK] Loaded {len(known_faces)} people | {total_images} images")


# ---------------- TRAIN FUNCTION ----------------
def train_faces():
    global known_faces, total_images, training_status

    if not os.path.exists(KNOWN_DIR):
        training_status["running"] = False
        print(f"[ERROR] Folder not found: {KNOWN_DIR}")
        return

    files = [
        f for f in os.listdir(KNOWN_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    total_files = len(files)

    if total_files == 0:
        print("[ERROR] No images found")
        training_status["running"] = False
        return

    print("\n========================================")
    print("[INFO] TRAINING STARTED")
    print(f"[INFO] Total Images Found: {total_files}")
    print("========================================\n")

    processed = 0
    failed = 0

    training_status["running"] = True
    training_status["total"] = total_files
    training_status["done"] = 0
    training_status["progress"] = 0

    for index, file in enumerate(files, start=1):

        try:
            path = os.path.join(KNOWN_DIR, file)

            img = cv2.imread(path)

            if img is None:
                failed += 1
                print(f"[FAILED] {file} -> Unable to read image")
                continue

            detected = face_app.get(img)

            if not detected:
                failed += 1
                print(f"[FAILED] {file} -> No face detected")
                continue

            name = os.path.splitext(file)[0]

            emb = normalize(detected[0].normed_embedding)

            if name not in known_faces:
                known_faces[name] = []

            known_faces[name].append(emb)

            processed += 1
            total_images += 1

        except Exception as ex:
            failed += 1
            print(f"[ERROR] {file}: {str(ex)}")

        training_status["done"] = index
        training_status["progress"] = int((index / total_files) * 100)

        print(
            f"[TRAINING] {index}/{total_files} "
            f"({training_status['progress']}%) | "
            f"Success={processed} | "
            f"Failed={failed} | "
            f"{file}"
        )

    with open(EMBEDDING_FILE, "wb") as f:
        pickle.dump({
            "faces": known_faces,
            "image_count": total_images
        }, f)

    training_status["running"] = False
    training_status["progress"] = 100

    print("\n========================================")
    print("[SUCCESS] TRAINING COMPLETED")
    print(f"Total Images Found : {total_files}")
    print(f"Successfully Trained : {processed}")
    print(f"Failed Images : {failed}")
    print(f"Total Faces Saved : {len(known_faces)}")
    print(f"Total Images Stored : {total_images}")
    print("========================================\n")

# ---------------- ZIP UPLOAD ----------------
@app.post("/upload-zip")
async def upload_zip(file: UploadFile = File(...)):

    zip_path = ZIP_NAME

    with open(zip_path, "wb") as f:
        f.write(await file.read())

    if os.path.exists(KNOWN_DIR):
        import shutil
        shutil.rmtree(KNOWN_DIR)

    os.makedirs(KNOWN_DIR, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(KNOWN_DIR)

    return {"status": "ZIP extracted", "folder": KNOWN_DIR}


# ---------------- TRAIN API ----------------
@app.post("/train")
def train():
    if face_app is None:
        return {"error": "model not loaded"}

    thread = threading.Thread(target=train_faces)
    thread.start()

    return {"status": "training started (background mode)"}


# ---------------- TRAIN STATUS ----------------
@app.get("/train-status")
def train_status():
    return training_status


# ---------------- STATUS API ----------------
@app.get("/status")
def status():
    return {
        "people_in_system": len(known_faces),
        "total_images_trained": total_images
    }


# ---------------- RECOGNIZE ----------------
@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):

    if face_app is None:
        return JSONResponse(
            status_code=500,
            content={"error": "model not loaded"}
        )

    if not known_faces:
        return JSONResponse(
            status_code=400,
            content={"error": "no trained data"}
        )

    image_bytes = await file.read()

    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid image"}
        )

    faces = face_app.get(img)

    results = []
    recognized_count = 0

    for f in faces:

        emb = normalize(f.normed_embedding)

        best_name = None
        best_score = -1

        # Compare against all trained embeddings
        for name, emb_list in known_faces.items():

            for ref in emb_list:

                score = float(np.dot(emb, ref))

                if score > best_score:
                    best_score = score
                    best_name = name

        confidence_percent = round(best_score * 100, 2)

        # Only count as recognized if >= threshold
        if best_score >= THRESHOLD:

            recognized_count += 1

            results.append({
                "name": best_name,
                "confidence": confidence_percent,
                "bbox": f.bbox.tolist()
            })

    return {
        "faces_detected": len(faces),
        "recognized_faces": recognized_count,
        "recognized": results
    }
# ---------------- STARTUP ----------------
@app.on_event("startup")
def startup():
    load_model()
    load_faces()
    print("[READY] System running")


# ---------------- RUN ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)