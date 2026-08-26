import os
import uuid
import cv2
import torch
import numpy as np
from typing import Optional
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from transformers import CLIPProcessor, CLIPModel
import chromadb

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
VIDEOS_DIR = "videos"
CHROMA_DB_DIR = "chroma_db"
STATIC_DIR = "static"
FRAMES_PER_SECOND = 1          # How many frames to extract per second of video
MODEL_NAME = "openai/clip-vit-base-patch32"

# ─────────────────────────────────────────────
# Ensure directories exist
# ─────────────────────────────────────────────
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Device Selection — CPU is most stable across all Mac setups.
# MPS (Apple Silicon) can cause indefinite hangs during model init.
# ─────────────────────────────────────────────
DEVICE = torch.device("cpu")
print("⚙️  Using CPU for inference.")

# ─────────────────────────────────────────────
# Load CLIP Model (done once at startup)
# ─────────────────────────────────────────────
print(f"Loading CLIP model: {MODEL_NAME} ...")
clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
clip_model.eval()
print("✅ CLIP model loaded and ready.")

# ─────────────────────────────────────────────
# Initialize ChromaDB
# ─────────────────────────────────────────────
print("Initializing ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = chroma_client.get_or_create_collection(
    name="video_frames",
    metadata={"hnsw:space": "cosine"}
)
print("✅ ChromaDB initialized and 'video_frames' collection is ready.")

# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(title="Zero-Shot Video Search Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/videos", StaticFiles(directory=VIDEOS_DIR), name="videos")


# ─────────────────────────────────────────────
# Helper: Extract Frames from Video
# ─────────────────────────────────────────────
def extract_frames(video_path: str, fps: int = FRAMES_PER_SECOND) -> list[dict]:
    """
    Extracts frames from a video at the specified frames-per-second rate.
    Returns a list of dicts: {timestamp_sec: float, frame: PIL.Image}.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / fps))  # e.g. extract every 30th frame for 1fps

    frames = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            timestamp_sec = round(frame_index / video_fps, 2)
            # Convert BGR (OpenCV) → RGB → PIL Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            frames.append({"timestamp_sec": timestamp_sec, "frame": pil_image})

        frame_index += 1

    cap.release()
    print(f"   Extracted {len(frames)} frames from video.")
    return frames


# ─────────────────────────────────────────────
# Helper: Generate CLIP Image Embeddings
# ─────────────────────────────────────────────
def get_image_embeddings(frames: list[dict]) -> list[dict]:
    """
    Runs PIL images through CLIP to generate normalized image embeddings.
    Returns original frame dicts enriched with 'embedding'.
    """
    pil_images = [f["frame"] for f in frames]

    # Process in batches of 32 to avoid OOM
    batch_size = 32
    all_embeddings = []

    for i in range(0, len(pil_images), batch_size):
        batch = pil_images[i : i + batch_size]
        inputs = clip_processor(images=batch, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            # Use low-level submodel API — always returns plain tensors regardless of transformers version
            vision_outputs = clip_model.vision_model(pixel_values=inputs["pixel_values"])
            pooled = vision_outputs.pooler_output          # (batch, 768) plain tensor
            image_features = clip_model.visual_projection(pooled)  # (batch, 512) plain tensor
            # L2-normalize for cosine similarity
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        all_embeddings.extend(image_features.cpu().float().numpy().tolist())

    for i, frame in enumerate(frames):
        frame["embedding"] = all_embeddings[i]

    return frames


# ─────────────────────────────────────────────
# Helper: Store Embeddings in ChromaDB
# ─────────────────────────────────────────────
def store_embeddings(video_id: str, video_filename: str, frames: list[dict]):
    """
    Stores frame embeddings and metadata into the ChromaDB collection.
    Each frame is identified by a unique ID: {video_id}_{timestamp}.
    """
    ids = []
    embeddings = []
    metadatas = []

    for frame in frames:
        frame_id = f"{video_id}_{frame['timestamp_sec']}"
        ids.append(frame_id)
        embeddings.append(frame["embedding"])
        metadatas.append({
            "video_id": video_id,
            "video_filename": video_filename,
            "timestamp_sec": frame["timestamp_sec"],
        })

    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
    print(f"   Stored {len(ids)} frame embeddings for video_id={video_id}.")


# ─────────────────────────────────────────────
# API: Serve Frontend
# ─────────────────────────────────────────────
@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


# ─────────────────────────────────────────────
# API: Health Check
# ─────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {
        "status": "online",
        "device": str(DEVICE),
        "clip_model": MODEL_NAME,
        "total_indexed_frames": collection.count(),
    }


# ─────────────────────────────────────────────
# API: Upload Video & Process
# ─────────────────────────────────────────────
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Accepts a video file upload, saves it locally (synced to Google Drive),
    extracts frames, generates CLIP embeddings, and stores them in ChromaDB.
    """
    # Validate file type
    allowed_types = {"video/mp4", "video/avi", "video/mov", "video/quicktime", "video/x-msvideo"}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}. Please upload an MP4, AVI, or MOV file.")

    # Generate a unique video ID and save the file
    video_id = str(uuid.uuid4())
    safe_filename = f"{video_id}_{file.filename}"
    video_path = os.path.join(VIDEOS_DIR, safe_filename)

    print(f"\n📥 Received upload: {file.filename} → saving to {video_path}")
    content = await file.read()
    with open(video_path, "wb") as f:
        f.write(content)
    print(f"   Saved video ({len(content) / 1_000_000:.2f} MB).")

    # Extract frames, generate embeddings, store in ChromaDB
    try:
        print("🎞️  Extracting frames...")
        frames = extract_frames(video_path)

        if not frames:
            raise HTTPException(status_code=422, detail="Could not extract any frames from the video.")

        print("🧠  Generating CLIP embeddings...")
        frames_with_embeddings = get_image_embeddings(frames)

        print("💾  Storing embeddings in ChromaDB...")
        store_embeddings(video_id, safe_filename, frames_with_embeddings)

    except Exception as e:
        # Clean up the saved video on failure
        if os.path.exists(video_path):
            os.remove(video_path)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    return JSONResponse({
        "status": "success",
        "video_id": video_id,
        "filename": safe_filename,
        "frames_indexed": len(frames),
        "video_url": f"/videos/{safe_filename}",
    })


# ─────────────────────────────────────────────
# Helper: Generate CLIP Text Embedding
# ─────────────────────────────────────────────
def get_text_embedding(query: str) -> list[float]:
    """
    Encodes a natural language query into a normalized CLIP text embedding.
    """
    inputs = clip_processor(
        text=[query], return_tensors="pt", padding=True, truncation=True
    ).to(DEVICE)
    with torch.no_grad():
        # Use low-level submodel API — always returns plain tensors regardless of transformers version
        text_outputs = clip_model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        pooled = text_outputs.pooler_output               # (1, 512) plain tensor
        text_features = clip_model.text_projection(pooled)  # (1, 512) plain tensor
        # L2-normalize for cosine similarity
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    return text_features.cpu().float().numpy().tolist()[0]


# ─────────────────────────────────────────────
# API: Search Videos by Natural Language Query
# ─────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5                   # Number of results to return
    video_id: Optional[str] = None   # Optional: scope search to one video


@app.post("/search")
def search_video(request: SearchRequest):
    """
    Takes a natural language query (e.g. 'a dog running'),
    encodes it with CLIP, and retrieves the top-K most relevant
    video frames from ChromaDB, returning timestamps and video URLs.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if collection.count() == 0:
        raise HTTPException(
            status_code=404,
            detail="No videos have been indexed yet. Please upload a video first.",
        )

    print(f"\n🔍 Search query: '{request.query}' | top_k={request.top_k}")

    # Optional where-filter to scope to a specific video
    where_filter = {"video_id": request.video_id} if request.video_id else None

    # Encode the text query into a CLIP embedding
    query_embedding = get_text_embedding(request.query)

    # Query ChromaDB for nearest neighbours
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(request.top_k, collection.count()),
        where=where_filter,
        include=["metadatas", "distances"],
    )

    # Format results into a clean response
    hits = []
    for meta, distance in zip(results["metadatas"][0], results["distances"][0]):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite.
        # Convert to a 0–1 similarity score.
        similarity = round(1 - (distance / 2), 4)
        hits.append({
            "video_id": meta["video_id"],
            "video_filename": meta["video_filename"],
            "video_url": f"/videos/{meta['video_filename']}",
            "timestamp_sec": meta["timestamp_sec"],
            "similarity_score": similarity,
        })

    print(f"   Returning {len(hits)} results for query: '{request.query}'")
    return JSONResponse({"query": request.query, "results": hits})


# ─────────────────────────────────────────────
# API: List All Indexed Videos
# ─────────────────────────────────────────────
@app.get("/videos-list")
def list_indexed_videos():
    """
    Returns a deduplicated list of all videos that have been indexed in ChromaDB.
    """
    total = collection.count()
    if total == 0:
        return JSONResponse({"videos": [], "total_frames_indexed": 0})

    all_results = collection.get(include=["metadatas"])
    seen: dict = {}
    for meta in all_results["metadatas"]:
        if not meta:
            continue
            
        vid = meta.get("video_id")
        video_filename = meta.get("video_filename")
        
        if vid and video_filename and vid not in seen:
            seen[vid] = {
                "video_id": vid,
                "video_filename": video_filename,
                "video_url": f"/videos/{video_filename}",
            }

    return JSONResponse({
        "videos": list(seen.values()),
        "total_frames_indexed": total,
    })
