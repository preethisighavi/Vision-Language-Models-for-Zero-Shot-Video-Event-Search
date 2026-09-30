import os
import io
import cv2
import tempfile
import webdataset as wds
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image

TARGET_IMAGE_SIZE = (224, 224)
OUTPUT_DIR = "data/msrvtt_shards"
SHARD_SIZE = 1000

def init_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_video_bytes(video_bytes):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as temp_video:
        temp_video.write(video_bytes)
        temp_video.flush()
        
        cap = cv2.VideoCapture(temp_video.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        if fps > 0:
            # Extract 1 frame per second
            duration = int(total_frames / fps)
            for i in range(duration):
                frame_idx = int(i * fps + fps / 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret: break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame).resize(TARGET_IMAGE_SIZE, Image.Resampling.LANCZOS)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=85)
                frames.append(img_byte_arr.getvalue())
        cap.release()
        return frames

def main():
    init_dir(OUTPUT_DIR)
    print("Starting Phase 1: MSR-VTT Data Ingestion (Evaluation Data)")
    
    dataset = load_dataset("AlexZigma/msr-vtt", split="val", streaming=True)
    
    successful_downloads = 0
    pattern = os.path.join(OUTPUT_DIR, "msrvtt-test-%04d.tar")
    sink = wds.ShardWriter(pattern, maxcount=SHARD_SIZE)
    
    for item in tqdm(dataset):
        video_bytes = item.get('video')
        caption = item.get('caption', '')
        
        if video_bytes:
            frames = process_video_bytes(video_bytes)
            if frames:
                sample = {"__key__": f"{successful_downloads:08d}", "txt": caption}
                for f_idx, frame_bytes in enumerate(frames):
                    sample[f"f{f_idx:02d}.jpg"] = frame_bytes
                sink.write(sample)
                successful_downloads += 1

    sink.close()
    print(f"\n--- MSR-VTT Ingestion Complete ---")
    print(f"Total Successful Clips Processed: {successful_downloads}")

if __name__ == "__main__":
    main()
