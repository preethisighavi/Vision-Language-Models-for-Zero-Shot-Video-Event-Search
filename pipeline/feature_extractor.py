import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

from pipeline.config import Config
from pipeline.data_cleaner import DataCleaner
from pipeline.video_processor import VideoProcessor

class FeatureExtractor:
    def __init__(self):
        print(f"Loading {Config.CLIP_MODEL_NAME} architecture...")
        
        # Determine strict hardware bounds
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps" # Apple Silicon mapping
        else:
            self.device = "cpu"
            
        print(f"Inference Device: {self.device.upper()}")
        
        # Setup Huggingface Preprocessors
        self.model = CLIPModel.from_pretrained(Config.CLIP_MODEL_NAME).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(Config.CLIP_MODEL_NAME)
        self.video_processor = VideoProcessor(fps_downsample_rate=Config.FRAMES_PER_SECOND)
        
    def _extract_text_embeddings(self, captions):
        """ Tokenizes arrays of text and evaluates via CLIPTextModel branch """
        # Natively enforces 77-token hard clipping boundary mapping via `truncation=True`
        inputs = self.processor(text=captions, return_tensors="pt", 
                                max_length=Config.MAX_TEXT_TOKENS, 
                                padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            text_embeds = self.model.get_text_features(**inputs)
            
        # Standardize representation size and move out of memory buffer
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds.cpu().numpy()

    def _extract_video_embeddings(self, frame_list):
        """ Evaluates a list of RGB PIL Images via CLIPVisionModel branch """
        if not frame_list:
            return np.zeros(512)
            
        inputs = self.processor(images=frame_list, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            image_embeds = self.model.get_image_features(**inputs)
            
        # Frame-wise embeddings (e.g. 15 tensors of size 512)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        
        # Mean Pooling to derive *one* absolute vector per video clip
        video_clip_embedding = image_embeds.mean(dim=0)
        video_clip_embedding = video_clip_embedding / video_clip_embedding.norm(dim=-1, keepdim=True)
        
        return video_clip_embedding.cpu().numpy()

    def run_pipeline(self, max_samples=None):
        """ Orchestrator linking NLP and Visual processing boundaries. """
        cleaner = DataCleaner()
        df = cleaner.validate_and_clean()
        
        if max_samples:
            print(f"Test Run flag passed: Capping process to {max_samples} videos.")
            df = df.head(max_samples)
        
        output_metadata = []
        video_tensor_db = []
        text_tensor_db = []
        
        print("\n--- Phase 1: Feature Extraction Started ---")
        
        # Using iterrows instead of DataLoader for structural layout simplicity
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            vid_id = row['video_id']
            caption = row['caption']
            vid_path = row['video_path']
            
            # --- 1. Video Processing ---
            frames = self.video_processor.extract_frames(vid_path)
            vid_embedding = self._extract_video_embeddings(frames)
            
            # --- 2. Text Processing ---
            txt_embedding = self._extract_text_embeddings([caption])[0]
            
            video_tensor_db.append(vid_embedding)
            text_tensor_db.append(txt_embedding)
            output_metadata.append({'video_id': vid_id, 'caption': caption})
            
        print("\nSaving extraction arrays to disk...")
        
        # Serialize mappings and embeddings to NPY and CSV logic
        np.save(os.path.join(Config.OUTPUT_DIR, 'video_embeddings.npy'), np.array(video_tensor_db))
        np.save(os.path.join(Config.OUTPUT_DIR, 'text_embeddings.npy'), np.array(text_tensor_db))
        
        metadata_df = pd.DataFrame(output_metadata)
        metadata_df.to_csv(os.path.join(Config.OUTPUT_DIR, 'metadata.csv'), index=False)
        
        print(f"Extraction Pipeline Completed. Output Directory: {Config.OUTPUT_DIR}")

if __name__ == "__main__":
    Config.setup()
    extractor = FeatureExtractor()
    # Runs extraction capped to 1 sample entries just for rapid testing to verify output dimensions exist.
    extractor.run_pipeline(max_samples=1)
