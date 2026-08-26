import json
import os
import pandas as pd
from pipeline.config import Config

class DataCleaner:
    def __init__(self):
        self.data_dir = Config.DATA_DIR
        self.video_dir = Config.VIDEO_DIR
        
    def load_raw_json_files(self):
        """Load all MSR-VTT json splits into a single DataFrame."""
        splits = ['msrvtt_test_1k.json', 'msrvtt_train_7k.json', 'msrvtt_train_9k.json']
        all_data = []
        for file in splits:
            path = os.path.join(self.data_dir, file)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    all_data.extend(json.load(f))
            else:
                print(f"[Warning] Partition {file} not found in {self.data_dir}")
        return pd.DataFrame(all_data)
        
    def validate_and_clean(self):
        """
        1. Drops entries where the video file mathematically does not exist to prevent extraction crashes.
        2. Applies lightweight NLP cleaning and merges dataset splits.
        """
        df = self.load_raw_json_files()
        print(f"Loaded {len(df)} initial metadata mappings.")
        
        valid_records = []
        missing_videos = set()
        
        # Determine strict bounds of reality
        for _, row in df.iterrows():
            vid = row['video_id']
            vid_path = os.path.join(self.video_dir, f"{vid}.mp4")
            
            if not os.path.exists(vid_path):
                missing_videos.add(vid)
                continue
                
            # Textual Standardization
            cleaned_caption = str(row['caption']).lower().strip()
            # We don't truncate exactly 77 WORDS here, because HuggingFace tokenizers compute 'tokens' 
            # (which can be parts of words). Truncation happens natively inside the HuggingFace Processor.
            
            valid_records.append({
                'video_id': vid,
                'caption': cleaned_caption,
                'video_path': vid_path
            })
            
        print(f"Dropped {len(missing_videos)} video IDs because physical MP4 files were missing.")
        
        clean_df = pd.DataFrame(valid_records)
        print(f"Final Count: {len(clean_df)} valid Query-to-Video mappings ready for extraction.")
        return clean_df
