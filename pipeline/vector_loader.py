import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pipeline.pinecone_manager import PineconeManager
from pipeline.config import Config

class VectorLoader:
    def __init__(self):
        self.db_manager = PineconeManager()
        self.index = self.db_manager.setup_index()
        self.batch_size = 100 # Safe size to prevent network overload
        
    def _generator_chunks(self, iterable, batch_size):
        """Yield successive n-sized chunks from iterable."""
        for i in range(0, len(iterable), batch_size):
            yield iterable[i:i + batch_size]
            
    def load_video_vectors(self):
        """
        Reads the .npy embeddings saved from Phase 1, associates them with their Video IDs,
        and upserts the dense 512-dim logic to Pinecone.
        """
        vec_path = os.path.join(Config.OUTPUT_DIR, 'video_embeddings.npy')
        meta_path = os.path.join(Config.OUTPUT_DIR, 'metadata.csv')
        
        if not os.path.exists(vec_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing Extraction files in {Config.OUTPUT_DIR}. Did Phase 1 execute?")
            
        print("Loading Extracted Phase 1 Embeddings from local disk...")
        video_embeddings = np.load(vec_path)
        metadata_df = pd.DataFrame(pd.read_csv(meta_path))
        
        if len(video_embeddings) != len(metadata_df):
             raise ValueError("Dimension mismatch between Embeddings tensor and mapping IDs.")
             
        # Format for Pinecone Upsert: List[Tuple(id, vector_list, metadata_dict)]
        print(f"Preparing to upsert {len(video_embeddings)} vectors to Pinecone index '{Config.PINECONE_INDEX_NAME}'")
        
        vectors_for_upsert = []
        for i in range(len(video_embeddings)):
            row = metadata_df.iloc[i]
            vid_id = row['video_id']
            caption = row['caption']
            
            # Pinecone expects native float arrays, not numpy
            vector_float_list = video_embeddings[i].tolist()
            
            # Storing the ground truth caption alongside the video helps us debug the search later
            metadata_dict = {"caption": caption}
            
            vectors_for_upsert.append( (str(vid_id), vector_float_list, metadata_dict) )
            
        print("Uploading to cloud DB...")
        for batch in tqdm(self._generator_chunks(vectors_for_upsert, self.batch_size), total=len(vectors_for_upsert)//self.batch_size):
            self.index.upsert(vectors=batch)
            
        print("\nAll Vectors successfully indexed and uploaded to Pinecone!")
        print("Final Backend Stats:", self.index.describe_index_stats())

if __name__ == "__main__":
    loader = VectorLoader()
    loader.load_video_vectors()
