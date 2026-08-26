import sys
from pipeline.pinecone_manager import PineconeManager
from pipeline.feature_extractor import FeatureExtractor

class ZeroShotSearchEngine:
    def __init__(self):
        print("Initializing Backend Integrations...")
        self.extractor = FeatureExtractor()
        
        self.pinecone_manager = PineconeManager()
        self.index = self.pinecone_manager.setup_index()
        
    def query(self, text_query, top_k=5):
        """
        End-to-end Search Pipeline:
        1. Encodes the raw string via HuggingFace CLIP locally into 512-dim.
        2. Casts tensor to float array.
        3. Queries Pinecone cloud index using Cosine Similarity for top_k videos.
        """
        print(f"\n[Search Request] -> '{text_query}'")
        
        # 1. Transform NLP Query to CLIP Embedding Space
        text_embedding = self.extractor._extract_text_embeddings([text_query])[0]
        query_vector = text_embedding.tolist()
        
        # 2. Query Cloud Vector Database
        print("Executing Cosine Similarity Search on Pinecone Index...")
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True # Let's us see the original caption for debug context
        )
        
        # 3. Present Results
        print("\n--- Top Matching Videos ---")
        if not results.get('matches'):
            print("No matching videos found. Ensure Phase 1 vectors were actually uploaded to Pinecone.")
            return

        for idx, match in enumerate(results['matches']):
            score = match['score']
            vid_id = match['id']
            # Ground truth metadata is attached to the vector
            original_caption = match['metadata'].get('caption', 'N/A')
            
            print(f"{idx+1}. Video: {vid_id} | Cosine Match Score: {score:.4f}")
            print(f"      (Ground Truth Mapping: '{original_caption[:50]}...')")

if __name__ == "__main__":
    engine = ZeroShotSearchEngine()
    
    # Simple CLI parsing test string
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        # Default test case mapped to MSR-VTT domains
        user_query = "a person playing a video game on a computer"
        
    engine.query(user_query)
