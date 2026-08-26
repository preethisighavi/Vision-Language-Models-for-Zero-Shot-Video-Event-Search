from pinecone import Pinecone, ServerlessSpec
from pipeline.config import Config

class PineconeManager:
    def __init__(self):
        print("Initializing Pinecone Client Connection...")
        if not Config.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is not set in Config.")
            
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        self.index_name = Config.PINECONE_INDEX_NAME
        
    def setup_index(self):
        """
        Idempotent function that guarantees the 'msrvtt-zero-shot' index exists
        and mathematically aligns strictly to OpenAI CLIP dimensions.
        """
        existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            print(f"Index '{self.index_name}' not found. Creating a new Serverless index...")
            print(f"Dimensions: {Config.PINECONE_DIMENSION} | Metric: {Config.PINECONE_METRIC}")
            
            self.pc.create_index(
                name=self.index_name,
                dimension=Config.PINECONE_DIMENSION,
                metric=Config.PINECONE_METRIC,
                # Using standard AWS serverless configuration for generic demo
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"Index '{self.index_name}' successfully provisioned.")
        else:
            print(f"Index '{self.index_name}' already exists. Connecting...")
            
        return self.pc.Index(self.index_name)

if __name__ == "__main__":
    db_manager = PineconeManager()
    index = db_manager.setup_index()
    print("Database connection fully initialized:", index.describe_index_stats())
