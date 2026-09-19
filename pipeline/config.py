import os

class Config:
    # Directories
    # Defaults to the repo root (two levels up from this file); override with
    # the BASE_DIR env var if the project lives somewhere else on your machine.
    BASE_DIR = os.getenv(
        "BASE_DIR",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    DATA_DIR = os.path.join(BASE_DIR, 'DATA', 'MSRVTT')
    VIDEO_DIR = os.path.join(DATA_DIR, 'raw_videos')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'pipeline', 'features')
    
    # Model Configurations
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    MAX_TEXT_TOKENS = 77
    
    # Video Processing Parameters
    TARGET_RESOLUTION = (224, 224)
    FRAMES_PER_SECOND = 1  # Down-sample to 1 frame per second to prevent OOM
    
    # Pinecone Database Parameters
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = "msrvtt-zero-shot"
    PINECONE_METRIC = "cosine"
    PINECONE_DIMENSION = 512 # CLIP output dimension
    
    # Hardware/Batch Configurations
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    @classmethod
    def setup(cls):
        """Ensure necessary output directories exist."""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
