from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Path("./data")
    checkpoint_dir: Path = Path("./checkpoints")
    database_url: str = ""
    redis_url: str = ""
    index_backend: str = "numpy"
    pinecone_api_key: str = ""
    pinecone_index: str = ""
    cors_origins: str = "http://localhost:5173"
    admin_key: str = "change-me"
    demo_upload_key: str = ""
    site_password: str = ""
    max_upload_mb: int = 200
    max_upload_seconds: int = 600
    preload_models: bool = False
    mock_latency: bool = True
    models_yaml: Path = BACKEND_DIR / "models.yaml"
    results_csv: Path = BACKEND_DIR.parent.parent / "results" / "model_results.csv"
    benchmarks_meta: Path = BACKEND_DIR / "benchmarks_meta.yaml"
    search_rate_per_min: int = 60
    upload_rate_per_hour: int = 10

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{self.data_dir / 'spartan.db'}"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def media_dir(self) -> Path:
        return self.data_dir / "media"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def seed_dir(self) -> Path:
        return self.data_dir / "seed"


@lru_cache
def get_settings() -> Settings:
    return Settings()
