# Application settings loaded from .env (kept outside the project tree).
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


def _resolve_env_files() -> tuple[Path, ...]:
    candidates: list[Path] = []
    user_home_env = Path.home() / ".monobazar-pricing-agent" / ".env"
    if user_home_env.exists():
        candidates.append(user_home_env)
    project_env = Path(__file__).resolve().parent.parent / ".env"
    if project_env.exists():
        candidates.append(project_env)
    custom = os.environ.get("MONOBAZAR_ENV_FILE")
    if custom:
        custom_path = Path(custom)
        if custom_path.exists():
            candidates.append(custom_path)
    return tuple(candidates) if candidates else (project_env,)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_resolve_env_files(),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    anthropic_api_key: str = ""
    serpapi_key: str = ""

    llm_model: str = "claude-opus-4-6"
    llm_vision_max_tokens: int = 1024
    llm_clarify_max_tokens: int = 512

    embedding_model_name: str = "intfloat/multilingual-e5-large"
    embedding_batch_size: int = 64

    faiss_index_dir: Path = Path("data/faiss_indexes")
    faiss_top_k: int = 20

    data_dir: Path = Path("data")
    raw_listings_csv: Path = Path("data/raw/listings.csv")
    processed_listings_csv: Path = Path("data/processed/listings.csv")

    models_dir: Path = Path("models")
    lgbm_num_leaves: int = 63
    lgbm_n_estimators: int = 500
    lgbm_learning_rate: float = 0.05

    olx_num_results: int = 5
    serpapi_engine: str = "google"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False


settings = Settings()
