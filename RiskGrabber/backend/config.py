from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = Field(
        default="postgresql+psycopg2://risk_grabber:risk_grabber@localhost:5433/risk_grabber",
        env="RISK_GRABBER_DB_URL",
        description="SQLAlchemy sync database URL (PostgreSQL)",
    )
    async_database_url: str = Field(
        default="",
        env="RISK_GRABBER_ASYNC_DB_URL",
        description="Async URL (postgresql+asyncpg). If empty, derived from database_url.",
    )

    def get_async_database_url(self) -> str:
        if self.async_database_url:
            return self.async_database_url
        url = self.database_url
        if "postgresql+psycopg2" in url:
            return url.replace("postgresql+psycopg2", "postgresql+asyncpg", 1)
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url

    banki_base_url: str = (
        "https://www.banki.ru/services/responses/list/?type=all&rate[]=1&rate[]=2"
    )
    scrape_pages: int = Field(
        default=3, description="How many pages of reviews to fetch per cycle"
    )
    proxies_dir: str = Field(
        default="",
        env="PROXIES_DIR",
        description="Папка с прокси для парсера (пусто = RiskGrabber/proxies). Файлы .txt, строка = один прокси.",
    )

    dataset_cache_dir: str = Field(
        default="",
        env="DATASET_CACHE_DIR",
        description="Папка кэша датасетов (пусто = data/datasets в корне проекта).",
    )

    debug: bool = Field(
        default=False,
        env="DEBUG",
        description="Enable debug logging (INFO level if False, DEBUG level if True)",
    )

    hf_model_name: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B",
        env="HF_EMBED_MODEL",
        description=(
            "HuggingFace model name for embeddings (используется, если не задан HF_EMBED_MODEL_PATH)."
        ),
    )
    embed_model_path: str | None = Field(
        default=None,
        env="HF_EMBED_MODEL_PATH",
        description=(
            "Локальный путь к папке с моделью эмбеддингов (например, data/models/qwen3-embedding-0.6b). "
            "Если задан, загрузка идёт с диска, иначе — по HF_EMBED_MODEL с HuggingFace."
        ),
    )

    qdrant_host: str = Field(
        default="localhost",
        env="QDRANT_HOST",
        description="Qdrant host (use 'qdrant' when running via docker-compose)",
    )
    qdrant_port: int = Field(
        default=6333,
        env="QDRANT_PORT",
        description="Qdrant HTTP port",
    )
    qdrant_collection: str = Field(
        default="bank_reviews",
        env="QDRANT_COLLECTION",
        description="Qdrant collection name for review embeddings",
    )
    qdrant_risks_collection: str = Field(
        default="bank_risks",
        env="QDRANT_RISKS_COLLECTION",
        description="Qdrant collection name for risk title embeddings",
    )

    llm_base_url: str = Field(
        default="http://localhost:8001/v1",
        env="LLM_BASE_URL",
        description="Base URL for LLM API (vllm: http://localhost:8001/v, Ollama: http://localhost:11434/v1)",
    )
    llm_api_key: str | None = Field(
        default='empty',
        env="LLM_API_KEY",
        description="API key for LLM",
    )
    llm_model: str = Field(
        default="Qwen3.5-35B-A3B",
        env="LLM_MODEL",
        description="Chat completion model name",
    )
    llm_top_k_risks: int = Field(
        default=5,
        env="LLM_TOP_K_RISKS",
        description="Количество ближайших существующих рисков для передачи в LLM промпт",
    )
    llm_critic_max_iter: int = Field(
        default=3,
        env="LLM_CRITIC_MAX_ITER",
        description="Максимум возвратов в генератор от критика (не более N заходов)",
    )
    generator_max_risk_factors: int = Field(
        default=3,
        ge=1,
        le=20,
        env="GENERATOR_MAX_RISK_FACTORS",
        description="Макс. число факторов риска на выходе генератора",
    )
    generator_max_implications: int = Field(
        default=3,
        ge=1,
        le=20,
        env="GENERATOR_MAX_IMPLICATIONS",
        description="Макс. число последствий на выходе генератора",
    )
    max_risk_factors: int = Field(
        default=5,
        ge=1,
        le=20,
        env="MAX_RISK_FACTORS",
        description="Макс. число факторов после консолидации и при записи в БД",
    )
    max_implications: int = Field(
        default=5,
        ge=1,
        le=20,
        env="MAX_IMPLICATIONS",
        description="Макс. число последствий после консолидации и при записи в БД",
    )
    max_words_per_factor: int = Field(
        default=10,
        ge=3,
        le=30,
        env="MAX_WORDS_PER_FACTOR",
        description="Макс. слов в одной формулировке фактора или последствия",
    )

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
    }


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()

