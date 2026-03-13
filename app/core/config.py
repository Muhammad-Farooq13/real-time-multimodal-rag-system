from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "multimodal-rag"
    app_env: str = "dev"
    openai_api_key: str = ""
    vector_backend: str = "in_memory"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_top_k: int = 20
    pinecone_api_key: str = ""
    pinecone_index_name: str = "multimodal-rag-index"
    pinecone_namespace: str = "default"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    faiss_index_path: str = "data/processed/faiss.index"
    cache_enabled: bool = True
    cache_backend: str = "redis"
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 300
    semantic_cache_threshold: float = 0.92
    semantic_cache_max_entries: int = 5000
    semantic_cache_distributed: bool = True
    redis_vector_index_name: str = "idx:rag:semantic"
    redis_vector_prefix: str = "rag:semantic:"
    redis_vector_search_k: int = 5

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
