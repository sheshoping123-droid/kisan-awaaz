from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")

    # WhatsApp (Twilio)
    twilio_account_sid: str = Field(default="")
    twilio_auth_token: str = Field(default="")
    twilio_whatsapp_number: str = Field(default="")
    webhook_base_url: str = Field(default="http://localhost:8000")

    # Vision
    vision_provider: str = Field(default="qwen_api")
    dashscope_api_key: str = Field(default="")
    vision_model: str = Field(default="qwen-vl-max")
    vision_max_image_bytes: int = Field(default=10 * 1024 * 1024)
    vision_timeout: int = Field(default=30)

    # LLM
    llm_provider: str = Field(default="qwen_api")
    llm_model: str = Field(default="qwen-max")
    llm_timeout: int = Field(default=60)
    together_api_key: str = Field(default="")
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="qwen2.5:7b")

    # Speech
    speech_provider: str = Field(default="whisper_api")
    openai_api_key: str = Field(default="")
    whisper_model: str = Field(default="whisper-1")
    speech_max_audio_bytes: int = Field(default=25 * 1024 * 1024)
    speech_timeout: int = Field(default=60)

    # RAG
    embedding_model: str = Field(default="paraphrase-multilingual-MiniLM-L12-v2")
    rag_raw_dir: str = Field(default="knowledge/raw")
    rag_processed_dir: str = Field(default="knowledge/processed")
    rag_chunk_size: int = Field(default=512)
    rag_chunk_overlap: int = Field(default=50)
    faiss_index_path: str = Field(default="knowledge/processed/index.faiss")
    metadata_path: str = Field(default="knowledge/processed/metadata.json")
    rag_top_k: int = Field(default=5)
    rag_context_k: int = Field(default=3)

    # Database
    database_url: str = Field(default="sqlite:///./data/kisan_awaaz.db")

    # Admin
    admin_api_key: str = Field(default="")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
