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

    # WhatsApp (Meta Cloud API) -- used instead of Twilio when these are set.
    # META_ACCESS_TOKEN: temporary (24h) or permanent system-user token from
    #   the app's WhatsApp > API Setup page.
    # META_PHONE_NUMBER_ID: the numeric Phone Number ID shown on that same page
    #   (NOT the phone number itself).
    # META_APP_SECRET: from the app's Settings > Basic page; used to verify
    #   the X-Hub-Signature-256 header on incoming webhooks.
    # META_VERIFY_TOKEN: any string you choose yourself; you enter the same
    #   value in Meta's webhook configuration screen so Meta can verify the
    #   webhook URL belongs to you (GET hub.challenge handshake).
    meta_access_token: str = Field(default="")
    meta_phone_number_id: str = Field(default="")
    meta_app_secret: str = Field(default="")
    meta_verify_token: str = Field(default="")
    meta_api_version: str = Field(default="v21.0")

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
