from pydantic import BaseSettings


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Audio Transcription API"

    # Model settings
    WHISPER_MODEL: str = "small"  # Options: tiny, base, small, medium, large
    DEFAULT_LANGUAGE: str = "auto"  # Use "auto" for automatic detection

    # File upload settings
    MAX_UPLOAD_SIZE: int = 25 * 1024 * 1024  # 25 MB

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()