from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    whisper_model: str = "large"
    whisper_language: str = "kk"

settings = Settings()
