from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    index_path: str = "index.bin"
    log_level: str = "INFO"
    port: int = 8000
    cors_origins: str = "*"
