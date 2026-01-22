from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_env: str = "local"
    openai_api_key: str | None = None

    class Config:
        env_file = ".env"


settings = Settings()
