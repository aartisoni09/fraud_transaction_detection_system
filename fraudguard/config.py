"""
fraudguard/config.py — Centralized Configuration

All configuration is loaded from environment variables or a .env file.
This eliminates all hardcoded values across the application.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # ── API Configuration ──────────────────────────────────────
    api_url: str = Field(
        default="http://localhost:8000",
        description="Base URL for the FastAPI backend",
    )
    api_host: str = Field(default="0.0.0.0", description="API bind host")
    api_port: int = Field(default=8000, description="API bind port")

    # ── Security ───────────────────────────────────────────────
    api_key: str = Field(
        default="",
        description="API key for authentication. Leave empty to disable auth.",
    )
    allowed_origins: str = Field(
        default="http://localhost:8501,http://localhost:3000",
        description="Comma-separated list of allowed CORS origins",
    )

    # ── Database ───────────────────────────────────────────────
    database_url: str = Field(
        default="fraudguard.db",
        description="SQLite database file path",
    )

    # ── Model & Data ───────────────────────────────────────────
    model_bundle_path: str = Field(
        default="model_bundle.pkl",
        description="Path to the model bundle pickle file",
    )
    metrics_path: str = Field(
        default="metrics.json",
        description="Path to the model metrics JSON file",
    )
    dataset_path: str = Field(
        default="fraud_dataset_1500.csv",
        description="Path to the fraud dataset CSV file",
    )

    # ── Risk Thresholds ────────────────────────────────────────
    critical_threshold: float = Field(
        default=0.80,
        description="Probability threshold for CRITICAL risk level",
    )
    medium_threshold: float = Field(
        default=0.30,
        description="Probability threshold for MEDIUM risk level",
    )
    # Note: HIGH threshold comes from the trained model (best_threshold)
    # and is loaded dynamically. These are the static boundaries.

    # ── Logging ────────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
        "protected_namespaces": ("settings_",),
    }

    @property
    def cors_origins(self) -> list[str]:
        """Parse comma-separated ALLOWED_ORIGINS into a list."""
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    @property
    def auth_enabled(self) -> bool:
        """Check if API key authentication is enabled."""
        return bool(self.api_key)


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings (singleton)."""
    return Settings()
