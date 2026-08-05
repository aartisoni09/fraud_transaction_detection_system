"""
tests/test_config.py — Configuration tests.
"""

import os


def test_settings_loads():
    """Test that settings can be loaded."""
    from fraudguard.config import Settings
    settings = Settings()
    assert settings.api_port == 8000
    assert settings.log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def test_cors_origins_parsing():
    """Test that CORS origins are parsed correctly."""
    from fraudguard.config import Settings
    settings = Settings(allowed_origins="http://localhost:8501,http://example.com")
    assert len(settings.cors_origins) == 2
    assert "http://localhost:8501" in settings.cors_origins


def test_auth_disabled_by_default():
    """Test that auth is disabled when API_KEY is empty."""
    from fraudguard.config import Settings
    settings = Settings(api_key="")
    assert not settings.auth_enabled


def test_auth_enabled_with_key():
    """Test that auth is enabled when API_KEY is set."""
    from fraudguard.config import Settings
    settings = Settings(api_key="test-key-123")
    assert settings.auth_enabled
