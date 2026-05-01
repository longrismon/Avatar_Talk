"""
Tests for engine/config.py — Pydantic v2 configuration system.
"""

import os
import textwrap

import pytest
import yaml
from pydantic import ValidationError

# Reset the module-level singleton between tests so each test starts clean.
import engine.config as config_module


@pytest.fixture(autouse=True)
def reset_config_singleton():
    """Ensure the cached _config singleton is cleared before every test."""
    config_module._config = None
    yield
    config_module._config = None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _write_yaml(tmp_path, data: dict) -> str:
    """Serialize *data* to a YAML file inside tmp_path and return its path."""
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(data), encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# Test 1 — load the real config.yaml
# ---------------------------------------------------------------------------

def test_load_valid_config():
    """Loading the actual engine/config.yaml must succeed and produce correct values."""
    # Locate config.yaml relative to this test file: engine/config.yaml
    engine_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(engine_dir, "config.yaml")

    settings = config_module.load_config(config_path)

    assert settings.server.port == 9600
    # Spot-check a few other fields to ensure the whole tree loaded.
    assert settings.server.host == "0.0.0.0"
    assert settings.llm.primary == "anthropic"
    assert settings.audio.stt.primary == "faster-whisper"
    assert settings.lipsync.enabled is False
    assert settings.logging.level == "INFO"


# ---------------------------------------------------------------------------
# Test 2 — ${ENV_VAR} interpolation works when the variable is set
# ---------------------------------------------------------------------------

def test_env_var_interpolation(tmp_path, monkeypatch):
    """${TEST_API_KEY} in YAML must be replaced with the env var value."""
    monkeypatch.setenv("TEST_API_KEY", "test-value")

    data = {
        "audio": {
            "tts": {
                "elevenlabs": {
                    "api_key": "${TEST_API_KEY}",
                }
            }
        }
    }
    path = _write_yaml(tmp_path, data)

    settings = config_module.load_config(path)
    assert settings.audio.tts.elevenlabs.api_key == "test-value"


# ---------------------------------------------------------------------------
# Test 3 — ${MISSING_VAR} becomes empty string when the variable is absent
# ---------------------------------------------------------------------------

def test_env_var_missing(tmp_path, monkeypatch):
    """${MISSING_VAR} must resolve to an empty string when the var is unset."""
    # Guarantee the variable is not set in this process.
    monkeypatch.delenv("MISSING_VAR", raising=False)

    data = {
        "audio": {
            "tts": {
                "elevenlabs": {
                    "api_key": "${MISSING_VAR}",
                }
            }
        }
    }
    path = _write_yaml(tmp_path, data)

    settings = config_module.load_config(path)
    assert settings.audio.tts.elevenlabs.api_key == ""


# ---------------------------------------------------------------------------
# Test 4 — port below minimum (1024) raises ValidationError
# ---------------------------------------------------------------------------

def test_invalid_port_low(tmp_path):
    """server.port: 80 is below the 1024 minimum and must raise ValidationError."""
    data = {"server": {"port": 80}}
    path = _write_yaml(tmp_path, data)

    with pytest.raises(ValidationError):
        config_module.load_config(path)


# ---------------------------------------------------------------------------
# Test 5 — port above maximum (65535) raises ValidationError
# ---------------------------------------------------------------------------

def test_invalid_port_high(tmp_path):
    """server.port: 99999 exceeds 65535 and must raise ValidationError."""
    data = {"server": {"port": 99999}}
    path = _write_yaml(tmp_path, data)

    with pytest.raises(ValidationError):
        config_module.load_config(path)


# ---------------------------------------------------------------------------
# Test 6 — review.timeout_seconds below minimum (2.0) raises ValidationError
# ---------------------------------------------------------------------------

def test_invalid_review_timeout(tmp_path):
    """review.timeout_seconds: 1.0 is below the 2.0 minimum and must raise ValidationError."""
    data = {"review": {"timeout_seconds": 1.0}}
    path = _write_yaml(tmp_path, data)

    with pytest.raises(ValidationError):
        config_module.load_config(path)
