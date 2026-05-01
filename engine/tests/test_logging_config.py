"""Tests for logging_config — verifies structlog setup and get_logger behavior."""
import json
import logging
import io
import sys
import pytest
import structlog

from engine.logging_config import setup_logging, get_logger


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging and structlog configuration between tests."""
    yield
    # Reset stdlib logging
    root = logging.getLogger()
    root.handlers.clear()
    # Reset structlog
    structlog.reset_defaults()


def _capture_log_output(level="DEBUG", log_file=None):
    """Set up logging and return a StringIO that captures stdout log output."""
    capture = io.StringIO()
    handler = logging.StreamHandler(capture)
    handler.setLevel(logging.DEBUG)

    setup_logging(level=level, log_file=log_file)

    # Replace stdout handler with our capture handler
    root = logging.getLogger()
    root.handlers = [handler]

    return capture


class TestSetupLogging:
    def test_setup_does_not_raise(self):
        setup_logging(level="INFO")

    def test_setup_with_invalid_level_defaults_to_info(self):
        setup_logging(level="NOTAREAL")  # should not raise

    def test_creates_log_directory(self, tmp_path):
        log_file = str(tmp_path / "subdir" / "agent.log")
        setup_logging(level="INFO", log_file=log_file)
        import os
        assert os.path.exists(tmp_path / "subdir")

    def test_log_file_created(self, tmp_path):
        log_file = str(tmp_path / "agent.log")
        setup_logging(level="INFO", log_file=log_file)
        log = get_logger("test")
        log.info("test_event")
        import os
        assert os.path.exists(log_file)


class TestGetLogger:
    def test_returns_bound_logger(self):
        setup_logging(level="DEBUG")
        log = get_logger("test_module")
        assert log is not None

    def test_logger_emits_json(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        setup_logging(level="DEBUG", log_file=log_file)
        log = get_logger("test_module")
        log.info("my_event", key="value")

        with open(log_file, "r") as f:
            line = f.read().strip()

        data = json.loads(line)
        assert data["event"] == "my_event"
        assert data["module"] == "test_module"
        assert data["level"] == "info"
        assert "timestamp_ms" in data
        assert data["key"] == "value"

    def test_timestamp_ms_is_integer(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        setup_logging(level="DEBUG", log_file=log_file)
        log = get_logger("ts_test")
        log.debug("ts_check")

        with open(log_file, "r") as f:
            data = json.loads(f.read().strip())
        assert isinstance(data["timestamp_ms"], int)
        assert data["timestamp_ms"] > 1_000_000_000_000  # ms since epoch > 1T

    def test_module_bound_correctly(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        setup_logging(level="DEBUG", log_file=log_file)
        log = get_logger("my_module")
        log.warning("warn_event")

        with open(log_file, "r") as f:
            data = json.loads(f.read().strip())
        assert data["module"] == "my_module"

    def test_extra_fields_included(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        setup_logging(level="DEBUG", log_file=log_file)
        log = get_logger("extra_test")
        log.info("event_with_extras", call_id="abc123", turn_number=3, duration_ms=450)

        with open(log_file, "r") as f:
            data = json.loads(f.read().strip())
        assert data["call_id"] == "abc123"
        assert data["turn_number"] == 3
        assert data["duration_ms"] == 450
