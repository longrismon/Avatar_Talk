"""Tests for browser automation interface and registry."""
import pytest
from engine.modules.browser.interface import ActionResult, ActionStatus
from engine.modules.browser.registry import get_automation_class, list_supported_apps


def test_action_result_success():
    result = ActionResult(status=ActionStatus.SUCCESS, data={"contact": "Alex"})
    assert result.succeeded is True
    assert result.needs_intervention is False
    assert result.data == {"contact": "Alex"}


def test_action_result_failed():
    result = ActionResult(status=ActionStatus.FAILED, error="Timeout")
    assert result.succeeded is False
    assert result.needs_intervention is False
    assert result.error == "Timeout"


def test_action_result_needs_intervention():
    result = ActionResult(
        status=ActionStatus.NEEDS_INTERVENTION,
        error="Login required",
        screenshot_path="/tmp/screenshot.png",
    )
    assert result.succeeded is False
    assert result.needs_intervention is True


def test_get_automation_teams():
    cls = get_automation_class("teams")
    # Import here to avoid circular import at module level
    from engine.modules.browser.teams import TeamsAutomation
    assert cls is TeamsAutomation


def test_get_automation_case_insensitive():
    cls_lower = get_automation_class("teams")
    cls_upper = get_automation_class("TEAMS")
    cls_mixed = get_automation_class("Teams")
    assert cls_lower is cls_upper is cls_mixed


def test_get_automation_unknown_raises():
    with pytest.raises(ValueError, match="Unsupported app"):
        get_automation_class("zoom")


def test_list_supported_apps():
    apps = list_supported_apps()
    assert "teams" in apps
    assert isinstance(apps, list)
