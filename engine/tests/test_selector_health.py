"""Tests for Phase 6 — SelectorHealthChecker."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engine.modules.browser.health_check import (
    HealthReport,
    SelectorHealthChecker,
    SelectorResult,
)


class TestSelectorResult:
    def test_found_true(self):
        r = SelectorResult(name="chat_nav", selector='[aria-label="Chat"]', found=True, latency_ms=120)
        assert r.found is True
        assert r.error is None

    def test_found_false_with_error(self):
        r = SelectorResult(name="x", selector="y", found=False, error="timeout")
        assert r.found is False
        assert r.error == "timeout"


class TestHealthReport:
    def test_ok_when_no_failures(self):
        report = HealthReport(app="teams", url="https://teams.microsoft.com", passed=3, failed=0)
        assert report.ok is True

    def test_not_ok_when_failures(self):
        report = HealthReport(app="teams", url="https://teams.microsoft.com", passed=2, failed=1)
        assert report.ok is False

    def test_not_ok_when_error(self):
        report = HealthReport(app="teams", url="https://teams.microsoft.com", error="Nav failed")
        assert report.ok is False


class TestSelectorHealthChecker:
    def _make_browser_config(self):
        cfg = MagicMock()
        cfg.user_data_dir = "/tmp/profile"
        cfg.app_urls.teams = "https://teams.microsoft.com"
        cfg.app_urls.slack = "https://app.slack.com"
        return cfg

    async def test_unknown_app_returns_error(self):
        checker = SelectorHealthChecker()
        cfg = self._make_browser_config()
        cfg.app_urls.unknown = ""
        report = await checker.check("unknown_app", cfg)
        assert report.error is not None

    async def test_no_url_returns_error(self):
        checker = SelectorHealthChecker()
        cfg = self._make_browser_config()
        # Make app_urls not have the attribute
        del cfg.app_urls.teams
        cfg.app_urls.__class__ = object
        type(cfg.app_urls).teams = property(lambda self: "")
        report = await checker.check("teams", cfg)
        assert report.error is not None

    async def test_playwright_not_installed_returns_error(self):
        checker = SelectorHealthChecker()
        cfg = self._make_browser_config()
        import engine.modules.browser.health_check as hc
        original = hc.async_playwright
        hc.async_playwright = None
        try:
            report = await checker.check("teams", cfg)
        finally:
            hc.async_playwright = original
        assert report.error is not None

    async def test_all_selectors_found(self):
        checker = SelectorHealthChecker()
        cfg = self._make_browser_config()

        mock_locator = MagicMock()
        mock_locator.first.wait_for = AsyncMock()

        mock_page = AsyncMock()
        mock_page.locator = MagicMock(return_value=mock_locator)
        mock_page.goto = AsyncMock()
        mock_page.wait_for_timeout = AsyncMock()

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.close = AsyncMock()

        mock_pw = AsyncMock()
        mock_pw.__aenter__ = AsyncMock(return_value=mock_pw)
        mock_pw.__aexit__ = AsyncMock(return_value=False)
        mock_pw.chromium.launch_persistent_context = AsyncMock(return_value=mock_context)

        with patch("engine.modules.browser.health_check.async_playwright", return_value=mock_pw):
            report = await checker.check("teams", cfg)

        assert report.error is None
        assert report.failed == 0
        assert report.passed == len(report.results)
        assert report.ok is True

    async def test_one_selector_fails(self):
        checker = SelectorHealthChecker()
        cfg = self._make_browser_config()

        call_count = [0]

        async def _wait_for(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Timeout waiting for element")

        mock_locator = MagicMock()
        mock_locator.first.wait_for = _wait_for

        mock_page = AsyncMock()
        mock_page.locator = MagicMock(return_value=mock_locator)
        mock_page.goto = AsyncMock()
        mock_page.wait_for_timeout = AsyncMock()

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.close = AsyncMock()

        mock_pw = AsyncMock()
        mock_pw.__aenter__ = AsyncMock(return_value=mock_pw)
        mock_pw.__aexit__ = AsyncMock(return_value=False)
        mock_pw.chromium.launch_persistent_context = AsyncMock(return_value=mock_context)

        with patch("engine.modules.browser.health_check.async_playwright", return_value=mock_pw):
            report = await checker.check("teams", cfg)

        assert report.failed == 1
        assert report.passed >= 1
        assert report.ok is False
        failed = [r for r in report.results if not r.found]
        assert len(failed) == 1
        assert "Timeout" in failed[0].error

    async def test_browser_launch_error(self):
        checker = SelectorHealthChecker()
        cfg = self._make_browser_config()

        mock_pw = AsyncMock()
        mock_pw.__aenter__ = AsyncMock(return_value=mock_pw)
        mock_pw.__aexit__ = AsyncMock(return_value=False)
        mock_pw.chromium.launch_persistent_context = AsyncMock(
            side_effect=Exception("Browser crashed")
        )

        with patch("engine.modules.browser.health_check.async_playwright", return_value=mock_pw):
            report = await checker.check("teams", cfg)

        assert report.error is not None
        assert "Browser crashed" in report.error

    async def test_no_selectors_for_app_returns_error(self):
        checker = SelectorHealthChecker()
        cfg = self._make_browser_config()
        # 'discord' has selectors; test with a patched empty map
        from engine.modules.browser import health_check as hc
        original = hc._SELECTORS.copy()
        hc._SELECTORS["teams"] = []
        try:
            report = await checker.check("teams", cfg)
        finally:
            hc._SELECTORS["teams"] = original["teams"]
        assert report.error is not None
