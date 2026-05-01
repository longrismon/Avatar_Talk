"""Tests for BrowserPool — uses mocked Playwright to avoid launching a real browser."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from engine.modules.browser.interface import ActionResult, ActionStatus


@pytest.fixture
def browser_config():
    cfg = MagicMock()
    cfg.user_data_dir = "./test_profile"
    cfg.headless = True
    cfg.default_app = "teams"
    cfg.default_step_timeout_ms = 5000
    return cfg


@pytest.fixture
def mock_playwright_context():
    """Returns a mock that simulates async_playwright().start() -> playwright"""
    mock_page = AsyncMock()
    mock_context = AsyncMock()
    mock_context.pages = [mock_page]
    mock_context.new_page = AsyncMock(return_value=mock_page)

    mock_chromium = AsyncMock()
    mock_chromium.launch_persistent_context = AsyncMock(return_value=mock_context)

    mock_pw = AsyncMock()
    mock_pw.chromium = mock_chromium

    return mock_pw, mock_context, mock_page


class TestBrowserPoolSetup:
    async def test_setup_launches_browser(self, browser_config, mock_playwright_context):
        mock_pw, mock_context, _ = mock_playwright_context

        # async_playwright() returns an object whose .start() coroutine returns the playwright
        mock_ap_instance = MagicMock()
        mock_ap_instance.start = AsyncMock(return_value=mock_pw)

        with patch("engine.modules.browser.pool.async_playwright", return_value=mock_ap_instance):
            from engine.modules.browser.pool import BrowserPool
            pool = BrowserPool(browser_config)
            await pool.setup()

            assert pool._started is True
            mock_pw.chromium.launch_persistent_context.assert_called_once()

    async def test_setup_is_idempotent(self, browser_config, mock_playwright_context):
        mock_pw, mock_context, _ = mock_playwright_context

        mock_ap_instance = MagicMock()
        mock_ap_instance.start = AsyncMock(return_value=mock_pw)

        with patch("engine.modules.browser.pool.async_playwright", return_value=mock_ap_instance):
            from engine.modules.browser.pool import BrowserPool
            pool = BrowserPool(browser_config)
            await pool.setup()
            await pool.setup()  # second call should not launch again

            mock_pw.chromium.launch_persistent_context.assert_called_once()

    async def test_setup_stores_context(self, browser_config, mock_playwright_context):
        mock_pw, mock_context, _ = mock_playwright_context

        mock_ap_instance = MagicMock()
        mock_ap_instance.start = AsyncMock(return_value=mock_pw)

        with patch("engine.modules.browser.pool.async_playwright", return_value=mock_ap_instance):
            from engine.modules.browser.pool import BrowserPool
            pool = BrowserPool(browser_config)
            await pool.setup()

            assert pool._context is mock_context
            assert pool._playwright is mock_pw


class TestGetAutomation:
    async def test_get_automation_reuses_existing_page(self, browser_config, mock_playwright_context):
        mock_pw, mock_context, mock_page = mock_playwright_context

        with patch("engine.modules.browser.pool.async_playwright"):
            from engine.modules.browser.pool import BrowserPool
            pool = BrowserPool(browser_config)
            pool._started = True
            pool._context = mock_context
            pool._playwright = mock_pw

            with patch("engine.modules.browser.pool.get_automation_class") as mock_get_cls:
                mock_automation_cls = MagicMock(return_value=AsyncMock())
                mock_get_cls.return_value = mock_automation_cls
                await pool.get_automation("teams")

            # Should not create a new page since pages already exist
            mock_context.new_page.assert_not_called()

    async def test_get_automation_opens_new_page_when_none(self, browser_config, mock_playwright_context):
        mock_pw, mock_context, mock_page = mock_playwright_context
        mock_context.pages = []  # No existing pages

        with patch("engine.modules.browser.pool.async_playwright"):
            from engine.modules.browser.pool import BrowserPool
            pool = BrowserPool(browser_config)
            pool._started = True
            pool._context = mock_context
            pool._playwright = mock_pw

            with patch("engine.modules.browser.pool.get_automation_class") as mock_get_cls:
                mock_automation_cls = MagicMock(return_value=AsyncMock())
                mock_get_cls.return_value = mock_automation_cls
                await pool.get_automation("teams")

            mock_context.new_page.assert_called_once()

    async def test_get_automation_raises_for_unknown_app(self, browser_config, mock_playwright_context):
        mock_pw, mock_context, _ = mock_playwright_context

        with patch("engine.modules.browser.pool.async_playwright"):
            from engine.modules.browser.pool import BrowserPool
            pool = BrowserPool(browser_config)
            pool._started = True
            pool._context = mock_context
            pool._playwright = mock_pw

            with pytest.raises(ValueError, match="Unsupported app"):
                await pool.get_automation("zoom")

    async def test_get_automation_lazy_starts_browser(self, browser_config, mock_playwright_context):
        mock_pw, mock_context, _ = mock_playwright_context

        mock_ap_instance = MagicMock()
        mock_ap_instance.start = AsyncMock(return_value=mock_pw)

        with patch("engine.modules.browser.pool.async_playwright", return_value=mock_ap_instance):
            with patch("engine.modules.browser.pool.get_automation_class") as mock_get_cls:
                mock_automation_cls = MagicMock(return_value=AsyncMock())
                mock_get_cls.return_value = mock_automation_cls

                from engine.modules.browser.pool import BrowserPool
                pool = BrowserPool(browser_config)
                assert pool._started is False
                await pool.get_automation("teams")
                assert pool._started is True


class TestRunPreflightCheck:
    async def test_preflight_returns_success_when_logged_in(self, browser_config, mock_playwright_context):
        mock_pw, mock_context, mock_page = mock_playwright_context

        with patch("engine.modules.browser.pool.async_playwright"):
            from engine.modules.browser.pool import BrowserPool
            pool = BrowserPool(browser_config)
            pool._started = True
            pool._context = mock_context
            pool._playwright = mock_pw

            mock_automation = AsyncMock()
            mock_automation.auth_check = AsyncMock(
                return_value=ActionResult(status=ActionStatus.SUCCESS)
            )

            async def fake_get_automation(app_name=None):
                return mock_automation

            pool.get_automation = fake_get_automation

            result = await pool.run_preflight_check()
            assert result.status == ActionStatus.SUCCESS

    async def test_preflight_returns_needs_intervention_when_not_logged_in(
        self, browser_config, mock_playwright_context
    ):
        mock_pw, mock_context, _ = mock_playwright_context

        with patch("engine.modules.browser.pool.async_playwright"):
            from engine.modules.browser.pool import BrowserPool
            pool = BrowserPool(browser_config)
            pool._started = True
            pool._context = mock_context
            pool._playwright = mock_pw

            mock_automation = AsyncMock()
            mock_automation.auth_check = AsyncMock(
                return_value=ActionResult(
                    status=ActionStatus.NEEDS_INTERVENTION,
                    error="Login required",
                )
            )

            async def fake_get_automation(app_name=None):
                return mock_automation

            pool.get_automation = fake_get_automation

            result = await pool.run_preflight_check()
            assert result.status == ActionStatus.NEEDS_INTERVENTION

    async def test_preflight_returns_failed_on_unknown_app(self, browser_config, mock_playwright_context):
        mock_pw, mock_context, _ = mock_playwright_context

        with patch("engine.modules.browser.pool.async_playwright"):
            from engine.modules.browser.pool import BrowserPool
            pool = BrowserPool(browser_config)
            pool._started = True
            pool._context = mock_context
            pool._playwright = mock_pw

            async def fake_get_automation(app_name=None):
                raise ValueError("Unsupported app: 'zoom'")

            pool.get_automation = fake_get_automation

            result = await pool.run_preflight_check("zoom")
            assert result.status == ActionStatus.FAILED
            assert "Unsupported app" in result.error

    async def test_preflight_returns_failed_on_unexpected_exception(
        self, browser_config, mock_playwright_context
    ):
        mock_pw, mock_context, _ = mock_playwright_context

        with patch("engine.modules.browser.pool.async_playwright"):
            from engine.modules.browser.pool import BrowserPool
            pool = BrowserPool(browser_config)
            pool._started = True
            pool._context = mock_context
            pool._playwright = mock_pw

            async def fake_get_automation(app_name=None):
                raise RuntimeError("Browser crashed")

            pool.get_automation = fake_get_automation

            result = await pool.run_preflight_check()
            assert result.status == ActionStatus.FAILED
            assert "Preflight check failed" in result.error


class TestExecuteStep:
    async def test_unknown_action_returns_failed(self, browser_config, mock_playwright_context):
        mock_pw, mock_context, mock_page = mock_playwright_context

        with patch("engine.modules.browser.pool.async_playwright"):
            from engine.modules.browser.pool import BrowserPool
            pool = BrowserPool(browser_config)
            pool._started = True
            pool._context = mock_context
            pool._playwright = mock_pw

            # Patch get_automation to return a mock automation without the action
            mock_automation = MagicMock(spec=[])  # empty spec — no attributes

            async def fake_get_automation(app_name=None):
                return mock_automation

            pool.get_automation = fake_get_automation

            result = await pool.execute_step({"action": "nonexistent_action", "params": {}})
            assert result.status == ActionStatus.FAILED
            assert "Unknown action" in result.error

    async def test_known_action_is_dispatched(self, browser_config, mock_playwright_context):
        mock_pw, mock_context, mock_page = mock_playwright_context

        with patch("engine.modules.browser.pool.async_playwright"):
            from engine.modules.browser.pool import BrowserPool
            pool = BrowserPool(browser_config)
            pool._started = True
            pool._context = mock_context
            pool._playwright = mock_pw

            expected_result = ActionResult(status=ActionStatus.SUCCESS)
            mock_automation = AsyncMock()
            mock_automation.end_call = AsyncMock(return_value=expected_result)

            async def fake_get_automation(app_name=None):
                return mock_automation

            pool.get_automation = fake_get_automation

            result = await pool.execute_step({"action": "end_call", "params": {}})
            assert result.status == ActionStatus.SUCCESS
            mock_automation.end_call.assert_called_once()

    async def test_open_app_extracts_app_name_from_params(self, browser_config, mock_playwright_context):
        mock_pw, mock_context, mock_page = mock_playwright_context

        with patch("engine.modules.browser.pool.async_playwright"):
            from engine.modules.browser.pool import BrowserPool
            pool = BrowserPool(browser_config)
            pool._started = True
            pool._context = mock_context
            pool._playwright = mock_pw

            captured_app = {}
            expected_result = ActionResult(status=ActionStatus.SUCCESS)
            mock_automation = AsyncMock()
            mock_automation.open_app = AsyncMock(return_value=expected_result)

            async def fake_get_automation(app_name=None):
                captured_app["name"] = app_name
                return mock_automation

            pool.get_automation = fake_get_automation

            result = await pool.execute_step(
                {"action": "open_app", "params": {"app_name": "teams"}}
            )
            assert captured_app["name"] == "teams"
            assert result.status == ActionStatus.SUCCESS

    async def test_execute_step_returns_failed_on_exception(self, browser_config, mock_playwright_context):
        mock_pw, mock_context, _ = mock_playwright_context

        with patch("engine.modules.browser.pool.async_playwright"):
            from engine.modules.browser.pool import BrowserPool
            pool = BrowserPool(browser_config)
            pool._started = True
            pool._context = mock_context
            pool._playwright = mock_pw

            async def fake_get_automation(app_name=None):
                raise RuntimeError("Unexpected crash")

            pool.get_automation = fake_get_automation

            result = await pool.execute_step({"action": "end_call", "params": {}})
            assert result.status == ActionStatus.FAILED
            assert "Step execution failed" in result.error


class TestRegistry:
    def test_registry_contains_teams(self, browser_config):
        from engine.modules.browser.pool import BrowserPool
        pool = BrowserPool(browser_config)
        assert "teams" in pool.registry

    def test_registry_returns_list(self, browser_config):
        from engine.modules.browser.pool import BrowserPool
        pool = BrowserPool(browser_config)
        assert isinstance(pool.registry, list)


class TestTeardown:
    async def test_teardown_closes_context_and_stops_playwright(
        self, browser_config, mock_playwright_context
    ):
        mock_pw, mock_context, _ = mock_playwright_context

        with patch("engine.modules.browser.pool.async_playwright"):
            from engine.modules.browser.pool import BrowserPool
            pool = BrowserPool(browser_config)
            pool._started = True
            pool._context = mock_context
            pool._playwright = mock_pw

            await pool.teardown()

            mock_context.close.assert_called_once()
            mock_pw.stop.assert_called_once()
            assert pool._started is False
            assert pool._context is None
            assert pool._playwright is None

    async def test_teardown_is_idempotent(self, browser_config, mock_playwright_context):
        mock_pw, mock_context, _ = mock_playwright_context

        with patch("engine.modules.browser.pool.async_playwright"):
            from engine.modules.browser.pool import BrowserPool
            pool = BrowserPool(browser_config)
            # Never started — teardown should be a no-op
            await pool.teardown()
            await pool.teardown()

            mock_context.close.assert_not_called()
            mock_pw.stop.assert_not_called()


class TestContextManager:
    async def test_context_manager_calls_setup_and_teardown(self, browser_config, mock_playwright_context):
        mock_pw, mock_context, _ = mock_playwright_context

        mock_ap_instance = MagicMock()
        mock_ap_instance.start = AsyncMock(return_value=mock_pw)

        with patch("engine.modules.browser.pool.async_playwright", return_value=mock_ap_instance):
            from engine.modules.browser.pool import BrowserPool
            async with BrowserPool(browser_config) as pool:
                assert pool._started is True

            # After __aexit__, teardown should have run
            assert pool._started is False

    async def test_context_manager_teardown_on_exception(self, browser_config, mock_playwright_context):
        mock_pw, mock_context, _ = mock_playwright_context

        mock_ap_instance = MagicMock()
        mock_ap_instance.start = AsyncMock(return_value=mock_pw)

        with patch("engine.modules.browser.pool.async_playwright", return_value=mock_ap_instance):
            from engine.modules.browser.pool import BrowserPool

            with pytest.raises(RuntimeError):
                async with BrowserPool(browser_config) as pool:
                    assert pool._started is True
                    raise RuntimeError("Something went wrong")

            # teardown should still have been called
            assert pool._started is False
            mock_context.close.assert_called_once()
