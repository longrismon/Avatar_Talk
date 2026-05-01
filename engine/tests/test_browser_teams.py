"""Tests for TeamsAutomation — all tests use mocked Playwright objects."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from engine.modules.browser.interface import ActionStatus
from engine.modules.browser.teams import TeamsAutomation


@pytest.fixture
def mock_page():
    page = AsyncMock()
    page.url = "https://teams.microsoft.com/v2/"
    # Playwright locator-building methods are synchronous (they return Locators, not coroutines)
    page.get_by_label = MagicMock()
    page.get_by_placeholder = MagicMock()
    page.get_by_role = MagicMock()
    page.get_by_text = MagicMock()
    page.locator = MagicMock()
    return page


@pytest.fixture
def mock_context():
    return AsyncMock()


@pytest.fixture
def teams(mock_context, mock_page):
    return TeamsAutomation(context=mock_context, page=mock_page, timeout_ms=1000)


class TestAuthCheck:
    async def test_logged_in(self, teams, mock_page):
        """Returns SUCCESS when home indicator is found."""
        locator = AsyncMock()
        locator.wait_for = AsyncMock()  # doesn't raise = found
        mock_page.get_by_label.return_value = locator
        mock_page.goto = AsyncMock()

        result = await teams.auth_check()
        assert result.status == ActionStatus.SUCCESS

    async def test_not_logged_in(self, teams, mock_page):
        """Returns NEEDS_INTERVENTION when sign-in page detected."""
        mock_page.goto = AsyncMock()
        mock_page.url = "https://login.microsoftonline.com/..."
        locator = AsyncMock()
        locator.wait_for = AsyncMock(side_effect=PlaywrightTimeoutError("timeout"))
        mock_page.get_by_label.return_value = locator

        result = await teams.auth_check()
        assert result.status == ActionStatus.NEEDS_INTERVENTION
        assert "login" in result.error.lower() or "sign in" in result.error.lower()


class TestSearchContact:
    async def test_success(self, teams, mock_page):
        """Returns SUCCESS with contact name when found."""
        search_locator = AsyncMock()
        search_locator.wait_for = AsyncMock()
        mock_page.get_by_label.return_value = search_locator

        result_locator = AsyncMock()
        result_locator.wait_for = AsyncMock()
        result_locator.first = result_locator
        # Make find_element return search_locator for search_box, result_locator for contact_result
        # We patch _find_element directly
        async def fake_find(key, **kwargs):
            if key == "search_box":
                return search_locator
            return result_locator
        teams._find_element = fake_find

        result = await teams.search_contact("Alex")
        assert result.status == ActionStatus.SUCCESS
        assert result.data == {"contact": "Alex"}

    async def test_contact_not_found(self, teams, mock_page):
        """Returns FAILED when contact element not found."""
        mock_page.screenshot = AsyncMock()

        async def fake_find(key, **kwargs):
            raise PlaywrightTimeoutError("timeout")
        teams._find_element = fake_find

        result = await teams.search_contact("Unknown Person")
        assert result.status == ActionStatus.FAILED
        assert "Unknown Person" in result.error or result.error is not None


class TestReadChatHistory:
    async def test_returns_messages(self, teams, mock_page):
        """Parses message elements into structured dict."""
        msg1 = AsyncMock()
        msg2 = AsyncMock()

        async def make_inner(text):
            m = AsyncMock()
            m.inner_text = AsyncMock(return_value=text)
            return m

        async def query_selector_side_effect(selector):
            if "author" in selector or "strong" in selector:
                return await make_inner("Alex")
            if "body" in selector or "p" in selector:
                return await make_inner("Hello!")
            if "timestamp" in selector:
                return None
            return None

        msg1.query_selector = AsyncMock(side_effect=query_selector_side_effect)
        msg2.query_selector = AsyncMock(side_effect=query_selector_side_effect)

        msg_locator = AsyncMock()
        msg_locator.wait_for = AsyncMock()
        msg_locator.all = AsyncMock(return_value=[msg1, msg2])

        teams._find_element = AsyncMock(return_value=msg_locator)

        result = await teams.read_chat_history("Alex", limit=10)
        assert result.status == ActionStatus.SUCCESS
        assert "messages" in result.data
        assert len(result.data["messages"]) == 2
        assert result.data["messages"][0]["sender"] == "Alex"


class TestGrantPermissions:
    async def test_grants_mic_and_camera(self, teams, mock_context):
        """Calls context.grant_permissions with microphone and camera."""
        result = await teams.grant_permissions(mic=True, camera=True)
        mock_context.grant_permissions.assert_called_once()
        call_args = mock_context.grant_permissions.call_args[0][0]
        assert "microphone" in call_args
        assert "camera" in call_args
        assert result.status == ActionStatus.SUCCESS

    async def test_grant_failure_returns_intervention(self, teams, mock_context):
        """Returns NEEDS_INTERVENTION on grant failure."""
        mock_context.grant_permissions = AsyncMock(side_effect=Exception("Permission denied"))
        result = await teams.grant_permissions()
        assert result.status == ActionStatus.NEEDS_INTERVENTION
