"""Microsoft Teams browser automation using three-tier selector strategy."""
import asyncio
import re
import time
from typing import Optional

from playwright.async_api import (
    BrowserContext,
    Page,
    TimeoutError as PlaywrightTimeoutError,
)

from .interface import ActionResult, ActionStatus, BrowserAutomation


class TeamsAutomation(BrowserAutomation):
    """Microsoft Teams browser automation using three-tier selector strategy."""

    APP_URL = "https://teams.microsoft.com"
    SIGN_IN_INDICATORS = ["Sign in", "sign-in", "login", "microsoft.com/login"]

    # Three-tier selectors: each list is tried in order (most → least stable)
    SELECTORS = {
        "search_box": [
            {"type": "aria", "label": "Search"},
            {"type": "tid", "tid": "search-input"},
            {"type": "placeholder", "text": "Search"},
        ],
        "contact_result": [
            {"type": "tid", "tid": "search-result-person"},
            {"type": "role", "role": "option"},
            {"type": "role", "role": "listitem"},
        ],
        "call_button": [
            {"type": "aria", "label": "Call"},
            {"type": "tid", "tid": "call-button"},
            {"type": "role_name", "role": "button", "name_pattern": r"(?i)call"},
        ],
        "video_toggle": [
            {"type": "aria", "label": "Turn camera on"},
            {"type": "tid", "tid": "toggle-video"},
        ],
        "hangup_button": [
            {"type": "aria", "label": "Hang up"},
            {"type": "tid", "tid": "hangup-button"},
            {"type": "role_name", "role": "button", "name_pattern": r"(?i)hang"},
        ],
        "chat_messages": [
            {"type": "tid", "tid": "chat-pane-message"},
            {"type": "role", "role": "listitem"},
        ],
        "message_input": [
            {"type": "aria", "label": "Type a message"},
            {"type": "role", "role": "textbox"},
        ],
        "send_button": [
            {"type": "aria", "label": "Send message"},
            {"type": "tid", "tid": "send-message-button"},
        ],
        "permission_allow": [
            {"type": "text", "text": "Allow"},
        ],
        "home_indicator": [
            {"type": "aria", "label": "Activity"},
            {"type": "aria", "label": "Chat"},
        ],
    }

    def __init__(self, context: BrowserContext, page: Page, timeout_ms: int = 15000):
        self._context = context
        self._page = page
        self._timeout = timeout_ms

    async def _find_element(self, selector_key: str, timeout_ms: Optional[int] = None):
        """Try each selector in order. Returns the first locator that resolves."""
        t = timeout_ms or self._timeout
        tiers = self.SELECTORS[selector_key]
        last_error = None
        for tier in tiers:
            try:
                locator = self._build_locator(tier)
                await locator.wait_for(state="visible", timeout=t)
                return locator
            except PlaywrightTimeoutError as e:
                last_error = e
                continue
        raise PlaywrightTimeoutError(
            f"Element '{selector_key}' not found after trying {len(tiers)} selectors. Last error: {last_error}"
        )

    def _build_locator(self, tier: dict):
        """Convert a selector tier dict into a Playwright Locator."""
        t = tier["type"]
        if t == "aria":
            return self._page.get_by_label(tier["label"])
        elif t == "tid":
            return self._page.locator(f'[data-tid="{tier["tid"]}"]')
        elif t == "placeholder":
            return self._page.get_by_placeholder(tier["text"])
        elif t == "role":
            return self._page.get_by_role(tier["role"])
        elif t == "role_name":
            return self._page.get_by_role(
                tier["role"], name=re.compile(tier["name_pattern"])
            )
        elif t == "text":
            return self._page.get_by_text(tier["text"])
        else:
            raise ValueError(f"Unknown selector type: {t}")

    async def _screenshot(self) -> str:
        import tempfile
        import os
        path = os.path.join(tempfile.gettempdir(), f"avatar_agent_{int(time.time())}.png")
        await self._page.screenshot(path=path)
        return path

    async def auth_check(self) -> ActionResult:
        """Verify the user is logged in. Returns NEEDS_INTERVENTION if sign-in page is shown."""
        try:
            await self._page.goto(self.APP_URL)
            try:
                await self._find_element("home_indicator", timeout_ms=10000)
                return ActionResult(status=ActionStatus.SUCCESS)
            except PlaywrightTimeoutError:
                url = self._page.url
                if any(indicator in url for indicator in ("login", "signin")):
                    return ActionResult(
                        status=ActionStatus.NEEDS_INTERVENTION,
                        error="Teams requires login. Please sign in in the browser window and press Retry when done.",
                    )
                # Check page content for sign-in text
                try:
                    sign_in_locator = self._page.get_by_text("Sign in")
                    await sign_in_locator.wait_for(state="visible", timeout=3000)
                    return ActionResult(
                        status=ActionStatus.NEEDS_INTERVENTION,
                        error="Teams requires login. Please sign in in the browser window and press Retry when done.",
                    )
                except PlaywrightTimeoutError:
                    pass
                return ActionResult(
                    status=ActionStatus.NEEDS_INTERVENTION,
                    error="Teams requires login. Please sign in in the browser window and press Retry when done.",
                )
        except Exception as e:
            return ActionResult(status=ActionStatus.FAILED, error=str(e))

    async def open_app(self, app_name: str) -> ActionResult:
        """Navigate to Teams and check authentication state."""
        return await self.auth_check()

    async def search_contact(self, name: str) -> ActionResult:
        """Find a contact by name. Returns SUCCESS with data={"contact": name} on success."""
        try:
            search_box = await self._find_element("search_box")
            await search_box.fill(name)
            await search_box.press("Enter")
            result = await self._find_element("contact_result")
            await result.first.click()
            return ActionResult(status=ActionStatus.SUCCESS, data={"contact": name})
        except PlaywrightTimeoutError as e:
            screenshot_path = await self._screenshot()
            return ActionResult(
                status=ActionStatus.FAILED,
                error=f"Contact '{name}' not found: {e}",
                screenshot_path=screenshot_path,
            )

    async def read_chat_history(self, contact: str, limit: int = 50) -> ActionResult:
        """Scrape recent messages with a contact."""
        try:
            msg_locator = await self._find_element("chat_messages")
            elements = await msg_locator.all()
            messages = []
            for element in elements[:limit]:
                # Extract sender
                sender_el = await element.query_selector('[data-tid="message-author"]')
                if sender_el is None:
                    sender_el = await element.query_selector("strong")
                sender = await sender_el.inner_text() if sender_el else ""

                # Extract text
                body_el = await element.query_selector('[data-tid="message-body"]')
                if body_el is None:
                    body_el = await element.query_selector("p")
                text = await body_el.inner_text() if body_el else ""

                # Extract timestamp
                ts_el = await element.query_selector('[data-tid="message-timestamp"]')
                timestamp = await ts_el.inner_text() if ts_el else ""

                messages.append({"sender": sender, "text": text, "timestamp": timestamp})
            return ActionResult(status=ActionStatus.SUCCESS, data={"messages": messages})
        except Exception as e:
            return ActionResult(status=ActionStatus.FAILED, error=str(e))

    async def start_call(self, contact: str, video: bool = True) -> ActionResult:
        """Initiate a voice/video call."""
        try:
            call_button = await self._find_element("call_button")
            await call_button.click()
            return ActionResult(status=ActionStatus.SUCCESS)
        except PlaywrightTimeoutError as e:
            screenshot_path = await self._screenshot()
            return ActionResult(
                status=ActionStatus.FAILED,
                error=f"Could not find call button: {e}",
                screenshot_path=screenshot_path,
            )

    async def grant_permissions(self, mic: bool = True, camera: bool = True) -> ActionResult:
        """Accept browser permission dialogs for mic/camera."""
        try:
            permissions = []
            if mic:
                permissions.append("microphone")
            if camera:
                permissions.append("camera")
            await self._context.grant_permissions(permissions)
            return ActionResult(status=ActionStatus.SUCCESS)
        except Exception as e:
            return ActionResult(
                status=ActionStatus.NEEDS_INTERVENTION,
                error="Could not auto-grant permissions. Please allow mic/camera manually.",
            )

    async def end_call(self) -> ActionResult:
        """Hang up the current call."""
        try:
            hangup = await self._find_element("hangup_button")
            await hangup.click()
            return ActionResult(status=ActionStatus.SUCCESS)
        except Exception as e:
            return ActionResult(status=ActionStatus.FAILED, error=str(e))

    async def send_message(self, contact: str, text: str) -> ActionResult:
        """Send a text chat message."""
        try:
            msg_input = await self._find_element("message_input")
            await msg_input.fill(text)
            try:
                send_btn = await self._find_element("send_button")
                await send_btn.click()
            except PlaywrightTimeoutError:
                await msg_input.press("Enter")
            return ActionResult(status=ActionStatus.SUCCESS)
        except Exception as e:
            screenshot_path = await self._screenshot()
            return ActionResult(
                status=ActionStatus.FAILED,
                error=str(e),
                screenshot_path=screenshot_path,
            )

    async def capture_tab_audio(self) -> ActionResult:
        """Start capturing audio from the active tab via CDP."""
        return ActionResult(
            status=ActionStatus.FAILED,
            error="Tab audio capture not yet implemented (Phase 2)",
        )

    async def take_screenshot(self) -> ActionResult:
        """Capture current page state."""
        path = await self._screenshot()
        return ActionResult(status=ActionStatus.SUCCESS, data={"path": path})
