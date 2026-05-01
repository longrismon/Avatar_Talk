"""
BrowserPool — manages a single persistent Playwright Chromium instance.
"""
import asyncio
from typing import Optional

from playwright.async_api import (
    async_playwright,
    BrowserContext,
    Playwright,
)

from .interface import ActionResult, ActionStatus, BrowserAutomation
from .registry import get_automation_class


class BrowserPool:
    """
    Manages a single persistent Chromium browser context with session data.

    The pool is lazy — the browser is not launched until the first call to
    get_automation() or run_preflight_check().
    """

    # Chromium launch flags for virtual device support and reduced detection
    CHROMIUM_ARGS = [
        "--use-fake-device-for-media-stream",        # Accept virtual audio/cam devices
        "--autoplay-policy=no-user-gesture-required", # Allow audio autoplay
        "--disable-blink-features=AutomationControlled",  # Reduce automation detection
        "--disable-infobars",
        "--no-sandbox",  # Required in some Linux environments
    ]

    def __init__(self, config):
        """
        Args:
            config: BrowserConfig instance from engine/config.py
        """
        self._config = config
        self._playwright: Optional[Playwright] = None
        self._context: Optional[BrowserContext] = None
        self._lock = asyncio.Lock()
        self._started = False

    async def setup(self) -> None:
        """Launch Playwright and open the persistent browser context.

        Idempotent — calling setup() when already started is a no-op.
        """
        async with self._lock:
            if self._started:
                return
            self._playwright = await async_playwright().start()
            self._context = await self._playwright.chromium.launch_persistent_context(
                user_data_dir=self._config.user_data_dir,
                headless=self._config.headless,
                args=self.CHROMIUM_ARGS,
                timeout=30000,
            )
            self._started = True

    async def get_automation(self, app_name: Optional[str] = None) -> BrowserAutomation:
        """Return an automation instance for the given app.

        Lazily starts the browser if not already running.

        Args:
            app_name: e.g. "teams". Defaults to config.default_app.

        Returns:
            A BrowserAutomation instance bound to the current page.

        Raises:
            ValueError: If app_name is not in the registry.
        """
        if not self._started:
            await self.setup()

        target_app = app_name or self._config.default_app

        # Reuse existing page or open a new one
        pages = self._context.pages
        if pages:
            page = pages[0]
        else:
            page = await self._context.new_page()

        automation_class = get_automation_class(target_app)
        return automation_class(
            context=self._context,
            page=page,
            timeout_ms=self._config.default_step_timeout_ms,
        )

    async def run_preflight_check(self, app_name: Optional[str] = None) -> ActionResult:
        """Run the auth check for the target app before starting a call.

        Returns:
            ActionResult with SUCCESS if logged in, NEEDS_INTERVENTION if login required.
        """
        try:
            automation = await self.get_automation(app_name)
            return await automation.auth_check()
        except ValueError as e:
            return ActionResult(status=ActionStatus.FAILED, error=str(e))
        except Exception as e:
            return ActionResult(
                status=ActionStatus.FAILED,
                error=f"Preflight check failed: {e}",
            )

    async def execute_step(self, step: dict) -> ActionResult:
        """Execute a single action plan step.

        Args:
            step: dict with keys "action" and "params", e.g.:
                  {"action": "open_app", "params": {"app_name": "teams"}}

        Returns:
            ActionResult from the corresponding automation method.
        """
        try:
            action = step["action"]
            params = dict(step.get("params", {}))  # copy — don't mutate caller's dict
            app_name = params.pop("app_name", None) if action == "open_app" else None
            automation = await self.get_automation(app_name)

            method = getattr(automation, action, None)
            if method is None:
                return ActionResult(
                    status=ActionStatus.FAILED,
                    error=f"Unknown action: '{action}'",
                )
            return await method(**params)
        except Exception as e:
            return ActionResult(
                status=ActionStatus.FAILED,
                error=f"Step execution failed: {e}",
            )

    @property
    def registry(self) -> list[str]:
        """Return the list of supported app names."""
        from .registry import list_supported_apps
        return list_supported_apps()

    async def teardown(self) -> None:
        """Close the browser context and stop Playwright cleanly."""
        async with self._lock:
            if not self._started:
                return
            try:
                if self._context:
                    await self._context.close()
            finally:
                if self._playwright:
                    await self._playwright.stop()
                self._started = False
                self._context = None
                self._playwright = None

    async def __aenter__(self):
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.teardown()
