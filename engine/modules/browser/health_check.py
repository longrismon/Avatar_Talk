"""Selector health monitor — smoke-tests key browser selectors for each platform.

Run via:  python avatar.py selector-health [--app teams]

Launches a headless Chromium instance, navigates to the app URL, and checks
that each selector in the registry is findable within 5 seconds.  Returns a
HealthReport summarising pass/fail counts and per-selector latency.
"""
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    from playwright.async_api import async_playwright
except ImportError:
    async_playwright = None  # type: ignore[assignment]


# Selectors that must be visible after login for each app.
# Ordered roughly by page-load priority (navbar items first).
_SELECTORS: dict[str, list[tuple[str, str]]] = {
    "teams": [
        ("chat_nav",       '[aria-label="Chat"]'),
        ("teams_nav",      '[aria-label="Teams"]'),
        ("people_nav",     '[aria-label="People"]'),
        ("search_box",     '[aria-label*="Search"]'),
        ("new_chat",       '[aria-label*="New chat"]'),
        ("message_input",  '[data-tid="ckeditor"]'),
    ],
    "slack": [
        ("channel_list",   '[aria-label="Channel browser"]'),
        ("message_input",  '[aria-label="Message Input"]'),
    ],
    "discord": [
        ("channel_list",   'nav[aria-label="Channels"]'),
        ("message_input",  '[aria-label^="Message"]'),
    ],
}


@dataclass
class SelectorResult:
    name: str
    selector: str
    found: bool
    latency_ms: int = 0
    error: Optional[str] = None


@dataclass
class HealthReport:
    app: str
    url: str
    passed: int = 0
    failed: int = 0
    results: list[SelectorResult] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.failed == 0


class SelectorHealthChecker:
    """Playwright-based selector smoke-tester.

    Navigates to the app URL (must already be logged in via user_data_dir) and
    verifies that each registered selector becomes visible within `selector_timeout_ms`.
    """

    def __init__(self, selector_timeout_ms: int = 5000) -> None:
        self._timeout = selector_timeout_ms

    async def check(self, app: str, config) -> HealthReport:
        """Run selector health check for the given app.

        Args:
            app: App key ("teams", "slack", "discord").
            config: BrowserConfig from engine/config.py.

        Returns:
            HealthReport with per-selector pass/fail results.
        """
        url = getattr(config.app_urls, app, "")
        report = HealthReport(app=app, url=url)

        if not url:
            report.error = f"No URL configured for app '{app}'"
            return report

        selectors = _SELECTORS.get(app, [])
        if not selectors:
            report.error = f"No selectors registered for app '{app}'"
            return report

        if async_playwright is None:
            report.error = "playwright not installed — run: pip install playwright && playwright install chromium"
            return report

        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch_persistent_context(
                    user_data_dir=config.user_data_dir,
                    headless=True,
                )
                page = await browser.new_page()
                try:
                    await page.goto(url, timeout=20_000)
                    # Allow page to settle
                    await page.wait_for_timeout(3000)

                    for name, selector in selectors:
                        t0 = int(time.time() * 1000)
                        try:
                            await page.locator(selector).first.wait_for(
                                state="visible", timeout=self._timeout
                            )
                            latency = int(time.time() * 1000) - t0
                            report.results.append(
                                SelectorResult(name=name, selector=selector, found=True, latency_ms=latency)
                            )
                            report.passed += 1
                        except Exception as exc:
                            latency = int(time.time() * 1000) - t0
                            report.results.append(
                                SelectorResult(
                                    name=name,
                                    selector=selector,
                                    found=False,
                                    latency_ms=latency,
                                    error=str(exc)[:120],
                                )
                            )
                            report.failed += 1
                finally:
                    await browser.close()
        except Exception as exc:
            report.error = str(exc)

        return report
