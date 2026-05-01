"""
Browser automation interface — abstract base class for all platform implementations.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ActionStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    NEEDS_INTERVENTION = "needs_intervention"


@dataclass
class ActionResult:
    """Result of a browser automation action."""
    status: ActionStatus
    data: Optional[dict] = None
    error: Optional[str] = None
    screenshot_path: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.status == ActionStatus.SUCCESS

    @property
    def needs_intervention(self) -> bool:
        return self.status == ActionStatus.NEEDS_INTERVENTION


class BrowserAutomation(ABC):
    """Abstract interface for browser-based communication platform automation."""

    @abstractmethod
    async def open_app(self, app_name: str) -> ActionResult:
        """Navigate to a communication app. Checks login state.
        Returns NEEDS_INTERVENTION if login is required."""

    @abstractmethod
    async def auth_check(self) -> ActionResult:
        """Verify the user is logged in. Returns NEEDS_INTERVENTION if sign-in page is shown."""

    @abstractmethod
    async def search_contact(self, name: str) -> ActionResult:
        """Find a contact by name. Returns SUCCESS with data={"contact": name} on success."""

    @abstractmethod
    async def read_chat_history(self, contact: str, limit: int = 50) -> ActionResult:
        """Scrape recent messages with a contact.
        Returns SUCCESS with data={"messages": [{"sender": str, "text": str, "timestamp": str}]}"""

    @abstractmethod
    async def start_call(self, contact: str, video: bool = True) -> ActionResult:
        """Initiate a voice/video call."""

    @abstractmethod
    async def grant_permissions(self, mic: bool = True, camera: bool = True) -> ActionResult:
        """Accept browser permission dialogs for mic/camera."""

    @abstractmethod
    async def end_call(self) -> ActionResult:
        """Hang up the current call."""

    @abstractmethod
    async def send_message(self, contact: str, text: str) -> ActionResult:
        """Send a text chat message."""

    @abstractmethod
    async def capture_tab_audio(self) -> ActionResult:
        """Start capturing audio from the active tab via CDP."""

    @abstractmethod
    async def take_screenshot(self) -> ActionResult:
        """Capture current page state. Returns SUCCESS with data={"path": str}."""
