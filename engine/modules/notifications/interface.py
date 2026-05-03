"""Notification interface — abstract base class for push notification providers."""
from abc import ABC, abstractmethod


class NotificationClient(ABC):
    """Abstract interface for push notification providers."""

    @abstractmethod
    async def send(self, title: str, body: str, data: dict | None = None) -> None:
        """Send a push notification.

        Args:
            title: Notification title (shown in banner).
            body: Notification body text.
            data: Optional key-value payload delivered to the mobile app.
        """

    @abstractmethod
    async def aclose(self) -> None:
        """Release any open connections or resources."""
