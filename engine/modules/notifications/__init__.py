"""Avatar Agent notifications module — factory and public API."""
from .interface import NotificationClient


def create_notifier(config, device_token: str = "") -> NotificationClient:
    """Create a NotificationClient from a NotificationsConfig instance.

    Currently only Firebase FCM is supported. Returns None-safe — callers
    should check if notifications are enabled before creating.

    Args:
        config: NotificationsConfig from engine/config.py.
        device_token: FCM registration token; overrides config.firebase.device_token.

    Returns:
        A NotificationClient subclass instance.

    Raises:
        ValueError: If the provider is not recognised.
    """
    from .firebase import FirebaseNotifier

    token = device_token or getattr(config.firebase, "device_token", "")
    return FirebaseNotifier(
        credentials_path=config.firebase.credentials_path,
        device_token=token,
    )


__all__ = ["NotificationClient", "create_notifier"]
