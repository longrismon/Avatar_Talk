"""Firebase Cloud Messaging (FCM) push notification client."""
import asyncio

from .interface import NotificationClient
from engine.logging_config import get_logger

log = get_logger("notifications.firebase")


class FirebaseNotifier(NotificationClient):
    """Sends FCM push notifications via the firebase-admin SDK.

    The SDK call is blocking; it runs in a thread executor so the event loop
    is not blocked during network I/O.

    Args:
        credentials_path: Path to the Firebase service-account JSON file.
        device_token: FCM registration token of the principal's mobile device.
    """

    def __init__(self, credentials_path: str, device_token: str) -> None:
        self._credentials_path = credentials_path
        self._device_token = device_token
        self._app = None

    def _ensure_app(self) -> None:
        """Initialise the Firebase Admin SDK app (idempotent)."""
        if self._app is not None:
            return
        import firebase_admin
        from firebase_admin import credentials

        cred = credentials.Certificate(self._credentials_path)
        self._app = firebase_admin.initialize_app(cred, name=f"avatar_{id(self)}")
        log.info("firebase_app_initialized", credentials=self._credentials_path)

    async def send(self, title: str, body: str, data: dict | None = None) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._send_sync, title, body, data or {})

    def _send_sync(self, title: str, body: str, data: dict) -> None:
        from firebase_admin import messaging

        self._ensure_app()

        message = messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            data={k: str(v) for k, v in data.items()},
            token=self._device_token,
        )
        message_id = messaging.send(message, app=self._app)
        log.info("firebase_notification_sent", message_id=message_id, title=title)

    async def aclose(self) -> None:
        if self._app is not None:
            try:
                import firebase_admin
                firebase_admin.delete_app(self._app)
            except Exception:
                pass
            self._app = None
        log.info("firebase_notifier_closed")
