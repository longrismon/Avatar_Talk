"""Bearer token verification for WebSocket connections."""
import hmac


def verify_token(provided: str, expected: str) -> bool:
    """Constant-time comparison to prevent timing attacks."""
    if not provided or not expected:
        return False
    return hmac.compare_digest(provided.encode(), expected.encode())
