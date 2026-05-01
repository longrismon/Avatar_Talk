"""
Browser app registry — maps app names to their BrowserAutomation implementations.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interface import BrowserAutomation

# Import lazily to avoid circular imports
def _get_teams_class():
    from .teams import TeamsAutomation
    return TeamsAutomation


APP_REGISTRY: dict[str, str] = {
    "teams": "teams",
    # "slack": "slack",    # Phase 2
    # "discord": "discord",  # Phase 2
}

_LOADERS = {
    "teams": _get_teams_class,
}


def get_automation_class(app_name: str):
    """Return the BrowserAutomation class for the given app name.

    Args:
        app_name: Case-insensitive app name (e.g., "teams", "TEAMS")

    Returns:
        The BrowserAutomation subclass for this app.

    Raises:
        ValueError: If the app is not in the registry.
    """
    key = app_name.lower().strip()
    if key not in _LOADERS:
        supported = list(APP_REGISTRY.keys())
        raise ValueError(f"Unsupported app: '{app_name}'. Supported apps: {supported}")
    return _LOADERS[key]()


def list_supported_apps() -> list[str]:
    """Return a list of currently supported app names."""
    return list(APP_REGISTRY.keys())
