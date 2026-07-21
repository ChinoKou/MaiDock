from src.version import DEFAULT_USER_AGENT, __version__


def test_default_user_agent_uses_plugin_version() -> None:
    assert DEFAULT_USER_AGENT == f"MaiDock/{__version__}"
