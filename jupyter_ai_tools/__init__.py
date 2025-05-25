from .extension import jupyter_server_extension_tools

__all__ = ["jupyter_server_extension_tools"]

__version__ = "0.1.2"


def _jupyter_server_extension_points():
    return [{"module": "jupyter_ai_tools"}]


def _load_jupyter_server_extension(serverapp):
    serverapp.log.info("✅ jupyter_ai_tools extension loaded.")
