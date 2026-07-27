from .toolkits.notebook import toolkit as nb_toolkit

__version__ = "0.6.1"

__all__ = [
    "nb_toolkit",
]


def _jupyter_server_extension_points():
    return [{"module": "jupyter_ai_tools"}]


def _load_jupyter_server_extension(serverapp):
    serverapp.log.info("✅ jupyter_ai_tools extension loaded.")
