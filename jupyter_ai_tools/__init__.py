from .adapters import notebook as notebook_adapters
from .toolkits.jupyterlab import toolkit as jlab_toolkit
from .toolkits.notebook import toolkit as nb_toolkit

__version__ = "0.7.0"

__all__ = [
    "nb_toolkit",
    "jlab_toolkit",
    "notebook_adapters",
]


def _jupyter_server_extension_points():
    return [{"module": "jupyter_ai_tools"}]


def _load_jupyter_server_extension(serverapp):
    serverapp.log.info("✅ jupyter_ai_tools extension loaded.")
