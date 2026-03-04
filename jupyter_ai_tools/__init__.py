from .toolkits.code_execution import toolkit as exec_toolkit
from .toolkits.file_system import toolkit as fs_toolkit
from .toolkits.git import toolkit as git_toolkit
from .toolkits.notebook import toolkit as nb_toolkit

__version__ = "0.3.0"

__all__ = [
    "fs_toolkit",
    "exec_toolkit",
    "git_toolkit",
    "nb_toolkit",
]


def _jupyter_server_extension_points():
    return [{"module": "jupyter_ai_tools"}]


def _load_jupyter_server_extension(serverapp):
    serverapp.log.info("✅ jupyter_ai_tools extension loaded.")
