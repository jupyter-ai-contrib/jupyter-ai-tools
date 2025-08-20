from .toolkits.code_execution import toolkit as codeexec_toolkit
from .toolkits.file_system import toolkit as fs_toolkit
from .toolkits.git import toolkit as git_toolkit
from .toolkits.notebook import toolkit as notebook_toolkit

__version__ = "0.2.1"

__all__ = [
    "fs_toolkit",
    "codeexec_toolkit", 
    "git_toolkit",
    "notebook_toolkit",
    "__version__",
    "_jupyter_server_extension_points",
    "_load_jupyter_server_extension"
]


def _jupyter_server_extension_points():
    return [{"module": "jupyter_ai_tools"}]


def _load_jupyter_server_extension(serverapp):
    serverapp.log.info("✅ jupyter_ai_tools extension loaded.")
