from .tools import get_git_tools, get_notebook_tools
from jupyter_server_ai_tools.models import ToolSet, Toolkit

__version__ = "0.1.2"


def _jupyter_server_extension_points():
    return [{"module": "jupyter_ai_tools"}]

def _load_jupyter_server_extension(serverapp):
    serverapp.log.info("✅ jupyter_ai_tools extension loaded.")