from .extension import get_git_tools, get_notebook_tools
from jupyter_server_ai_tools.models import ToolSet, Toolkit

__version__ = "0.1.2"


def _jupyter_server_extension_points():
    return [{"module": "jupyter_ai_tools"}]


def _load_jupyter_server_extension(serverapp):
    serverapp.log.info("✅ jupyter_ai_tools extension loaded.")

async def _start_jupyter_server_extension(serverapp):
    registry = serverapp.web_app.settings["toolkit_registry"]
    if registry:
        notebook_tools = ToolSet(get_notebook_tools())
        registry.register_toolkit(
            Toolkit(
                name="notebook_toolkit", tools=notebook_tools
            )
        )

        git_tools = ToolSet(get_git_tools())
        registry.register_toolkit(
            Toolkit(
                name="git_toolkit", tools=git_tools
            )
        )