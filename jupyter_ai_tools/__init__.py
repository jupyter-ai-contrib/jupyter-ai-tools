from .extension import get_git_tools, get_notebook_tools
from jupyter_server_ai_tools.models import Tool, Toolkit

__version__ = "0.1.2"


def _jupyter_server_extension_points():
    return [{"module": "jupyter_ai_tools"}]


def _load_jupyter_server_extension(serverapp):
    serverapp.log.info("✅ jupyter_ai_tools extension loaded.")

async def _start_jupyter_server_extension(serverapp):
    registry = serverapp.extension_manager.extensions.get(
        "jupyter_server_ai_tools"
    )
    if registry:
        registry.register_toolkit(
            Toolkit(
                name="notebook_toolkit", tools=get_notebook_tools()
            )
        )

        registry.register_toolkit(
            Toolkit(
                name="git_toolkit", tools=get_git_tools()
            )
        )