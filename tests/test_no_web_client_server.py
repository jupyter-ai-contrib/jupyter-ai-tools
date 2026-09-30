"""End-to-end test of the fallback to the file on disk, on a real Jupyter Server.

No web client is connected to the test server, so jupyterlab-commands-toolkit
reports ``error_code: "no_web_client"`` and the RTC-free tools edit the file.
"""

import os

import nbformat
import pytest

from jupyter_ai_tools.toolkits.notebook import add_cell, read_notebook_json

try:
    from jupyterlab_commands_toolkit.tools import NoWebClientError  # noqa: F401
except ImportError:
    NoWebClientError = None

pytestmark = pytest.mark.skipif(
    NoWebClientError is None,
    reason="jupyterlab-commands-toolkit does not report a missing web client",
)


@pytest.fixture
def jp_server_config(jp_server_config):
    return {
        "ServerApp": {
            "jpserver_extensions": {
                "jupyter_ai_tools": True,
                "jupyterlab_commands_toolkit": True,
            }
        }
    }


async def test_add_cell_without_web_client(jp_serverapp, jp_root_dir):
    path = os.path.join(jp_root_dir, "notebook.ipynb")
    with open(path, "w") as f:
        nbformat.write(nbformat.v4.new_notebook(), f)

    result = await add_cell("notebook.ipynb", "x = 1")

    assert result["success"]
    assert "on disk" in result["result"]["message"]
    notebook = await read_notebook_json("notebook.ipynb")
    assert ["".join(cell["source"]) for cell in notebook["cells"]] == ["x = 1"]
