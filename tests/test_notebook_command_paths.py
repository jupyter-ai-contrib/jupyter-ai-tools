from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from jupyter_ai_tools import utils


@pytest.mark.asyncio
@pytest.mark.parametrize("alias", ["relative", "absolute", "home"])
async def test_frontend_uses_jupyter_api_path(tmp_path, monkeypatch, alias):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(utils, "get_serverapp", lambda: SimpleNamespace(root_dir=str(tmp_path)))
    command = AsyncMock(return_value={"success": True})
    monkeypatch.setattr("jupyterlab_commands_toolkit.tools.execute_command", command)
    path = {
        "relative": "dir/My%20Notebook.ipynb",
        "absolute": str(tmp_path / "dir/My Notebook.ipynb"),
        "home": "~/dir/My Notebook.ipynb",
    }[alias]
    await utils.run_lab_command("test:open", {"notebookPath": path})
    command.assert_awaited_once_with("test:open", {"notebookPath": "dir/My Notebook.ipynb"})


def test_rejects_paths_outside_jupyter_root(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "get_serverapp", lambda: SimpleNamespace(root_dir=str(tmp_path)))
    with pytest.raises(ValueError, match="inside the Jupyter root"):
        utils.notebook_api_path(str(tmp_path.parent / "notebook.ipynb"))
