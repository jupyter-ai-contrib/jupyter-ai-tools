"""File IDs must retain the contents API's path through shared directory links."""

from types import SimpleNamespace

import nbformat
import pytest
from jupyter_server_fileid.manager import ArbitraryFileIdManager, LocalFileIdManager
from jupyter_ydoc import YNotebook

from jupyter_ai_tools import utils
from jupyter_ai_tools.adapters import notebook as adapters
from jupyter_ai_tools.toolkits import notebook


@pytest.fixture(params=[ArbitraryFileIdManager, LocalFileIdManager])
def notebook_room(request, tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    shared = tmp_path / "shared"
    shared.mkdir()
    (root / "Projects").symlink_to(shared, target_is_directory=True)
    api_path = "Projects/shared notebook.ipynb"
    path = shared / "shared notebook.ipynb"
    document = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("saved = 1", id="original")]
    )
    nbformat.write(document, path)
    manager = request.param(root_dir=str(root), db_path=str(tmp_path / "ids.db"))
    file_id = manager.index(api_path)
    assert file_id is not None
    server = SimpleNamespace(
        root_dir=str(root),
        web_app=SimpleNamespace(settings={"file_id_manager": manager}),
    )
    monkeypatch.setattr(utils, "get_serverapp", lambda: server)
    yield root, path, api_path, manager, file_id
    # LocalFileIdManager owns its connection's lifetime via __del__.
    if isinstance(manager, ArbitraryFileIdManager):
        manager.con.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("encoded", [False, True])
async def test_file_id_preserves_shared_directory_path(notebook_room, absolute, encoded):
    root, path, api_path, manager, expected = notebook_room
    value = str(root / api_path) if absolute else api_path
    if encoded:
        value = value.replace(" ", "%20")
    assert await utils.get_file_id(value) == expected
    assert manager.get_id(api_path) == expected


@pytest.mark.asyncio
async def test_file_id_does_not_index_unknown_notebook(notebook_room):
    root, path, api_path, manager, expected = notebook_room
    missing = "Projects/missing.ipynb"
    assert manager.get_id(missing) is None
    assert await utils.get_file_id(missing) is None
    assert manager.get_id(missing) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["read", "add", "insert", "delete", "edit", "active"])
async def test_tools_use_shared_live_notebook(notebook_room, monkeypatch, operation):
    root, path, api_path, manager, expected = notebook_room
    saved = path.read_bytes()
    ydoc = YNotebook()
    ydoc.set(
        nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("unsaved = 2", id="original")])
    )

    async def get_ydoc(file_id):
        return ydoc if file_id == expected else None

    async def no_frontend(command_id, args=None):
        return {"success": False, "error_code": "no_web_client"}

    monkeypatch.setattr(adapters, "run_lab_command", no_frontend)
    monkeypatch.setattr(adapters, "rtc_available", lambda: True)
    monkeypatch.setattr(adapters, "get_jupyter_ydoc", get_ydoc)
    monkeypatch.setattr(
        adapters.YDocAdapter,
        "_get_active_cell_id",
        staticmethod(lambda doc, *_: "original" if doc is not None else None),
    )
    if operation == "read":
        result = await adapters._read_notebook_to_run(api_path)
        assert result.cells[0].source == "unsaved = 2"
    elif operation == "add":
        await notebook.add_cell(api_path, "new = 3")
        assert ydoc.cell_number == 2
    elif operation == "insert":
        await notebook.insert_cell(api_path, "new = 3", 0)
        assert ydoc.cell_number == 2
    elif operation == "delete":
        await notebook.delete_cell(api_path, "original")
        assert ydoc.cell_number == 0
    elif operation == "edit":
        await notebook.edit_cell(api_path, "original", "edited = 3")
        assert ydoc.get()["cells"][0]["source"] == "edited = 3"
    else:
        assert await notebook.get_active_cell_id(api_path) == "original"
    assert path.read_bytes() == saved


@pytest.mark.asyncio
async def test_file_id_keeps_plain_paths_working(notebook_room):
    root, path, api_path, manager, expected = notebook_room
    plain = root / "plain.ipynb"
    plain.write_text("{}")
    expected = manager.index("plain.ipynb")
    assert await utils.get_file_id("plain.ipynb") == expected
    assert await utils.get_file_id(str(plain)) == expected
