"""End-to-end test of the fallback to the file on disk, on a real Jupyter Server.

No web client is connected to the test server, so jupyterlab-commands-toolkit
reports ``error_code: "no_web_client"`` and the RTC-free tools edit the file.
"""

import asyncio
import os
from unittest.mock import AsyncMock, patch

import nbformat
import pytest

from jupyter_ai_tools.toolkits.jupyterlab import run_all_cells, run_cell
from jupyter_ai_tools.toolkits.notebook import _run_cells, add_cell, read_notebook_json

try:
    from jupyterlab_commands_toolkit.tools import NoWebClientError  # noqa: F401
except ImportError:
    NoWebClientError = None


@pytest.fixture(autouse=True)
def no_web_client_error():
    """
    Report a missing web client, with a release of jupyterlab-commands-toolkit that does not.
    """
    if NoWebClientError is not None:
        yield
        return
    error = {
        "success": False,
        "error": "No JupyterLab web client is connected.",
        "error_code": "no_web_client",
    }
    targets = [
        "jupyter_ai_tools.toolkits.notebook.run_lab_command",
        "jupyter_ai_tools.toolkits.jupyterlab.run_lab_command",
        "jupyter_ai_tools.toolkits.jupyterlab.execute_command",
    ]
    patches = [patch(target, new_callable=AsyncMock, return_value=error) for target in targets]
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


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


def _write_notebook(root_dir, *sources):
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_code_cell(source, id=f"cell-{i}") for i, source in enumerate(sources)
    ]
    with open(os.path.join(root_dir, "notebook.ipynb"), "w") as f:
        nbformat.write(notebook, f)


def _cells(root_dir):
    with open(os.path.join(root_dir, "notebook.ipynb")) as f:
        return nbformat.read(f, as_version=nbformat.NO_CONVERT).cells


async def test_run_cell_without_web_client(jp_serverapp, jp_root_dir):
    _write_notebook(jp_root_dir, "x = 40", "x + 2")

    first = await run_cell("cell-0", file_path="notebook.ipynb")
    second = await run_cell("cell-1", file_path="notebook.ipynb")

    assert first["success"]
    assert "on disk" in first["result"]["message"]
    assert second["result"]["status"] == "ok"
    assert second["result"]["outputs"] == [
        {"output_type": "execute_result", "text": "42", "image": None}
    ]
    cells = _cells(jp_root_dir)
    assert cells[1].outputs[0].data["text/plain"] == "42"
    assert [cell.execution_count for cell in cells] == [1, 2]
    sessions = await jp_serverapp.session_manager.list_sessions()
    assert [session["path"] for session in sessions] == ["notebook.ipynb"]


async def test_run_all_cells_stops_at_error(jp_serverapp, jp_root_dir):
    _write_notebook(jp_root_dir, "print('a')", "1 / 0", "print('c')")

    result = await run_all_cells(file_path="notebook.ipynb")

    assert result["result"]["status"] == "error"
    assert [cell["cellId"] for cell in result["result"]["cells"]] == ["cell-0", "cell-1"]
    assert result["result"]["cells"][1]["errorName"] == "ZeroDivisionError"
    cells = _cells(jp_root_dir)
    assert cells[0].outputs[0].text == "a\n"
    assert cells[1].outputs[0].output_type == "error"
    assert cells[2].outputs == []


async def test_run_all_cells_without_cell_ids(jp_serverapp, jp_root_dir):
    notebook = nbformat.v4.new_notebook(nbformat_minor=4)
    notebook.cells = [nbformat.v4.new_markdown_cell("# Title"), nbformat.v4.new_code_cell("1 + 1")]
    for cell in notebook.cells:
        del cell["id"]
    with open(os.path.join(jp_root_dir, "notebook.ipynb"), "w") as f:
        nbformat.write(notebook, f)

    result = await run_all_cells(file_path="notebook.ipynb")

    assert result["result"]["status"] == "ok"
    assert _cells(jp_root_dir)[1].outputs[0].data["text/plain"] == "2"


async def test_run_cell_writes_outputs_after_timeout(jp_serverapp, jp_root_dir):
    _write_notebook(jp_root_dir, "import time; time.sleep(1); 42")

    result = await run_cell("cell-0", file_path="notebook.ipynb", timeout=0.1)

    assert result["status"] == "timed_out"
    for _ in range(100):
        if _cells(jp_root_dir)[0].outputs:
            break
        await asyncio.sleep(0.1)
    assert _cells(jp_root_dir)[0].outputs[0].data["text/plain"] == "42"


async def test_run_cell_without_file_path(jp_serverapp):
    result = await run_cell("cell-0")

    assert result["error_code"] == "no_web_client"


async def test_run_cell_without_writing_outputs(jp_serverapp, jp_root_dir):
    _write_notebook(jp_root_dir, "x = 40", "x + 2")
    path = os.path.join(jp_root_dir, "notebook.ipynb")
    with open(path) as f:
        before = f.read()

    await run_cell("cell-0", file_path="notebook.ipynb", write_outputs=False)
    result = await run_cell("cell-1", file_path="notebook.ipynb", write_outputs=False)

    assert "did not change" in result["result"]["message"]
    assert result["result"]["outputs"][0]["text"] == "42"
    with open(path) as f:
        assert f.read() == before


async def test_run_all_cells_without_writing_outputs(jp_serverapp, jp_root_dir):
    _write_notebook(jp_root_dir, "print('a')", "1 / 0", "print('c')")

    result = await run_all_cells(file_path="notebook.ipynb", write_outputs=False)

    assert result["result"]["status"] == "error"
    outputs = [cell["outputs"] for cell in result["result"]["cells"]]
    assert outputs[0] == [{"output_type": "stream", "text": "a\n"}]
    assert outputs[1][0]["output_type"] == "error"
    assert all(cell.outputs == [] for cell in _cells(jp_root_dir))


async def test_write_outputs_false_needs_file_path(jp_serverapp):
    with pytest.raises(ValueError, match="file_path"):
        await run_cell("cell-0", write_outputs=False)


async def test_input_fails(jp_serverapp, jp_root_dir):
    _write_notebook(jp_root_dir, "input()")

    result = await run_cell("cell-0", file_path="notebook.ipynb")

    assert result["result"]["status"] == "error"
    assert result["result"]["errorName"] == "StdinNotImplementedError"


async def test_runs_wait_for_each_other(jp_serverapp, jp_root_dir):
    _write_notebook(jp_root_dir, "import time; time.sleep(2); x = 1", "x + 1")

    first = await run_cell("cell-0", file_path="notebook.ipynb", timeout=0.1)
    second = await run_cell("cell-1", file_path="notebook.ipynb")

    assert first["status"] == "timed_out"
    assert second["result"]["outputs"][0]["text"] == "2"
    assert [cell.execution_count for cell in _cells(jp_root_dir)] == [1, 2]


async def test_concurrent_runs_start_one_session(jp_serverapp, jp_root_dir):
    _write_notebook(jp_root_dir, "1", "2")

    await asyncio.gather(
        run_cell("cell-0", file_path="notebook.ipynb"),
        run_cell("cell-1", file_path="notebook.ipynb"),
    )

    assert len(await jp_serverapp.session_manager.list_sessions()) == 1


async def test_kernel_restart_stops_the_run(jp_serverapp, jp_root_dir):
    _write_notebook(jp_root_dir, "import time; time.sleep(60)")
    run = asyncio.create_task(_run_cells("notebook.ipynb", "cell-0"))
    kernel_id = ""
    for _ in range(200):
        sessions = await jp_serverapp.session_manager.list_sessions()
        if sessions:
            kernel_id = sessions[0]["kernel"]["id"]
            if jp_serverapp.kernel_manager.get_kernel(kernel_id).execution_state == "busy":
                break
        await asyncio.sleep(0.1)

    await jp_serverapp.kernel_manager.restart_kernel(kernel_id)

    with pytest.raises(RuntimeError, match="restarted"):
        await asyncio.wait_for(run, 20)


async def test_empty_and_tagged_cells(jp_serverapp, jp_root_dir):
    _write_notebook(jp_root_dir, "", "print('tagged')")
    path = os.path.join(jp_root_dir, "notebook.ipynb")
    notebook = nbformat.read(path, as_version=nbformat.NO_CONVERT)
    notebook.cells[1].metadata["tags"] = ["skip-execution"]
    nbformat.write(notebook, path)

    result = await run_all_cells(file_path="notebook.ipynb")

    assert [cell["status"] for cell in result["result"]["cells"]] == ["no-op", "ok"]
    assert _cells(jp_root_dir)[1].outputs[0].text == "tagged\n"


async def test_large_outputs_are_truncated(jp_serverapp, jp_root_dir):
    _write_notebook(jp_root_dir, "print('x' * 20000)")

    result = await run_cell("cell-0", file_path="notebook.ipynb", write_outputs=False)

    assert result["result"]["outputs"][0]["text"].endswith("[Output truncated]")
    assert len(result["result"]["outputs"][0]["text"]) < 10100
