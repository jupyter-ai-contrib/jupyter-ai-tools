import asyncio
import os
from typing import Optional, Set

from jupyterlab_commands_toolkit.tools import execute_command

from ..utils import get_serverapp, no_web_client, rtc_available, run_lab_command

# The frontend commands get the same limit from jupyterlab-commands-toolkit.
_MAX_TIMEOUT = 10.0

_background_tasks: Set[asyncio.Task] = set()


def _finish_background_task(task: asyncio.Task) -> None:
    _background_tasks.discard(task)
    if not task.cancelled() and task.exception() is not None:
        get_serverapp().log.error("A tool call failed after its timeout", exc_info=task.exception())


async def _run_with_timeout(coro, timeout: Optional[float], started_msg: str) -> dict:
    """Run a coroutine with an optional timeout.

    If timeout is exceeded, the task continues in the background.
    """
    task = asyncio.create_task(coro)
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
    except asyncio.TimeoutError:
        # Keep a reference to the task, and log its error, as nothing awaits it now.
        _background_tasks.add(task)
        task.add_done_callback(_finish_background_task)
        return {
            "status": "timed_out",
            "message": f"{started_msg}, timed out after {timeout}s of waiting",
        }


async def open_file(file_path: str):
    """Opens a file in JupyterLab main area.

    Args:
        file_path: Path to the file relative to jupyter root

    Returns:
        dict: Response from JupyterLab with success, result, and optional error fields
    """
    if os.path.isabs(file_path):
        try:
            root_dir = get_serverapp().root_dir
        except Exception:
            root_dir = os.getcwd()

        try:
            file_path = os.path.relpath(file_path, root_dir)
        except ValueError:
            pass

    return await execute_command("docmanager:open", {"path": file_path})


async def _run_cells_on_server(
    file_path: str,
    cell_id: Optional[str],
    write_outputs: bool,
    timeout: Optional[float],
    started_msg: str,
) -> dict:
    """
    Run cells in the kernel of the notebook on the server, with the limit of the frontend.
    """
    from .notebook import _run_cells

    timeout = _MAX_TIMEOUT if timeout is None else min(timeout, _MAX_TIMEOUT)
    return await _run_with_timeout(
        _run_cells(file_path, cell_id, write_outputs), timeout, started_msg
    )


async def run_all_cells(
    file_path: Optional[str] = None, timeout: Optional[float] = None, write_outputs: bool = True
) -> dict:
    """Runs all cells in a Jupyter notebook.

    The result can include the cell outputs. If not, call `read_notebook_cells`
    to inspect results.

    Valid argument combinations:
        - `file_path`: Run all cells in the specified notebook.
        - No arguments: Run all cells in the currently active notebook.

    Args:
        file_path: Path to the notebook file. If provided, JupyterLab opens or
                   focuses the notebook before running. If None, runs in the
                   currently active notebook. Without RTC and without a
                   JupyterLab web client, the cells run in the kernel of the
                   notebook on the server, and the outputs go to the notebook
                   file on disk.
        timeout: Max seconds to wait (default and max: 10s). A timeout does
                 NOT mean execution failed; the kernel continues running.
        write_outputs: If False, the cells run in the kernel of the notebook on
                       the server, and the result has the outputs. The notebook
                       does not change and JupyterLab does not show the run,
                       but the kernel state changes. A run that times out loses
                       its outputs. Needs `file_path`.

    Returns:
        dict with `success` (bool) and optional `error` or `result` fields.
    """
    started_msg = "Run all cells started"
    if not write_outputs:
        if not file_path:
            raise ValueError("file_path is required when write_outputs is False")
        return await _run_cells_on_server(file_path, None, False, timeout, started_msg)

    if file_path:
        result = await open_file(file_path)
        if no_web_client(result) and not rtc_available():
            return await _run_cells_on_server(file_path, None, True, timeout, started_msg)
        if not result.get("success"):
            return result

    return await _run_with_timeout(execute_command("notebook:run-all-cells"), timeout, started_msg)


async def run_cell(
    cell_id: str,
    file_path: Optional[str] = None,
    username: Optional[str] = None,
    timeout: Optional[float] = None,
    write_outputs: bool = True,
) -> dict:
    """Runs a specific cell in a notebook by selecting it and executing it.

    The result can include the cell outputs. If not, call `read_notebook_cells`
    to inspect results.

    Valid argument combinations:
        - `file_path` + `cell_id`: Run a specific cell in the given notebook.
        - `username` + `cell_id`: Run a specific cell in the user's active notebook.
        - `cell_id` only: Run a specific cell in the currently active notebook.

    Args:
        cell_id: The nbformat id of the cell to run.
        file_path: Path to the notebook file. If provided, JupyterLab opens or
                   focuses the notebook before running, and the cell is found
                   in it. If None, the user's active notebook is used. Without
                   RTC and without a JupyterLab web client, the cell runs in
                   the kernel of the notebook on the server, and the outputs go
                   to the notebook file on disk.
        username: Optional username to get the active cell for that specific user.
                  Also used when file_path is provided, to pick whose active
                  cell the cursor navigation starts from.
        timeout: Max seconds to wait (default and max: 10s). A timeout does
                 NOT mean execution failed; the kernel continues running.
        write_outputs: If False, the cell runs in the kernel of the notebook on
                       the server, and the result has the outputs. The notebook
                       does not change and JupyterLab does not show the run,
                       but the kernel state changes. A run that times out loses
                       its outputs. Needs `file_path`.

    Returns:
        dict with `success` (bool) and optional `error` or `result` fields.
    """
    from .notebook import select_cell

    started_msg = "Cell execution started"
    if not write_outputs:
        if not file_path:
            raise ValueError("file_path is required when write_outputs is False")
        return await _run_cells_on_server(file_path, cell_id, False, timeout, started_msg)

    if not rtc_available():
        # RTC-free: jupyterlab-ai-commands run-cell targets the cell by id and
        # opens the notebook itself, so no awareness-based select is needed.
        result = await _run_with_timeout(
            run_lab_command(
                "jupyterlab-ai-commands:run-cell",
                {"notebookPath": file_path, "cellId": cell_id},
            ),
            timeout,
            started_msg,
        )
        if not (file_path and no_web_client(result)):
            return result
        return await _run_cells_on_server(file_path, cell_id, True, timeout, started_msg)

    if file_path:
        result = await open_file(file_path)
        if not result.get("success"):
            return result

    await select_cell(cell_id, username, file_path=file_path)

    return await _run_with_timeout(execute_command("notebook:run-cell"), timeout, started_msg)


toolkit = [open_file, run_cell, run_all_cells]
