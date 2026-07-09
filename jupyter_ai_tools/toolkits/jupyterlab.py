import asyncio
import os
from typing import Optional

from jupyterlab_commands_toolkit.tools import execute_command

from ..utils import get_serverapp


async def _run_with_timeout(coro, timeout: Optional[float], started_msg: str) -> dict:
    """Run a coroutine with an optional timeout.

    If timeout is exceeded, the task continues in the background.
    """
    task = asyncio.create_task(coro)
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
    except asyncio.TimeoutError:
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


async def run_all_cells(file_path: Optional[str] = None, timeout: Optional[float] = None) -> dict:
    """Runs all cells in a Jupyter notebook.

    Does NOT return cell outputs — call `read_notebook_cells` to inspect results.

    Valid argument combinations:
        - `file_path`: Run all cells in the specified notebook.
        - No arguments: Run all cells in the currently active notebook.

    Args:
        file_path: Path to the notebook file. If provided, the notebook is
                   opened/focused before running. If None, runs in the
                   currently active notebook.
        timeout: Max seconds to wait (default and max: 10s). A timeout does
                 NOT mean execution failed; the kernel continues running.

    Returns:
        dict with `success` (bool) and optional `error` or `result` fields.
    """
    if file_path:
        result = await open_file(file_path)
        if not result.get("success"):
            return result

    return await _run_with_timeout(
        execute_command("notebook:run-all-cells"), timeout, "Run all cells started"
    )


async def run_cell(
    cell_id: str,
    file_path: Optional[str] = None,
    username: Optional[str] = None,
    timeout: Optional[float] = None,
) -> dict:
    """Runs a specific cell in a notebook by selecting it and executing it.

    Does NOT return cell outputs — call `read_notebook_cells` to inspect results.

    Valid argument combinations:
        - `file_path` + `cell_id`: Run a specific cell in the given notebook.
        - `username` + `cell_id`: Run a specific cell in the user's active notebook.
        - `cell_id` only: Run a specific cell in the currently active notebook.

    Args:
        cell_id: The UUID of the cell to run, or a numeric index as string.
        file_path: Path to the notebook file. If provided, the notebook is
                   opened/focused before running and used to resolve the cell.
                   If None, the user's active notebook is used.
        username: Optional username to get the active cell for that specific user.
                  Also used when file_path is provided, to pick whose active
                  cell the cursor navigation starts from.
        timeout: Max seconds to wait (default and max: 10s). A timeout does
                 NOT mean execution failed; the kernel continues running.

    Returns:
        dict with `success` (bool) and optional `error` or `result` fields.
    """
    from .notebook import select_cell

    if file_path:
        result = await open_file(file_path)
        if not result.get("success"):
            return result

    await select_cell(cell_id, username, file_path=file_path)

    return await _run_with_timeout(
        execute_command("notebook:run-cell"), timeout, "Cell execution started"
    )


toolkit = [open_file, run_cell, run_all_cells]
