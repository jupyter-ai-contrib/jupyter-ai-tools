import asyncio
import os
from typing import Optional

from jupyterlab_commands_toolkit.tools import execute_command

from ..utils import get_serverapp


async def _run_with_timeout(coro, timeout: Optional[float], started_msg: str) -> dict:
    """Run a coroutine with an optional timeout. If timeout is exceeded, the task continues in the background."""
    task = asyncio.create_task(coro)
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
    except asyncio.TimeoutError:
        return {"status": "timed_out", "message": f"{started_msg}, timed out after {timeout}s of waiting"}


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


async def run_all_cells(timeout: Optional[float] = None) -> dict:
    """Runs all cells in the currently active Jupyter notebook.

    This does NOT return cell outputs. To inspect outputs after running,
    call `read_notebook_cells`.

    Args:
        timeout: Max seconds to wait. None (default) waits until complete.
                 If exceeded, returns early while cells continue running
                 in the kernel — a timeout does NOT mean execution failed.

    Returns:
        dict: {"result": true} on success, {"result": false} on error,
              or {"status": "timed_out", ...} if timeout exceeded.
    """
    return await _run_with_timeout(
        execute_command("notebook:run-all-cells"), timeout, "Run all cells started"
    )


async def run_cell(cell_id: str, username: Optional[str] = None, timeout: Optional[float] = None) -> dict:
    """Runs a specific cell in the active notebook by selecting it and executing it.

    This does NOT return cell outputs. To inspect outputs after running,
    call `read_notebook_cells` with the cell's ID.

    Args:
        cell_id: The UUID of the cell to run, or a numeric index as string
        username: Optional username to get the active cell for that specific user
        timeout: Max seconds to wait. None (default) waits until complete.
                 If exceeded, returns early while the cell continues running
                 in the kernel — a timeout does NOT mean execution failed.

    Returns:
        dict: {"result": true} on success, {"result": false} on error,
              or {"status": "timed_out", ...} if timeout exceeded.
    """
    from .notebook import select_cell

    await select_cell(cell_id, username)

    return await _run_with_timeout(
        execute_command("notebook:run-cell"), timeout, "Cell execution started"
    )


toolkit = [open_file, run_cell, run_all_cells]
