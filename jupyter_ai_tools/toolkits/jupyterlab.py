import os
from typing import Optional

from jupyterlab_commands_toolkit.tools import execute_command

from ..utils import get_serverapp


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


async def run_all_cells(wait: bool = True) -> dict:
    """Runs all cells in the currently active Jupyter notebook.

    Args:
        wait: If True, waits for execution to complete. If False, starts asynchronously.

    Returns:
        dict: The result from execution, or a success message if wait=False.
    """
    import asyncio

    if wait:
        return await execute_command("notebook:run-all-cells")

    asyncio.create_task(execute_command("notebook:run-all-cells"))
    return {"success": True, "message": "Run all cells started"}


async def run_cell(cell_id: str, username: Optional[str] = None, wait: bool = True) -> dict:
    """Runs a specific cell in the active notebook by selecting it and executing it.

    Args:
        cell_id: The UUID of the cell to run, or a numeric index as string
        username: Optional username to get the active cell for that specific user
        wait: If True, waits for cell execution to complete and returns the result.
              If False, starts execution asynchronously and returns immediately.

    Returns:
        dict: The result from execution, or a success message if wait=False.
    """
    import asyncio
    from .notebook import select_cell

    await select_cell(cell_id, username)

    if wait:
        return await execute_command("notebook:run-cell")

    asyncio.create_task(execute_command("notebook:run-cell"))
    return {"success": True, "message": "Cell execution started"}


toolkit = [open_file, run_cell, run_all_cells]
