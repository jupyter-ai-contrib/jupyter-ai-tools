import os
from typing import Optional

from jupyterlab_commands_toolkit.tools import execute_command

from ..adapters.notebook import get_adapter
from ..utils import get_serverapp

__all__ = ["open_file", "run_cell", "run_all_cells"]


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
    adapter = await get_adapter(file_path)
    return await adapter.run_all_cells(file_path, timeout, write_outputs)


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
    adapter = await get_adapter(file_path)
    return await adapter.run_cell(cell_id, file_path, username, timeout, write_outputs)


toolkit = [open_file, run_cell, run_all_cells]
