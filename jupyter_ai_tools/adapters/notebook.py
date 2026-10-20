import asyncio
import difflib
import json
import os
import re
import weakref
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import unquote
from uuid import uuid4

import nbformat
from jupyter_ydoc import YNotebook
from nbclient import NotebookClient
from pycrdt import Assoc, Text

from ..utils import (
    command_result,
    get_file_id,
    get_global_awareness,
    get_jupyter_ydoc,
    get_serverapp,
    no_web_client,
    normalize_filepath,
    rtc_available,
    run_lab_command,
)
from ._notebook_helpers import (
    NEW_CELL,
    determine_insert_index,
    get_cell_index_from_id_nbformat,
    process_notebook_output,
    read_notebook_file,
    write_notebook_file,
)

# One lock for each notebook, so that its runs and the creation of its session do not
# overlap. The runs that hold or wait for a lock keep it alive.
_notebook_locks: "weakref.WeakValueDictionary[str, asyncio.Lock]" = weakref.WeakValueDictionary()

_MAX_OUTPUT_TEXT = 10000
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

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


async def _read_notebook_to_run(file_path: str):
    """
    Read the current content of a notebook, preferring live state over disk.

    With RTC: reads from the YDoc (the YDoc may be ahead of the file on disk
    when the room has not autosaved yet).
    Without RTC: reads from the JupyterLab frontend model (which may have
    unsaved edits), falling back to disk when no browser is connected.
    """
    content = None
    if rtc_available():
        ydoc = await get_jupyter_ydoc(await get_file_id(file_path))
        if ydoc is not None:
            content = ydoc.get()
    if content is None:
        content = await FrontendAdapter().read(file_path)
    return nbformat.reads(json.dumps(content), as_version=nbformat.NO_CONVERT)


class BaseAdapter(ABC):
    """Strategy for reading and mutating a notebook."""

    @abstractmethod
    async def read(self, file_path: str) -> dict: ...

    @abstractmethod
    async def add_cell(
        self,
        file_path: str,
        content: Optional[str],
        cell_id: Optional[str],
        add_above: bool,
        cell_type: str,
        animate: bool = False,
    ): ...

    @abstractmethod
    async def insert_cell(
        self,
        file_path: str,
        content: Optional[str],
        insert_index: Optional[int],
        cell_type: str,
        animate: bool = False,
    ): ...

    @abstractmethod
    async def delete_cell(self, file_path: str, cell_id: str) -> dict: ...

    @abstractmethod
    async def edit_cell(
        self,
        file_path: str,
        cell_id: str,
        content: Optional[str],
        cell_type: Optional[str],
        animate: bool = False,
    ) -> dict: ...

    @abstractmethod
    async def get_active_cell_id(
        self, file_path: str, username: Optional[str]
    ) -> Optional[str]: ...

    @abstractmethod
    async def get_active_notebook(self, username: Optional[str]) -> Optional[str]: ...

    @abstractmethod
    async def select_cell(
        self, cell_id: str, username: Optional[str], file_path: Optional[str]
    ) -> dict: ...

    @abstractmethod
    async def run_cell(
        self,
        cell_id: str,
        file_path: Optional[str],
        username: Optional[str],
        timeout: Optional[float],
        write_outputs: bool,
    ) -> dict: ...

    @abstractmethod
    async def run_all_cells(
        self,
        file_path: Optional[str],
        timeout: Optional[float],
        write_outputs: bool,
    ) -> dict: ...


class DiskAdapter(BaseAdapter):
    """Reads/writes the .ipynb file on disk. Used when no live model is available."""

    @staticmethod
    def _file_result(message: str, **fields) -> dict:
        """Build the result of a write tool that edited the notebook file on disk.

        The message tells the agent that the edit went to the file, not to a
        notebook open in JupyterLab.
        """
        return {"success": True, "result": {"message": message, **fields}}

    def _add_cell_to_file(
        self,
        file_path: str,
        content: Optional[str],
        cell_id: Optional[str],
        add_above: bool,
        cell_type: str,
    ) -> str:
        """Add a cell to the notebook file on disk and return the id of the new cell."""
        notebook = read_notebook_file(file_path)
        cell_index = get_cell_index_from_id_nbformat(notebook, cell_id) if cell_id else None
        insert_index = determine_insert_index(len(notebook.cells), cell_index, add_above)
        return self._insert_cell_in_file(file_path, content, insert_index, cell_type, notebook)

    @staticmethod
    def _insert_cell_in_file(
        file_path: str,
        content: Optional[str],
        insert_index: Optional[int],
        cell_type: str,
        notebook=None,
    ) -> str:
        """Insert a cell in the notebook file on disk and return the id of the new cell."""
        if notebook is None:
            notebook = read_notebook_file(file_path)
        if insert_index is None:
            insert_index = len(notebook.cells)
        cell = NEW_CELL[cell_type](source=content or "")
        notebook.cells.insert(insert_index, cell)
        write_notebook_file(file_path, notebook)
        return cell.id

    @staticmethod
    def _delete_cell_from_file(file_path: str, cell_id: str) -> Optional[int]:
        """Delete a cell from the notebook file on disk and return its former index."""
        notebook = read_notebook_file(file_path)
        cell_index = get_cell_index_from_id_nbformat(notebook, cell_id)
        if cell_index is not None:
            notebook.cells.pop(cell_index)
            write_notebook_file(file_path, notebook)
        return cell_index

    @staticmethod
    def _edit_cell_in_file(
        file_path: str,
        cell_id: str,
        content: Optional[str],
        cell_type: Optional[str],
    ) -> None:
        """Edit the content and/or type of a cell in the notebook file on disk."""
        notebook = read_notebook_file(file_path)
        cell_index = get_cell_index_from_id_nbformat(notebook, cell_id)
        if cell_index is None:
            raise ValueError(f"Cell with {cell_id=} not found in notebook at {file_path=}")

        old_cell = notebook.cells[cell_index]
        source = content if content is not None else old_cell.source

        if cell_type is not None and cell_type != old_cell.cell_type:
            new_cell = NEW_CELL[cell_type](source=source)
            if getattr(old_cell, "id", None):
                new_cell.id = old_cell.id
            new_cell.metadata.update(old_cell.get("metadata", {}))
            notebook.cells[cell_index] = new_cell
        elif content is not None:
            old_cell.source = content
        else:
            return
        write_notebook_file(file_path, notebook)

    async def read(self, file_path: str) -> dict:
        import json

        normalized = normalize_filepath(file_path)
        with open(normalized, "r", encoding="utf-8") as f:
            return json.load(f)

    async def add_cell(self, file_path, content, cell_id, add_above, cell_type, animate=False):
        new_id = self._add_cell_to_file(
            normalize_filepath(file_path), content, cell_id, add_above, cell_type
        )
        return self._file_result(
            f"{cell_type} cell added to the notebook file on disk, "
            "as the notebook is not open in JupyterLab",
            cellId=new_id,
        )

    async def insert_cell(self, file_path, content, insert_index, cell_type, animate=False):
        new_id = self._insert_cell_in_file(
            normalize_filepath(file_path), content, insert_index, cell_type
        )
        return self._file_result(
            f"{cell_type} cell inserted in the notebook file on disk, "
            "as the notebook is not open in JupyterLab",
            cellId=new_id,
        )

    async def delete_cell(self, file_path, cell_id):
        index = self._delete_cell_from_file(normalize_filepath(file_path), cell_id)
        if index is None:
            raise ValueError(f"Could not find cell index for {cell_id=}")
        return self._file_result(
            "Cell deleted from the notebook file on disk, "
            "as the notebook is not open in JupyterLab",
            cellId=cell_id,
        )

    async def edit_cell(self, file_path, cell_id, content, cell_type, animate=False):
        self._edit_cell_in_file(normalize_filepath(file_path), cell_id, content, cell_type)
        return self._file_result(
            "Cell edited in the notebook file on disk, "
            "as the notebook is not open in JupyterLab",
            cellId=cell_id,
        )

    async def get_active_cell_id(self, file_path, username):
        raise RuntimeError("No active cell: notebook is not open in JupyterLab")

    async def get_active_notebook(self, username):
        raise RuntimeError("No active notebook: notebook is not open in JupyterLab")

    async def select_cell(self, cell_id, username, file_path):
        raise RuntimeError("Cannot select cell: notebook is not open in JupyterLab")

    @staticmethod
    def _notebook_lock(path: str) -> asyncio.Lock:
        lock = _notebook_locks.get(path)
        if lock is None:
            lock = _notebook_locks[path] = asyncio.Lock()
        return lock

    @staticmethod
    def _session_path(file_path: str) -> str:
        """
        Return the path of a notebook relative to the server root, as the sessions store it.

        The symlinks stay, as in the session paths of JupyterLab.
        """
        root_dir = get_serverapp().root_dir
        for root in (root_dir, os.path.realpath(root_dir)):
            path = os.path.relpath(os.path.join(root, unquote(file_path)), root)
            if path != os.pardir and not path.startswith(os.pardir + os.sep):
                return Path(path).as_posix()
        raise ValueError(f"{file_path=} is not in the root directory of the server: {root_dir}")

    @staticmethod
    async def _get_notebook_kernel(path: str, notebook):
        """
        Return the kernel manager of the notebook session, and start a session if there is none.

        JupyterLab connects to the same session when it opens the notebook.
        """
        from tornado import web

        serverapp = get_serverapp()
        try:
            session = await serverapp.session_manager.get_session(path=path)
        # jupyter_server < 1.13.2 raises a KeyError when the kernel of the session was culled.
        except (web.HTTPError, KeyError):
            session = await serverapp.session_manager.create_session(
                path=path,
                name=os.path.basename(path),
                type="notebook",
                kernel_name=notebook.metadata.get("kernelspec", {}).get("name") or None,
            )
        return serverapp.kernel_manager.get_kernel(session["kernel"]["id"])

    @staticmethod
    def _write_cell_outputs(file_path: str, cell_index: int, cell) -> None:
        """
        Write the outputs and execution count of a cell to the notebook file on disk.

        The file is read again, as other tools can edit it while the cell runs.
        A cell without an id (nbformat < 4.5) is found by its index.
        """
        notebook = read_notebook_file(file_path)
        index = get_cell_index_from_id_nbformat(notebook, cell.id) if "id" in cell else cell_index
        if (
            index is None
            or index >= len(notebook.cells)
            or notebook.cells[index].cell_type != "code"
        ):
            return
        notebook.cells[index].outputs = cell.outputs
        notebook.cells[index].execution_count = cell.execution_count
        write_notebook_file(file_path, notebook)

    @staticmethod
    def _summarize_outputs(outputs, budget: int) -> Tuple[List[Dict[str, Any]], int]:
        """
        Process outputs for a tool result, and return them with the remaining text budget.

        The result has no image data and no ANSI codes, and its text stops at the budget.
        """
        summaries = []
        for output in outputs:
            summary = dict(process_notebook_output(output))
            if summary.get("image"):
                summary["image"] = {"mime_type": summary["image"]["mime_type"]}
            if summary.get("text"):
                text = _ANSI_ESCAPE.sub("", summary["text"])
                if len(text) > budget:
                    text = text[:budget] + "\n[Output truncated]"
                budget = max(budget - len(text), 0)
                summary["text"] = text
            summaries.append(summary)
        return summaries, budget

    @staticmethod
    async def _until_kernel_restart(km, coro):
        """
        Await a coroutine that waits for kernel replies, and stop it if the kernel restarts,
        as the replies then never come. nbclient already stops when the kernel dies.
        """
        # A restart keeps the provisioner, but replaces its process.
        process = getattr(getattr(km, "provisioner", None), "process", None)
        task = asyncio.ensure_future(coro)
        try:
            while not task.done():
                await asyncio.wait({task}, timeout=1)
                restarted = (
                    getattr(getattr(km, "provisioner", None), "process", None) is not process
                )
                if not task.done() and restarted:
                    raise RuntimeError("The kernel restarted during the run")
            return task.result()
        finally:
            task.cancel()

    @staticmethod
    async def _execute_cells(
        client: NotebookClient, cell_indices: List[int], file_path: str, write_outputs: bool
    ) -> List[Dict[str, Any]]:
        """
        Execute cells with an nbclient client, and stop at the first cell with an error.
        """
        results: List[Dict[str, Any]] = []
        budget = _MAX_OUTPUT_TEXT
        for cell_index in cell_indices:
            cell = client.nb.cells[cell_index]
            result: Dict[str, Any] = {"cellId": cell.get("id"), "status": "no-op"}
            results.append(result)
            if cell.cell_type != "code":
                continue
            if cell.source.strip():
                await client.async_execute_cell(cell, cell_index)
                result["status"] = "ok"
            else:
                # As in JupyterLab, an empty cell does not run, and loses its outputs.
                cell.outputs = []
                cell.execution_count = None
            if write_outputs:
                DiskAdapter._write_cell_outputs(file_path, cell_index, cell)
            outputs, budget = DiskAdapter._summarize_outputs(cell.outputs, budget)
            result.update(executionCount=cell.execution_count, outputs=outputs)
            error = next((o for o in cell.outputs if o.output_type == "error"), None)
            if error is not None:
                result.update(status="error", errorName=error.ename, errorValue=error.evalue)
                break
        return results

    async def _run_cells_impl(
        self, file_path: str, cell_id: Optional[str] = None, write_outputs: bool = True
    ) -> dict:
        """
        Run a cell, or all the code cells, of a notebook in the kernel of its session.

        With write_outputs, the outputs go to the notebook file after each cell. Without it,
        the notebook does not change. The result has the outputs, with a limit on their size.
        The run stops at the first cell with an error, as in JupyterLab.
        """
        path = self._session_path(file_path)
        async with self._notebook_lock(path):
            notebook = await _read_notebook_to_run(file_path)
            if cell_id is None:
                cell_indices = [
                    i for i, cell in enumerate(notebook.cells) if cell.cell_type == "code"
                ]
            else:
                cell_index = get_cell_index_from_id_nbformat(notebook, cell_id)
                if cell_index is None:
                    raise ValueError(f"Cell with {cell_id=} not found in notebook at {file_path=}")
                cell_indices = [cell_index]

            km = await self._get_notebook_kernel(path, notebook)
            # Not km.client(): it shares the session of the manager, so its socket identity is
            # the same as other clients, and the kernel can send the replies to the wrong one.
            kc = km.client_factory(parent=km)
            kc.load_connection_info(km.get_connection_info())
            # JupyterLab also runs the cells with the skip-execution tag.
            client = NotebookClient(notebook, km=km, allow_errors=True, skip_cells_with_tag="")
            client.kc = kc
            try:
                # Without stdin, input() fails at once, as no one can reply to it.
                kc.start_channels(stdin=False, hb=False, control=False)
                await kc.wait_for_ready(timeout=None)
                cells = await self._until_kernel_restart(
                    km,
                    self._execute_cells(
                        client, cell_indices, normalize_filepath(file_path), write_outputs
                    ),
                )
            finally:
                kc.stop_channels()

        if write_outputs:
            message = (
                "Ran in the kernel of the notebook on the server, as the notebook is not open "
                "in JupyterLab. The outputs are in the notebook file on disk."
            )
        else:
            message = (
                "Ran in the kernel of the notebook on the server. The notebook did not change."
            )
        if cell_id is not None:
            return {"success": True, "result": {"message": message, **cells[0]}}
        # The run stops at the first error, so only the last cell can have one.
        status = "error" if cells and cells[-1]["status"] == "error" else "ok"
        return {"success": True, "result": {"message": message, "status": status, "cells": cells}}

    async def run_cell(self, cell_id, file_path, username, timeout, write_outputs):
        timeout = _MAX_TIMEOUT if timeout is None else min(timeout, _MAX_TIMEOUT)
        return await _run_with_timeout(
            self._run_cells_impl(file_path, cell_id, write_outputs),
            timeout,
            "Cell execution started",
        )

    async def run_all_cells(self, file_path, timeout, write_outputs):
        timeout = _MAX_TIMEOUT if timeout is None else min(timeout, _MAX_TIMEOUT)
        return await _run_with_timeout(
            self._run_cells_impl(file_path, None, write_outputs),
            timeout,
            "Run all cells started",
        )


class FrontendAdapter(BaseAdapter):
    """Drives the JupyterLab frontend via jupyterlab-ai-commands.

    Falls back to DiskAdapter when no web client is connected.
    """

    _disk = DiskAdapter()

    async def read(self, file_path: str) -> dict:
        res = await run_lab_command(
            "jupyterlab-ai-commands:get-notebook-content",
            {"notebookPath": file_path},
        )
        if no_web_client(res):
            return await self._disk.read(file_path)
        return command_result(res)["content"]

    async def add_cell(self, file_path, content, cell_id, add_above, cell_type, animate=False):
        result = await run_lab_command(
            "jupyterlab-ai-commands:add-cell",
            {
                "notebookPath": file_path,
                "referenceCellId": cell_id,
                "content": content or "",
                "cellType": cell_type,
                "position": "above" if add_above else "below",
            },
        )
        if no_web_client(result):
            return await self._disk.add_cell(file_path, content, cell_id, add_above, cell_type)
        return result

    async def insert_cell(self, file_path, content, insert_index, cell_type, animate=False):
        info = await run_lab_command(
            "jupyterlab-ai-commands:get-notebook-info", {"notebookPath": file_path}
        )
        if no_web_client(info):
            return await self._disk.insert_cell(file_path, content, insert_index, cell_type)
        if not info.get("success"):
            return info
        cells = (info.get("result") or {}).get("cells") or []
        idx = insert_index if insert_index is not None else len(cells)
        if not cells:
            ref, position = None, "below"
        elif idx <= 0:
            ref, position = cells[0]["cellId"], "above"
        elif idx >= len(cells):
            ref, position = cells[-1]["cellId"], "below"
        else:
            ref, position = cells[idx]["cellId"], "above"
        return await run_lab_command(
            "jupyterlab-ai-commands:add-cell",
            {
                "notebookPath": file_path,
                "referenceCellId": ref,
                "content": content or "",
                "cellType": cell_type,
                "position": position,
            },
        )

    async def delete_cell(self, file_path, cell_id):
        result = await run_lab_command(
            "jupyterlab-ai-commands:delete-cell",
            {"notebookPath": file_path, "cellId": cell_id},
        )
        if no_web_client(result):
            return await self._disk.delete_cell(file_path, cell_id)
        return result

    async def edit_cell(self, file_path, cell_id, content, cell_type, animate=False):
        result: dict = {"success": True}
        if content is not None:
            result = await run_lab_command(
                "jupyterlab-ai-commands:set-cell-content",
                {
                    "notebookPath": file_path,
                    "cellId": cell_id,
                    "content": content,
                    "showDiff": False,
                },
            )
        if cell_type is not None and result.get("success"):
            result = await self.select_cell(cell_id, None, file_path)
            if result.get("success"):
                type_command = {
                    "code": "notebook:change-cell-to-code",
                    "markdown": "notebook:change-cell-to-markdown",
                    "raw": "notebook:change-cell-to-raw",
                }[cell_type]
                result = await run_lab_command(type_command)
        if no_web_client(result):
            return await self._disk.edit_cell(file_path, cell_id, content, cell_type)
        return result

    async def get_active_cell_id(self, file_path, username):
        resp = await run_lab_command(
            "jupyterlab-ai-commands:get-notebook-info", {"notebookPath": file_path}
        )
        if no_web_client(resp):
            raise RuntimeError(resp["error"])
        return (resp.get("result") or {}).get("activeCellId")

    async def get_active_notebook(self, username):
        resp = await run_lab_command("jupyterlab-ai-commands:get-notebook-info", {})
        if no_web_client(resp):
            raise RuntimeError(resp["error"])
        return (resp.get("result") or {}).get("notebookPath")

    async def select_cell(self, cell_id, username, file_path):
        from jupyterlab_commands_toolkit.tools import execute_command

        info = await run_lab_command(
            "jupyterlab-ai-commands:get-notebook-info", {"notebookPath": file_path}
        )
        if not info.get("success"):
            return info
        info_result = info.get("result") or {}
        cells = info_result.get("cells") or []
        ids = [c.get("cellId") for c in cells]
        if cell_id not in ids:
            raise ValueError(f"Cell with ID {cell_id} not found in notebook")
        target_index = ids.index(cell_id)
        active_id = info_result.get("activeCellId")
        active_index = ids.index(active_id) if active_id in ids else 0
        distance = target_index - active_index
        if distance == 0:
            return {"success": True, "result": "Already at target cell"}
        cmd = "notebook:move-cursor-down" if distance > 0 else "notebook:move-cursor-up"
        move_result: dict = {}
        for _ in range(abs(distance)):
            move_result = await execute_command(cmd)
        return move_result

    async def run_cell(self, cell_id, file_path, username, timeout, write_outputs):
        if not write_outputs:
            if not file_path:
                raise ValueError("file_path is required when write_outputs is False")
            return await self._disk.run_cell(cell_id, file_path, username, timeout, write_outputs)
        # RTC-free: jupyterlab-ai-commands run-cell targets the cell by id and
        # opens the notebook itself, so no awareness-based select is needed.
        result = await _run_with_timeout(
            run_lab_command(
                "jupyterlab-ai-commands:run-cell",
                {"notebookPath": file_path, "cellId": cell_id},
            ),
            timeout,
            "Cell execution started",
        )
        if file_path and no_web_client(result):
            return await self._disk.run_cell(cell_id, file_path, username, timeout, True)
        return result

    async def run_all_cells(self, file_path, timeout, write_outputs):
        from jupyterlab_commands_toolkit.tools import execute_command

        from ..toolkits.jupyterlab import open_file

        if not write_outputs:
            if not file_path:
                raise ValueError("file_path is required when write_outputs is False")
            return await self._disk.run_all_cells(file_path, timeout, write_outputs)
        if file_path:
            result = await open_file(file_path)
            if no_web_client(result):
                return await self._disk.run_all_cells(file_path, timeout, True)
            if not result.get("success"):
                return result
        return await _run_with_timeout(
            execute_command("notebook:run-all-cells"), timeout, "Run all cells started"
        )


class YDocAdapter(BaseAdapter):
    """Mutates the live YDoc when the notebook has an open collaboration room.

    Falls back to DiskAdapter when the notebook has no open room (ydoc is None).
    The ydoc is fetched lazily on the first write operation so that reads never
    trigger a room lookup.
    """

    _disk = DiskAdapter()

    def __init__(self, file_path: Optional[str]):
        self._file_path = file_path
        self._ydoc = None
        self._loaded = False

    @staticmethod
    def _is_single_empty_notebook(ydoc: YNotebook) -> bool:
        """True iff the notebook has exactly one cell and it is empty.

        Mirrors jupyterlab-ai-commands' add-cell behavior, which replaces a lone
        empty first cell instead of appending a new one, so the RTC and RTC-free
        paths agree.
        """
        try:
            cells = ydoc.get().get("cells", [])
            if len(cells) != 1:
                return False
            source = cells[0].get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            return not source.strip()
        except Exception:
            return False

    @staticmethod
    def _atomic_replace_cell_source(ycell, content: str) -> None:
        """Atomically replaces cell source: clear then insert new, no await between ops."""
        old_content = ycell.to_py().get("source", "")
        if old_content == content:
            return
        cell_source = ycell["source"]
        cell_source.clear()
        cell_source += content

    @staticmethod
    def _get_cell_index_from_id(ydoc, cell_id: str) -> Optional[int]:
        for i, ycell in enumerate(ydoc.ycells):
            if ycell.get("id") == cell_id:
                return i
        return None

    @staticmethod
    def _get_active_cell_id(ydoc, username: Optional[str] = None) -> Optional[str]:
        if not ydoc or not ydoc.awareness:
            return None
        for _, state in ydoc.awareness.states.items():
            _username = state.get("user", {}).get("username", None)
            if username and username != _username:
                continue
            if active_cell_id := state.get("activeCellId"):
                return active_cell_id
        return None

    async def _load_ydoc(self):
        if not self._loaded:
            if self._file_path is None:
                return None
            file_id = await get_file_id(self._file_path)
            self._ydoc = await get_jupyter_ydoc(file_id)
            self._loaded = True
        return self._ydoc

    async def read(self, file_path: str) -> dict:
        return await self._disk.read(file_path)

    async def add_cell(self, file_path, content, cell_id, add_above, cell_type, animate=False):
        ydoc = await self._load_ydoc()
        if ydoc is None:
            return await self._disk.add_cell(file_path, content, cell_id, add_above, cell_type)
        cells_count = ydoc.cell_number
        cell_index = self._get_cell_index_from_id(ydoc, cell_id) if cell_id else None
        insert_index = determine_insert_index(cells_count, cell_index, add_above)
        cell: Dict[str, Any] = {"cell_type": cell_type, "source": ""}
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        ycell = ydoc.create_ycell(cell)
        if self._is_single_empty_notebook(ydoc):
            del ydoc.ycells[0]
            ydoc.ycells.append(ycell)
        elif insert_index >= cells_count:
            ydoc.ycells.append(ycell)
        else:
            ydoc.ycells.insert(insert_index, ycell)
        if animate:
            await self.write_to_cell_collaboratively(ydoc, ycell, content or "")
        else:
            self._atomic_replace_cell_source(ycell, content or "")
        return None

    async def insert_cell(self, file_path, content, insert_index, cell_type, animate=False):
        ydoc = await self._load_ydoc()
        if ydoc is None:
            return await self._disk.insert_cell(file_path, content, insert_index, cell_type)
        cells_count = ydoc.cell_number
        cell: Dict[str, Any] = {"cell_type": cell_type, "source": ""}
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        ycell = ydoc.create_ycell(cell)
        if self._is_single_empty_notebook(ydoc):
            del ydoc.ycells[0]
            ydoc.ycells.append(ycell)
        elif insert_index is None or insert_index >= cells_count:
            ydoc.ycells.append(ycell)
        else:
            ydoc.ycells.insert(insert_index, ycell)
        if animate:
            await self.write_to_cell_collaboratively(ydoc, ycell, content or "")
        else:
            self._atomic_replace_cell_source(ycell, content or "")
        return None

    async def delete_cell(self, file_path, cell_id):
        ydoc = await self._load_ydoc()
        if ydoc is None:
            return await self._disk.delete_cell(file_path, cell_id)
        cell_index = self._get_cell_index_from_id(ydoc, cell_id)
        if cell_index is None or cell_index >= len(ydoc.ycells):
            raise ValueError(f"Could not find cell index for {cell_id=}")
        del ydoc.ycells[cell_index]
        return {"success": True}

    async def edit_cell(self, file_path, cell_id, content, cell_type, animate=False):
        ydoc = await self._load_ydoc()
        if ydoc is None:
            return await self._disk.edit_cell(file_path, cell_id, content, cell_type)
        cell_index = self._get_cell_index_from_id(ydoc, cell_id)
        if cell_index is None:
            raise ValueError(f"Cell with {cell_id=} not found in notebook")
        ycell = ydoc._ycells[cell_index]
        old_cell = ycell.to_py()
        old_type = old_cell.get("cell_type")
        needs_type_change = cell_type is not None and cell_type != old_type
        source = content if content is not None else old_cell.get("source", "")
        if needs_type_change:
            new_cell: Dict[str, Any] = {
                "cell_type": cell_type,
                "source": source,
                "id": old_cell.get("id", str(uuid4())),
                "metadata": old_cell.get("metadata", {}),
            }
            if cell_type == "code":
                new_cell["outputs"] = []
                new_cell["execution_count"] = None
            ydoc.set_cell(cell_index, new_cell)
            if animate and content is not None:
                new_ycell = ydoc._ycells[cell_index]
                self._atomic_replace_cell_source(new_ycell, "")
                await self.write_to_cell_collaboratively(ydoc, new_ycell, source)
        elif content is not None:
            if animate:
                await self.write_to_cell_collaboratively(ydoc, ycell, content)
            else:
                self._atomic_replace_cell_source(ycell, content)
        return {"success": True}

    async def get_active_cell_id(self, file_path, username):
        ydoc = await self._load_ydoc()
        return self._get_active_cell_id(ydoc, username)

    async def get_active_notebook(self, username):
        awareness = await get_global_awareness()
        if not awareness:
            return None
        for _, state in awareness.states.items():
            _username = state.get("user", {}).get("username", None)
            if username and username != _username:
                continue
            if (active_notebook := state.get("current")) and "notebook" in active_notebook:
                return active_notebook.replace("notebook:", "")
            if documents := state.get("documents"):
                notebooks = [doc for doc in documents if doc.endswith(".ipynb")]
                if len(notebooks) == 1:
                    return notebooks[0]
        return None

    async def select_cell(self, cell_id, username, file_path):
        from jupyterlab_commands_toolkit.tools import execute_command

        ydoc = await self._load_ydoc()
        if not ydoc:
            raise RuntimeError(f"Notebook at {file_path} is not currently open")
        target_cell_index = self._get_cell_index_from_id(ydoc, cell_id)
        if target_cell_index is None:
            raise ValueError(f"Cell with ID {cell_id} not found in notebook")
        active_cell_id = self._get_active_cell_id(ydoc, username)
        if not active_cell_id:
            raise RuntimeError("No active cell found. Make sure the notebook is focused.")
        active_cell_index = self._get_cell_index_from_id(ydoc, active_cell_id)
        if active_cell_index is None:
            raise RuntimeError(f"Active cell {active_cell_id} not found in notebook")
        distance = target_cell_index - active_cell_index
        if distance == 0:
            return {"success": True, "result": "Already at target cell"}
        result: dict = {}
        if distance > 0:
            for _ in range(distance):
                result = await execute_command("notebook:move-cursor-down")
        else:
            for _ in range(abs(distance)):
                result = await execute_command("notebook:move-cursor-up")
        return result

    async def run_cell(self, cell_id, file_path, username, timeout, write_outputs):
        if not write_outputs:
            if not file_path:
                raise ValueError("file_path is required when write_outputs is False")
            return await self._disk.run_cell(cell_id, file_path, username, timeout, write_outputs)
        result = await _run_with_timeout(
            run_lab_command(
                "jupyterlab-ai-commands:run-cell",
                {"notebookPath": file_path, "cellId": cell_id},
            ),
            timeout,
            "Cell execution started",
        )
        if file_path and no_web_client(result):
            return await self._disk.run_cell(cell_id, file_path, username, timeout, True)
        return result

    async def run_all_cells(self, file_path, timeout, write_outputs):
        from jupyterlab_commands_toolkit.tools import execute_command

        from ..toolkits.jupyterlab import open_file

        if not write_outputs:
            if not file_path:
                raise ValueError("file_path is required when write_outputs is False")
            return await self._disk.run_all_cells(file_path, timeout, write_outputs)
        if file_path:
            result = await open_file(file_path)
            if not result.get("success"):
                return result
        return await _run_with_timeout(
            execute_command("notebook:run-all-cells"), timeout, "Run all cells started"
        )

    @staticmethod
    def get_cursor_details(
        cell_source: Text, start_index: int, stop_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """Creates cursor details for collaborative notebook cursor positioning."""
        head_sticky_index = cell_source.sticky_index(start_index, Assoc.BEFORE)
        head_sticky_index_data = head_sticky_index.to_json()

        cursor_details: Dict[str, Any] = {"primary": True, "empty": True}
        cursor_details["head"] = {
            "type": head_sticky_index_data["item"],
            "tname": None,
            "item": head_sticky_index_data["item"],
            "assoc": 0,
        }
        cursor_details["anchor"] = cursor_details["head"]

        if stop_index is not None:
            anchor_sticky_index = cell_source.sticky_index(stop_index, Assoc.BEFORE)
            anchor_sticky_index_data = anchor_sticky_index.to_json()
            cursor_details["anchor"] = {
                "type": anchor_sticky_index_data["item"],
                "tname": None,
                "item": anchor_sticky_index_data["item"],
                "assoc": 0,
            }
            cursor_details["empty"] = False

        return cursor_details

    @staticmethod
    def set_cursor_in_ynotebook(
        ynotebook: YNotebook, cell_source: Text, start_index: int, stop_index: Optional[int] = None
    ) -> None:
        """Sets the cursor position in a collaborative notebook environment."""
        try:
            details = YDocAdapter.get_cursor_details(
                cell_source, start_index, stop_index=stop_index
            )
            if ynotebook.awareness:
                ynotebook.awareness.set_local_state_field("cursors", [details])
        except Exception:
            pass

    @staticmethod
    def _safe_set_cursor(
        ynotebook: YNotebook,
        cell_source: Text,
        cursor_position: int,
        stop_cursor: Optional[int] = None,
    ) -> None:
        """Safely set cursor position with error handling."""
        try:
            YDocAdapter.set_cursor_in_ynotebook(
                ynotebook, cell_source, cursor_position, stop_cursor
            )
        except Exception:
            pass

    @staticmethod
    async def _handle_delete_operation(
        ynotebook, cell_source, cursor_position: int, delete_length: int, typing_speed: float
    ) -> None:
        """Handle deletion of text chunks with visual feedback."""
        YDocAdapter._safe_set_cursor(
            ynotebook, cell_source, cursor_position, cursor_position + delete_length
        )
        await asyncio.sleep(min(0.3, typing_speed * 3))
        del cell_source[cursor_position : cursor_position + delete_length]
        await asyncio.sleep(typing_speed)

    @staticmethod
    async def _handle_insert_operation(
        ynotebook,
        cell_source,
        cursor_position: int,
        new_content: str,
        new_start: int,
        new_end: int,
        typing_speed: float,
    ) -> int:
        """Handle insertion of text with word-by-word typing simulation."""
        text_to_insert = new_content[new_start:new_end]
        words = text_to_insert.split()

        if not words or text_to_insert.strip() == "":
            cell_source.insert(cursor_position, text_to_insert)
            cursor_position += len(text_to_insert)
            YDocAdapter._safe_set_cursor(ynotebook, cell_source, cursor_position)
            await asyncio.sleep(typing_speed)
            return cursor_position

        current_pos = 0
        for word in words:
            word_start = text_to_insert.find(word, current_pos)
            if word_start > current_pos:
                prefix = text_to_insert[current_pos:word_start]
                cell_source.insert(cursor_position, prefix)
                cursor_position += len(prefix)
            cell_source.insert(cursor_position, word)
            cursor_position += len(word)
            current_pos = word_start + len(word)
            YDocAdapter._safe_set_cursor(ynotebook, cell_source, cursor_position)
            await asyncio.sleep(typing_speed)

        if current_pos < len(text_to_insert):
            suffix = text_to_insert[current_pos:]
            cell_source.insert(cursor_position, suffix)
            cursor_position += len(suffix)
            YDocAdapter._safe_set_cursor(ynotebook, cell_source, cursor_position)

        return cursor_position

    @staticmethod
    async def _handle_replace_operation(
        ynotebook,
        cell_source,
        cursor_position: int,
        new_content: str,
        delete_length: int,
        new_start: int,
        new_end: int,
        typing_speed: float,
    ) -> int:
        """Handle replacement operations by deleting then inserting."""
        await YDocAdapter._handle_delete_operation(
            ynotebook, cell_source, cursor_position, delete_length, typing_speed
        )
        await asyncio.sleep(typing_speed * 2)
        cursor_position = await YDocAdapter._handle_insert_operation(
            ynotebook, cell_source, cursor_position, new_content, new_start, new_end, typing_speed
        )
        return cursor_position

    @staticmethod
    async def write_to_cell_collaboratively(
        ynotebook, ycell, content: str, typing_speed: float = 0.1
    ) -> bool:
        """Writes content to a Jupyter notebook cell with collaborative typing simulation."""
        if ynotebook is None:
            raise ValueError("ynotebook cannot be None")
        if ycell is None:
            raise ValueError("ycell cannot be None")
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        if typing_speed < 0:
            raise ValueError("typing_speed must be non-negative")

        try:
            cell = ycell.to_py()
            old_content = cell.get("source", "")
            cell_source = ycell["source"]
            new_content = content

            if old_content == new_content:
                return True

        except Exception as e:
            raise RuntimeError(f"Failed to extract cell content: {e}")

        try:
            sequence_matcher = difflib.SequenceMatcher(None, old_content, new_content)
            cursor_position = 0

            YDocAdapter._safe_set_cursor(ynotebook, cell_source, cursor_position)

            for operation, old_start, old_end, new_start, new_end in sequence_matcher.get_opcodes():
                if operation == "equal":
                    cursor_position += old_end - old_start

                elif operation == "delete":
                    delete_length = old_end - old_start
                    await YDocAdapter._handle_delete_operation(
                        ynotebook, cell_source, cursor_position, delete_length, typing_speed
                    )

                elif operation == "insert":
                    cursor_position = await YDocAdapter._handle_insert_operation(
                        ynotebook,
                        cell_source,
                        cursor_position,
                        new_content,
                        new_start,
                        new_end,
                        typing_speed,
                    )

                elif operation == "replace":
                    delete_length = old_end - old_start
                    cursor_position = await YDocAdapter._handle_replace_operation(
                        ynotebook,
                        cell_source,
                        cursor_position,
                        new_content,
                        delete_length,
                        new_start,
                        new_end,
                        typing_speed,
                    )

            YDocAdapter._safe_set_cursor(ynotebook, cell_source, cursor_position)

            return True

        except Exception as e:
            raise RuntimeError(f"Failed to write cell content collaboratively: {e}")


async def get_adapter(file_path: Optional[str] = None) -> BaseAdapter:
    """Return the BaseAdapter appropriate for the given notebook path.

    With RTC: returns a YDocAdapter that fetches the ydoc lazily on first write.
    Without RTC: returns a FrontendAdapter that falls back to disk when no browser is connected.
    """
    if rtc_available():
        return YDocAdapter(normalize_filepath(file_path) if file_path else None)
    return FrontendAdapter()
