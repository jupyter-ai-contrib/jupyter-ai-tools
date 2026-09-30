import functools
import inspect
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

from jupyter_server.auth.identity import User
from jupyter_server.serverapp import ServerApp
from pycrdt import Awareness

JSD_PRESENT = False
try:
    import jupyter_server_documents  # noqa: F401

    JSD_PRESENT = True
except ImportError:
    pass


def get_serverapp():
    """Returns the server app from the request context"""

    server = ServerApp.instance()
    return server


# --- RTC availability -------------------------------------------------------
# A minimal version of jupyterlab_chat's rtc_lib: is a real-time collaboration
# provider actually active this session? "installed" is not "enabled", so we
# check the extension manager (and honor jupyter_server_ydoc's disable_rtc
# trait). When no provider is active, the tools fall back to driving the
# JupyterLab frontend via jupyterlab-ai-commands.
_RTC_PROVIDERS = ("jupyter_server_documents", "jupyter_server_ydoc")


def rtc_available() -> bool:
    """Return True iff an RTC provider is enabled for the current session."""
    try:
        serverapp = get_serverapp()
        manager = serverapp.extension_manager
    except Exception:
        return False

    for name in _RTC_PROVIDERS:
        ext = manager.extensions.get(name)
        if not (ext and getattr(ext, "enabled", False)):
            continue
        if name == "jupyter_server_ydoc":
            # jupyter_server_ydoc can be turned off via YDocExtension.disable_rtc.
            apps = getattr(manager, "extension_apps", {}) or {}
            instances = apps.get(name)
            app = next(iter(instances)) if instances else None
            if app is not None and bool(getattr(app, "disable_rtc", False)):
                continue
        return True
    return False


async def run_lab_command(command_id: str, args: Optional[Dict] = None) -> dict:
    """Execute a JupyterLab frontend command (via jupyterlab-commands-toolkit).

    Used by the RTC-free tool implementations to drive jupyterlab-ai-commands.
    """
    from jupyterlab_commands_toolkit.tools import execute_command

    return await execute_command(command_id, args or {})


def no_web_client(result: Any) -> bool:
    """Whether a frontend command failed because no JupyterLab web client received it.

    jupyterlab-commands-toolkit reports this with an error code. An older
    version only times out, and the result then has no error code.
    """
    return isinstance(result, dict) and result.get("error_code") == "no_web_client"


def command_result(result: dict) -> Any:
    """Return the value of a successful frontend command, or raise its error."""
    if not result.get("success"):
        raise RuntimeError(result.get("error") or "The JupyterLab command failed")
    return result.get("result")


def normalize_filepath(file_path: str) -> str:
    """
    Normalizes a file path for Jupyter applications to return an absolute path.

    Handles various input formats:
    - Relative paths from current working directory
    - URL-encoded relative paths (common in Jupyter contexts)
    - Absolute paths (returned as-is after normalization)

    Args:
        file_path: Path in any of the supported formats

    Returns:
        Absolute path to the file

    Example:
        >>> normalize_filepath("notebooks/my%20notebook.ipynb")
        "/current/working/dir/notebooks/my notebook.ipynb"
        >>> normalize_filepath("/absolute/path/file.ipynb")
        "/absolute/path/file.ipynb"
        >>> normalize_filepath("relative/file.ipynb")
        "/current/working/dir/relative/file.ipynb"
    """
    if not file_path or not file_path.strip():
        raise ValueError("file_path cannot be empty")

    # URL decode the path in case it contains encoded characters
    decoded_path = unquote(file_path)

    # Convert to Path object for easier manipulation
    path = Path(decoded_path)

    # If already absolute, just normalize and return
    if path.is_absolute():
        return str(path.resolve())

    # For relative paths, get the Jupyter server's root directory
    try:
        serverapp = get_serverapp()
        root_dir = serverapp.root_dir
    except Exception:
        # Fallback to current working directory if server app is not available
        root_dir = os.getcwd()

    # Resolve relative path against the root directory
    resolved_path = Path(root_dir) / path
    return str(resolved_path.resolve())


async def get_room(room_id: str):
    """Returns the yroom."""

    serverapp = get_serverapp()
    if JSD_PRESENT:
        manager = serverapp.web_app.settings["yroom_manager"]
        return manager.get_room(room_id) if manager.has_room(room_id) else None
    else:
        manager = serverapp.web_app.settings["jupyter_server_ydoc"].ywebsocket_server
        return await manager.get_room(room_id) if manager.room_exists(room_id) else None


async def get_jupyter_ydoc(file_id: str):
    """Returns the notebook ydoc

    Args:
        file_id: The file ID for the document

    Returns:
        `YNotebook` ydoc for the notebook
    """
    room_id = f"json:notebook:{file_id}"
    yroom = await get_room(room_id)
    if not yroom:
        return

    if JSD_PRESENT:
        return await yroom.get_jupyter_ydoc()
    else:
        return yroom._document


async def get_global_awareness() -> Optional[Awareness]:
    """Returns the global awareness object for JupyterLab collaboration.

    Returns:
        Awareness instance for global collaboration, or None if not available
    """
    yroom = await get_room("JupyterLab:globalAwareness")
    if not yroom:
        return None

    if JSD_PRESENT:
        return yroom.get_awareness()
    else:
        return yroom.awareness


async def get_file_id(file_path: str) -> str:
    """Returns the file_id for the document

    Args:
        file_path:
            absolute path to the document file

    Returns:
        The file ID of the document
    """
    normalized_file_path = normalize_filepath(file_path)

    serverapp = get_serverapp()
    file_id_manager = serverapp.web_app.settings["file_id_manager"]
    file_id = file_id_manager.get_id(normalized_file_path)

    return file_id


def collaborative_tool(user: User):
    """
    Decorator factory to enable collaborative awareness for toolkit functions.

    This decorator automatically sets up user awareness in the global
    and notebook-specific awareness systems when functions are called.
    It enables real-time collaborative features by making the user's
    presence visible to other users in the same Jupyter environment.

    Args:
        user: Optional user dictionary with user information. If None, no awareness is set.
              Should contain keys like 'name', 'color', 'display_name', etc.

    Returns:
        Decorator function that wraps the target function with collaborative awareness.

    Example:
        >>> user_info = {
        ...     "name": "Alice",
        ...     "color": "var(--jp-collaborator-color1)",
        ...     "display_name": "Alice Smith"
        ... }
        >>>
        >>> @collaborative_tool(user=user_info)
        ... async def my_notebook_tool(file_path: str, content: str):
        ...     # Your tool implementation here
        ...     return f"Processed {file_path}"
    """

    def decorator(tool_func):
        @functools.wraps(tool_func)
        async def wrapper(*args, **kwargs):
            # Skip awareness if no user provided

            # Get serverapp for logging
            try:
                serverapp = get_serverapp()
                logger = serverapp.log
            except Exception:
                logger = None

            # Extract file_path from tool function arguments for notebook-specific awareness
            file_path = None
            try:
                # Try to find file_path in kwargs first
                if "file_path" in kwargs:
                    file_path = kwargs["file_path"]
                else:
                    # Try to find file_path in positional args by inspecting the function signature
                    sig = inspect.signature(tool_func)
                    param_names = list(sig.parameters.keys())

                    # Look for file_path parameter
                    if "file_path" in param_names and len(args) > param_names.index("file_path"):
                        file_path = args[param_names.index("file_path")]
            except Exception as e:
                # Log error in file_path detection
                if logger:
                    logger.warning(f"Error detecting file_path in collaborative_tool: {e}")

            # Set notebook-specific collaborative awareness if we have a file_path
            if file_path and file_path.endswith(".ipynb"):
                try:
                    file_id = await get_file_id(file_path)
                    ydoc = await get_jupyter_ydoc(file_id)

                    if ydoc:
                        # Set the local user field in the notebook's awareness
                        ydoc.awareness.set_local_state_field("user", user)
                except Exception as e:
                    # Log error but don't block tool execution
                    if logger:
                        logger.warning(
                            f"Error setting notebook awareness in collaborative_tool: {e}"
                        )

            # Set global awareness
            try:
                g_awareness = await get_global_awareness()
                if g_awareness:
                    g_awareness.set_local_state(
                        {
                            "user": user,
                            "current": file_path or "",
                            "documents": [file_path] if file_path else [],
                        }
                    )
            except Exception as e:
                # Log error but don't block tool execution
                if logger:
                    logger.warning(f"Error setting global awareness in collaborative_tool: {e}")

            # Execute the original tool function
            return await tool_func(*args, **kwargs)

        return wrapper

    return decorator


def notebook_json_to_md(notebook_json: dict, include_outputs: bool = True) -> str:
    """Converts a notebook json dict to markdown string using a custom format.

    Args:
        notebook_json: The notebook JSON dictionary
        include_outputs: Whether to include cell outputs in the markdown. Default is True.

    Returns:
        Markdown string representation of the notebook

    Example:
        ```markdown
        ```yaml
        kernelspec:
          display_name: Python 3
          language: python
          name: python3
        ```

        ### Cell 0

        #### Metadata
        ```yaml
        id: 8f2c1a3b
        type: code
        execution_count: 1
        ```

        #### Source
        ```python
        print("Hello world")
        ```

        #### Output
        ```
        Hello world
        ```
        ```
    """
    # Extract notebook metadata
    md_parts = []

    # Add notebook metadata at the top
    md_parts.append(metadata_to_md(notebook_json.get("metadata", {})))

    # Process all cells
    for i, cell in enumerate(notebook_json.get("cells", [])):
        md_parts.append(cell_to_md(cell, index=i, include_outputs=include_outputs))

    # Join all parts with double newlines
    return "\n\n".join(md_parts)


def metadata_to_md(metadata_json: Dict) -> str:
    """Converts notebook or cell metadata to markdown string in YAML format.

    Args:
        metadata_json: The metadata JSON dictionary

    Returns:
        Markdown string with YAML formatted metadata
    """
    import yaml  # type: ignore[import-untyped]

    yaml_str = yaml.dump(metadata_json, default_flow_style=False)
    return f"```yaml\n{yaml_str}```"


def cell_to_md(cell_json: dict, index: int = 0, include_outputs: bool = True) -> str:
    """Converts notebook cell to markdown string.

    Args:
        cell_json: The cell JSON dictionary
        index: Cell index number for the heading
        include_outputs: Whether to include cell outputs in the markdown

    Returns:
        Markdown string representation of the cell
    """
    md_parts = []

    # Add cell heading with index
    md_parts.append(f"### Cell {index}")

    # Add metadata section
    md_parts.append("#### Metadata")
    metadata = {
        "id": cell_json.get("id"),
        "type": cell_json.get("cell_type"),
        "execution_count": cell_json.get("execution_count"),
    }
    # Filter out None values
    metadata = {k: v for k, v in metadata.items() if v is not None}
    # Add any additional metadata from the cell
    if "metadata" in cell_json:
        for key, value in cell_json["metadata"].items():
            metadata[key] = value

    md_parts.append(metadata_to_md(metadata))

    # Add source section
    md_parts.append("#### Source")
    source = "".join(cell_json.get("source", []))

    if cell_json.get("cell_type") == "code":
        # For code cells, use python code block
        md_parts.append(f"```python\n{source}```")
    else:
        # For markdown cells, use regular code block
        md_parts.append(f"```\n{source}```")

    # Add output section if available and requested
    if (
        include_outputs
        and cell_json.get("cell_type") == "code"
        and "outputs" in cell_json
        and cell_json["outputs"]
    ):
        md_parts.append("#### Output")
        md_parts.append(format_outputs(cell_json["outputs"]))

    return "\n\n".join(md_parts)


def format_outputs(outputs: List) -> str:
    """Formats cell outputs into markdown.

    Args:
        outputs: List of cell output dictionaries

    Returns:
        Formatted markdown string of the outputs
    """
    result = []

    for output in outputs:
        output_type = output.get("output_type")

        if output_type == "stream":
            text = "".join(output.get("text", []))
            result.append(f"```\n{text}```")

        elif output_type == "execute_result" or output_type == "display_data":
            data = output.get("data", {})

            # Handle text/plain output
            if "text/plain" in data:
                text = "".join(data["text/plain"])
                result.append(f"```\n{text}```")

            # TODO: Add other mime types

        elif output_type == "error":
            traceback = "\n".join(output.get("traceback", []))
            result.append(f"```\n{traceback}```")

    return "\n\n".join(result)
