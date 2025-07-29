import functools
import inspect
import typing
from typing import Optional

from jupyter_server.serverapp import ServerApp
from jupyter_server.auth.identity import User
from pycrdt import Awareness


async def get_serverapp():
    """Returns the server app from the request context"""

    server = ServerApp.instance()
    return server


async def get_jupyter_ydoc(file_id: str):
    """Returns the notebook ydoc

    Args:
        file_id: The file ID for the document

    Returns:
        `YNotebook` ydoc for the notebook
    """
    serverapp = await get_serverapp()
    yroom_manager = serverapp.web_app.settings["yroom_manager"]
    room_id = f"json:notebook:{file_id}"

    if yroom_manager.has_room(room_id):
        yroom = yroom_manager.get_room(room_id)
        notebook = await yroom.get_jupyter_ydoc()
        return notebook


async def get_global_awareness() -> Optional[Awareness]:
    serverapp = await get_serverapp()
    yroom_manager = serverapp.web_app.settings["yroom_manager"]
    
    room_id = "JupyterLab:globalAwareness" 
    if yroom_manager.has_room(room_id):
        yroom = yroom_manager.get_room(room_id)
        return yroom.get_awareness()
    
    # Return None if room doesn't exist
    return None


async def get_file_id(file_path: str) -> str:
    """Returns the file_id for the document

    Args:
        file_path:
            absolute path to the document file

    Returns:
        The file ID of the document
    """

    serverapp = await get_serverapp()
    file_id_manager = serverapp.web_app.settings["file_id_manager"]
    file_id = file_id_manager.get_id(file_path)

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
                serverapp = await get_serverapp()
                logger = serverapp.log
            except Exception:
                logger = None
            
            # Extract file_path from tool function arguments for notebook-specific awareness
            file_path = None
            try:
                # Try to find file_path in kwargs first
                if 'file_path' in kwargs:
                    file_path = kwargs['file_path']
                else:
                    # Try to find file_path in positional args by inspecting the function signature
                    sig = inspect.signature(tool_func)
                    param_names = list(sig.parameters.keys())
                    
                    # Look for file_path parameter
                    if 'file_path' in param_names and len(args) > param_names.index('file_path'):
                        file_path = args[param_names.index('file_path')]
            except Exception as e:
                # Log error in file_path detection
                if logger:
                    logger.warning(f"Error detecting file_path in collaborative_tool: {e}")
            
            # Set notebook-specific collaborative awareness if we have a file_path
            if file_path and file_path.endswith('.ipynb'):
                try:
                    file_id = await get_file_id(file_path)
                    ydoc = await get_jupyter_ydoc(file_id)
                    
                    if ydoc:
                        # Set the local user field in the notebook's awareness
                        ydoc.awareness.set_local_state_field("user", user)
                except Exception as e:
                    # Log error but don't block tool execution
                    if logger:
                        logger.warning(f"Error setting notebook awareness in collaborative_tool: {e}")
            
            # Set global awareness
            try:
                g_awareness = await get_global_awareness()
                if g_awareness:
                    g_awareness.set_local_state({
                        "user": user,
                        "current": file_path or "",
                        "documents": [file_path] if file_path else []
                    })
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


def metadata_to_md(metadata_json: dict) -> str:
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


def format_outputs(outputs: list) -> str:
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
