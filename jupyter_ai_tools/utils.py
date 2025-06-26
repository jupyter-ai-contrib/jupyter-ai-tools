from jupyter_server.base.call_context import CallContext


def get_serverapp():
    """Returns the server app from the request context"""
    handler = CallContext.get(CallContext.JUPYTER_HANDLER)
    serverapp = handler.serverapp
    return serverapp


def get_jupyter_ydoc(file_id: str):
    """Returns the notebook ydoc

    Args:
        file_id: The file ID for the document

    Returns:
        `YNotebook` ydoc for the notebook
    """
    serverapp = get_serverapp()
    yroom_manager = serverapp.web_app.settings["yroom_manager"]
    room_id = f"json:notebook:{file_id}"
    if yroom_manager.has_room(room_id):
        yroom = yroom_manager.get_room(room_id)
        notebook = yroom.get_jupyter_ydoc()
        return notebook


def get_file_id(file_path: str) -> str:
    """Returns the file_id for the document

    Args:
        file_path:
            absolute path to the document file

    Returns:
        The file ID of the document
    """

    serverapp = get_serverapp()
    file_id_manager = serverapp.web_app.settings["file_id_manager"]
    file_id = file_id_manager.get_id(file_path)

    return file_id


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
