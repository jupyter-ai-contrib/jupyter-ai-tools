from jupyter_server_ai_tools.models import ToolDefinition

from . import git_tools, ynotebook_tools


def jupyter_server_extension_tools():
    return [
        ToolDefinition(
            callable=ynotebook_tools.delete_cell,
            metadata={
                "name": "delete_cell",
                "description": "Remove the cell at the specified index and return its contents.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "integer",
                            "description": "The index of the cell to delete",
                        }
                    },
                    "required": ["index"],
                },
            },
        ),
        ToolDefinition(
            callable=ynotebook_tools.add_cell,
            metadata={
                "name": "add_cell",
                "description": "Insert a blank cell at the specified index.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer", "description": "The index to insert at"},
                        "cell_type": {
                            "type": "string",
                            "description": "The type of cell: 'code' or 'markdown' ",
                            "default": "code",
                        },
                    },
                    "required": ["index"],
                },
            },
        ),
        ToolDefinition(
            callable=ynotebook_tools.write_to_cell,
            metadata={
                "name": "write_to_cell",
                "description": "Overwrite the source of a cell with content at the given index "
                "in the notebook.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer", "description": "The index to write at"},
                        "content": {
                            "type": "string",
                            "description": "The content to write into the cell, either python "
                            "code or markdown",
                        },
                    },
                    "required": ["index", "content"],
                },
            },
        ),
        ToolDefinition(
            callable=ynotebook_tools.get_max_cell_index,
            metadata={
                "name": "get_max_cell_index",
                "description": "Return the highest valid cell index in the current notebook.",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ),
        ToolDefinition(
            callable=ynotebook_tools.read_cell,
            metadata={
                "name": "read_cell",
                "description": "Read the full content of a specific cell, including outputs, "
                "source, and metadata.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer", "description": "The index of the cell to read"}
                    },
                    "required": ["index"],
                },
            },
        ),
        ToolDefinition(
            callable=ynotebook_tools.read_notebook,
            metadata={
                "name": "read_notebook",
                "description": "Return all cells in the notebook as a JSON-formatted list.",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ),
        ToolDefinition(
            callable=git_tools.git_clone,
            metadata={
                "name": "git_clone",
                "description": "Clone a Git repo into the specified path.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Target path"},
                        "url": {"type": "string", "description": "Repository URL"},
                    },
                    "required": ["path", "url"],
                },
            },
        ),
        ToolDefinition(
            callable=git_tools.git_status,
            metadata={
                "name": "git_status",
                "description": "Get the current Git status in the specified path.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the Git repository root directory",
                        }
                    },
                    "required": ["path"],
                },
            },
        ),
        ToolDefinition(
            callable=git_tools.git_log,
            metadata={
                "name": "git_log",
                "description": "Get the last N Git commits.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the Git repository root directory",
                        },
                        "history_count": {
                            "type": "integer",
                            "description": "Number of commits",
                            "default": 10,
                        },
                    },
                    "required": ["path"],
                },
            },
        ),
        ToolDefinition(
            callable=git_tools.git_pull,
            metadata={
                "name": "git_pull",
                "description": "Pull the latest changes from the remote.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the Git repository root directory",
                        }
                    },
                    "required": ["path"],
                },
            },
        ),
        ToolDefinition(
            callable=git_tools.git_push,
            metadata={
                "name": "git_push",
                "description": "Push local changes to the remote.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the Git repository root directory",
                        },
                        "branch": {"type": "string", "description": "Repo branch"},
                    },
                    "required": ["path"],
                },
            },
        ),
        ToolDefinition(
            callable=git_tools.git_commit,
            metadata={
                "name": "git_commit",
                "description": "Commit staged changes with a message.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the Git repository root directory",
                        },
                        "message": {"type": "string", "description": "Commit message"},
                    },
                    "required": ["path", "message"],
                },
            },
        ),
        ToolDefinition(
            callable=git_tools.git_add,
            metadata={
                "name": "git_add",
                "description": "Stage files for commit. Optionally add all files.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the Git repository root directory",
                        },
                        "add_all": {
                            "type": "boolean",
                            "default": True,
                            "description": "Stage all files",
                        },
                        "filename": {
                            "type": "string",
                            "description": "File to add (used if add_all is false)",
                        },
                    },
                    "required": ["path"],
                },
            },
        ),
        ToolDefinition(
            callable=git_tools.git_get_repo_root,
            metadata={
                "name": "git_get_repo_root_from_notebookpath",
                "description": "Given the path of a file, return the path to the Repo root"
                " if any.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "the path to a file"}},
                    "required": ["path"],
                },
            },
        ),
    ]
