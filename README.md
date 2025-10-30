# jupyter_ai_tools

[![Github Actions Status](https://github.com/Abigayle-Mercer/jupyter-ai-tools/workflows/Build/badge.svg)](https://github.com/Abigayle-Mercer/jupyter-ai-tools/actions/workflows/build.yml)

**`jupyter_ai_tools`** is a Jupyter Server extension that exposes a collection of powerful, agent-friendly tools for interacting with notebooks and Git repositories. It is designed for use by AI personas (like those in Jupyter AI) to programmatically modify notebooks, manage code cells, and interact with version control systems.

______________________________________________________________________

## ✨ Features

This extension provides runtime-discoverable tools compatible with OpenAI-style function calling or MCP tool schemas. These tools can be invoked by agents to:

### 📁 File System Tools (`fs_toolkit`)

- `read`: Read file contents from the filesystem
- `edit`: Edit file contents with search and replace functionality
- `write`: Write content to a file
- `search_and_replace`: Search and replace text patterns in files
- `glob`: Find files matching a glob pattern
- `grep`: Search for text patterns within file contents
- `ls`: List directory contents

### 🧠 Notebook Tools (`nb_toolkit`)

- `read_notebook`: Read entire notebook contents as markdown
- `read_cell`: Read a specific notebook cell by index
- `add_cell`: Add a new cell to a notebook
- `insert_cell`: Insert a cell at a specific index in the notebook
- `delete_cell`: Remove a cell from the notebook
- `edit_cell`: Modify a cell's content
- `get_cell_id_from_index`: Get cell ID from its index position
- `create_notebook`: Create a new Jupyter notebook

### 🌀 Git Tools (`git_toolkit`)

- `git_clone`: Clone a Git repository to a specified path
- `git_status`: Get the current working tree status
- `git_log`: View recent commit history
- `git_pull`: Pull changes from remote repository
- `git_push`: Push local changes to remote branch
- `git_commit`: Commit staged changes with a message
- `git_add`: Stage files for commit (individually or all)
- `git_get_repo_root`: Get the root directory of the Git repository

### ⚙️ Code Execution Tools (`exec_toolkit`)

- `bash`: Execute bash commands in the system shell

These tools are ideal for agents that assist users with code editing, version control, or dynamic notebook interaction.

______________________________________________________________________

## 🔧 Creating Collaborative Tools

For developers building AI tools that need collaborative awareness, `jupyter_ai_tools` provides a `collaborative_tool` decorator that automatically enables real-time collaboration features.

This decorator enables other users in the same Jupyter environment to see when your AI tool is actively working on shared notebooks, improving the collaborative experience.

```python
from jupyter_ai_tools.utils import collaborative_tool

# Define user information
user_info = {
    "name": "Alice",
    "color": "var(--jp-collaborator-color1)",
    "display_name": "Alice Smith"
}

# Apply collaborative awareness to your tool
@collaborative_tool(user=user_info)
async def my_notebook_tool(file_path: str, content: str):
    """Your tool implementation here"""
    # Tool automatically sets user awareness for:
    # - Global awareness system (all users can see Alice is active)
    # - Notebook-specific awareness (for .ipynb files)
    return f"Processed {file_path}"
```


______________________________________________________________________

## Requirements

- Jupyter Server

## Install

To install the extension, execute:

```bash
pip install jupyter_ai_tools
```

## Uninstall

To remove the extension, execute:

```bash
pip uninstall jupyter_ai_tools
```

## Troubleshoot

If you are seeing the frontend extension, but it is not working, check
that the server extension is enabled:

```bash
jupyter server extension list
```

## Contributing

### Development install

```bash
# Clone the repo to your local environment
# Change directory to the jupyter_ai_tools directory
# Install package in development mode - will automatically enable
# The server extension.
pip install -e .
```

You can watch the source directory and run your Jupyter Server-based application at the same time in different terminals to watch for changes in the extension's source and automatically rebuild the extension. For example,
when running JupyterLab:

```bash
jupyter lab --autoreload
```

If your extension does not depend a particular frontend, you can run the
server directly:

```bash
jupyter server --autoreload
```

### Running Tests

Install dependencies:

```bash
pip install -e ".[test]"
```

### Development uninstall

```bash
pip uninstall jupyter_ai_tools
```

### Packaging the extension

See [RELEASE](RELEASE.md)
