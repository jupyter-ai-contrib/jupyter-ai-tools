# jupyter_ai_tools

[![Github Actions Status](https://github.com/Abigayle-Mercer/jupyter-ai-tools/workflows/Build/badge.svg)](https://github.com/Abigayle-Mercer/jupyter-ai-tools/actions/workflows/build.yml)

**`jupyter_ai_tools`** is a Jupyter Server extension that exposes a collection of powerful, agent-friendly tools for interacting with notebooks and Git repositories. It is designed for use by AI personas (like those in Jupyter AI) to programmatically modify notebooks, manage code cells, and interact with version control systems.

______________________________________________________________________

## ✨ Features

This extension provides runtime-discoverable tools compatible with OpenAI-style function calling or MCP tool schemas. These tools can be invoked by agents to:

### 🧠 YNotebook Tools

- `read_cell`: Return the full content of a cell by index
- `read_notebook`: Return all cells as a JSON-formatted list
- `add_cell`: Insert a blank cell at a specific index
- `delete_cell`: Remove a cell and return its contents
- `write_to_cell`: Overwrite the content of a cell with new source
- `get_max_cell_index`: Return the last valid cell index

### 🌀 Git Tools

- `git_clone`: Clone a Git repo into a given path
- `git_status`: Get the working tree status
- `git_log`: View recent commit history
- `git_add`: Stage files (individually or all)
- `git_commit`: Commit staged changes with a message
- `git_push`: Push local changes to a remote branch
- `git_pull`: Pull remote updates
- `git_get_repo_root_from_notebookpath`: Find the Git root from a notebook path

These tools are ideal for agents that assist users with code editing, version control, or dynamic notebook interaction.

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
