"""Tools that provide file system functionality"""

import asyncio
import fnmatch
import glob as glob_module
import os
from typing import List, Optional

from jupyter_ai.tools.models import Tool, Toolkit


def read(file_path: str, offset: Optional[int] = None, limit: Optional[int] = None) -> str:
    """Reads a file from the local filesystem

    Args:
        file_path: The absolute path to the file to read
        offset: The line number to start reading from (optional)
        limit: The number of lines to read (optional)

    Returns:
        The contents of the file, potentially with line numbers
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File not found: {file_path}"

        if not os.path.isfile(file_path):
            return f"Error: Not a file: {file_path}"

        content = _read_file_content(file_path, offset, limit)
        return content
    except Exception as e:
        return f"Error: Failed to read file: {str(e)}"


def _read_file_content(
    file_path: str, offset: Optional[int] = None, limit: Optional[int] = None
) -> str:
    """Helper function to read file content in a separate thread"""
    with open(file_path, "r", encoding="utf-8") as f:
        if offset is not None:
            # Skip lines until we reach the offset
            for _ in range(offset):
                line = f.readline()
                if not line:
                    break

        # Read the specified number of lines or all lines if limit is None
        if limit is not None:
            lines = [f.readline() for _ in range(limit)]
            # Filter out None values in case we hit EOF
            lines = [line for line in lines if line]
        else:
            lines = f.readlines()

        # Add line numbers (starting from offset+1 if offset is provided)
        start_line = (offset or 0) + 1
        numbered_lines = [f"{i}→{line}" for i, line in enumerate(lines, start=start_line)]

        return "".join(numbered_lines)


def write(file_path: str, content: str) -> str:
    """Writes content to a file on the local filesystem

    Args:
        file_path: The absolute path to the file to write
        content: The content to write to the file

    Returns:
        A success message or error message
    """
    try:
        # Ensure the directory exists
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        _write_file_content(file_path, content)
        return f"File written successfully at: {file_path}"
    except Exception as e:
        return f"Error: Failed to write file: {str(e)}"


def _write_file_content(file_path: str, content: str) -> None:
    """Helper function to write file content in a separate thread"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def edit(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Performs string replacement in a file

    Args:
        file_path: The absolute path to the file to modify
        old_string: The text to replace
        new_string: The text to replace it with
        replace_all: Replace all occurrences of old_string (default False)

    Returns:
        A success message or error message
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File not found: {file_path}"

        if not os.path.isfile(file_path):
            return f"Error: Not a file: {file_path}"

        # Read the file content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if old_string exists in the file
        if old_string not in content:
            return "Error: String to replace not found in file"

        # Perform the replacement
        if replace_all:
            new_content = content.replace(old_string, new_string)
        else:
            # Replace only the first occurrence
            new_content = content.replace(old_string, new_string, 1)

        # If nothing changed, old and new strings might be identical
        if new_content == content:
            return "Error: No changes made. Old string and new string might be identical"

        # Write the updated content back to the file
        _write_file_content(file_path, new_content)

        return f"File {file_path} has been updated successfully"
    except Exception as e:
        return f"Error: Failed to edit file: {str(e)}"


async def search_and_replace(
    file_path: str, pattern: str, replacement: str, replace_all: bool = False
) -> str:
    """Performs pattern search and replace in a file.

    Args:
        file_path: The absolute path to the file to modify
        pattern: The pattern to search for (supports sed syntax)
        replacement: The replacement text
        replace_all: Replace all occurrences of pattern (default False)

    Returns:
        A success message or error message
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File not found: {file_path}"

        if not os.path.isfile(file_path):
            return f"Error: Not a file: {file_path}"

        # Build the sed command
        sed_cmd = ["sed"]

        # -i option for in-place editing (macOS requires an extension)
        if os.name == "posix" and "darwin" in os.uname().sysname.lower():
            sed_cmd.extend(["-i", ""])
        else:
            sed_cmd.append("-i")

        # Add the search and replace expression
        expression = f"s/{pattern}/{replacement}/"
        if replace_all:
            expression += "g"

        sed_cmd.extend([expression, file_path])

        # Run sed command
        proc = await asyncio.create_subprocess_exec(
            *sed_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            if stderr:
                error = stderr.decode("utf-8")
                return f"Error: sed command failed: {error}"
            return f"Error: sed command failed with return code {proc.returncode}"

        return f"File {file_path} has been updated successfully"
    except Exception as e:
        return f"Error: Failed to search and edit file: {str(e)}"


async def glob(pattern: str, path: Optional[str] = None) -> str:
    """Searches for files that matches the glob pattern

    Args:
        pattern: The glob pattern to match files against
        path: The directory to search in (optional, defaults to current directory)

    Returns:
        A list of matching file paths sorted by modification time
    """
    try:
        search_path = path or os.getcwd()
        if not os.path.exists(search_path):
            return f"Error: Path not found: {search_path}"

        # Use asyncio.to_thread to run glob in a separate thread
        matching_files = await asyncio.to_thread(_glob_search, search_path, pattern)

        if not matching_files:
            return "No matching files found"

        # Sort files by modification time (most recent first)
        matching_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
        matching_files = [str(f) for f in matching_files]

        return "\n".join(matching_files)
    except Exception as e:
        return f"Error: Failed to perform glob search: {str(e)}"


def _glob_search(search_path: str, pattern: str) -> List[str]:
    """Helper function to perform glob search in a separate thread"""
    # Construct the full pattern
    if not search_path.endswith(os.sep) and not pattern.startswith(os.sep):
        full_pattern = os.path.join(search_path, pattern)
    else:
        full_pattern = search_path + pattern

    # Use glob.glob for the actual search
    return glob_module.glob(full_pattern, recursive=True)


async def grep(
    pattern: str, include: Optional[str] = None, path: Optional[str] = None
) -> List[str]:
    """Fast content search using regular expressions

    Args:
        pattern: The regular expression pattern to search for in file contents
        include: File pattern to include in the search (e.g. "*.js", "*.{ts,tsx}") (optional)
        path: The directory to search in (optional, defaults to current directory)

    Returns:
        A list of file paths with at least one match
    """
    try:
        search_path = path or os.getcwd()
        if not os.path.exists(search_path):
            return [f"Error: Path not found: {search_path}"]

        # Prepare the command arguments for running grep
        command_args = ["grep", "-l", "--include", include or "*", "-r", pattern, search_path]

        # Run grep command asynchronously
        proc = await asyncio.create_subprocess_exec(
            *command_args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await proc.communicate()

        if (
            proc.returncode != 0 and proc.returncode != 1
        ):  # 1 means no matches, which is not an error
            if stderr:
                error = stderr.decode("utf-8")
                return [f"Error: Grep command failed: {error}"]
            return [f"Error: Grep command failed with return code {proc.returncode}"]

        # Parse the output and get the list of files
        matching_files = stdout.decode("utf-8").strip().split("\n")

        # Filter out empty entries
        matching_files = [f for f in matching_files if f]

        if not matching_files:
            return []

        # Sort files by modification time (most recent first)
        matching_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

        return matching_files
    except Exception as e:
        return [f"Error: Failed to perform grep search: {str(e)}"]


async def ls(path: str, ignore: Optional[List[str]] = None) -> str:
    """Lists files and directories in a given path

    Args:
        path: The absolute path to the directory to list
        ignore: List of glob patterns to ignore (optional)

    Returns:
        A list of files and directories in the given path
    """
    try:
        if not os.path.exists(path):
            return f"Error: Path not found: {path}"

        if not os.path.isdir(path):
            return f"Error: Not a directory: {path}"

        # Get all files and directories in the given path
        items = await asyncio.to_thread(os.listdir, path)

        # Apply ignore patterns if provided
        if ignore:
            filtered_items = []
            for item in items:
                item_path = os.path.join(path, item)
                should_ignore = False

                for pattern in ignore:
                    if fnmatch.fnmatch(item, pattern) or fnmatch.fnmatch(item_path, pattern):
                        should_ignore = True
                        break

                if not should_ignore:
                    filtered_items.append(item)

            items = filtered_items

        # Construct full paths
        full_paths = [os.path.join(path, item) for item in items]

        # Sort by type (directories first) and then by name
        full_paths.sort(key=lambda p: (0 if os.path.isdir(p) else 1, p.lower()))

        return "\n".join(full_paths)
    except Exception as e:
        return f"Error: Failed to list directory: {str(e)}"


toolkit = Toolkit(
    name="file_system_toolkit",
    description="Tools to do search, list, read, write and edit operations on files.",
)
toolkit.add_tool(Tool(callable=read, read=True))
toolkit.add_tool(Tool(callable=edit, read=True, write=True))
toolkit.add_tool(Tool(callable=write, write=True))
toolkit.add_tool(Tool(callable=search_and_replace, read=True, write=True))
toolkit.add_tool(Tool(callable=glob, read=True))
toolkit.add_tool(Tool(callable=grep, read=True))
toolkit.add_tool(Tool(callable=ls, read=True))
