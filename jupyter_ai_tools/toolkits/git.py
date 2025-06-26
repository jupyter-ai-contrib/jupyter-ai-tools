import json
import os

from jupyter_ai.tools.models import Tool, Toolkit
from jupyterlab_git.git import Git

git = Git()


# Clone a repo
async def git_clone(path: str, url: str) -> str:
    """
    Clone a Git repository from a given URL into the specified local path.

    Parameters:
        path (str): The target directory to clone into.
        url (str): The Git repository URL.

    Returns:
        str: Success or error message.
    """
    res = await git.clone(path, repo_url=url)
    if res["code"] == 0:
        return f"✅ Cloned repo into {res['path']}"
    return f"❌ Clone failed: {res.get('message', 'Unknown error')}"


# Get repo status
async def git_status(path: str) -> str:
    """
    Return the current Git status at the specified path.

    Parameters:
        path (str): The path to the Git working directory.

    Returns:
        str: A JSON-formatted string of status or an error message.
    """
    res = await git.status(path)
    if res["code"] == 0:
        return f"📋 Status:\n{json.dumps(res, indent=2)}"
    return f"❌ Git status failed: {res.get('message', 'Unknown error')}"


# Get Git log
async def git_log(path: str, history_count: int = 10) -> str:
    """
    Return the most recent Git commits for the given repository.

    Parameters:
        path (str): The path to the Git working directory.
        history_count (int): Number of commits to retrieve (default: 10).

    Returns:
        str: A JSON-formatted commit log or error message.
    """
    res = await git.log(path, history_count=history_count)
    if res["code"] == 0:
        return f"🕓 Recent commits:\n{json.dumps(res, indent=2)}"
    return f"❌ Git log failed: {res.get('message', 'Unknown error')}"


# Pull changes
async def git_pull(path: str) -> str:
    """
    Pull the latest changes from the remote into the local repository.

    Parameters:
        path (str): The path to the Git working directory.

    Returns:
        str: Success or error message.
    """
    res = await git.pull(path)
    return (
        "✅ Pulled latest changes."
        if res["code"] == 0
        else f"❌ Pull failed: {res.get('message', 'Unknown error')}"
    )


# Push changes
async def git_push(path: str, branch: str) -> str:
    """
    Push the current branch to the remote repository.

    Parameters:
        path (str): The path to the Git working directory.
        branch (str): The name of the branch to push.

    Returns:
        str: Success or error message.
    """
    res = await git.push(remote="origin", branch=branch, path=path)
    return (
        "✅ Pushed changes."
        if res["code"] == 0
        else f"❌ Push failed: {res.get('message', 'Unknown error')}"
    )


# Commit staged changes
async def git_commit(path: str, message: str) -> str:
    """
    Commit all staged changes to the repository with the given message.

    Parameters:
        path (str): The path to the Git working directory.
        message (str): The commit message.

    Returns:
        str: Success or error message.
    """
    res = await git.commit(commit_msg=message, amend=False, path=path)
    return (
        "✅ Commit successful."
        if res["code"] == 0
        else f"❌ Commit failed: {res.get('message', 'Unknown error')}"
    )


# Stage files
async def git_add(path: str, add_all: bool = True, filename: str = "") -> str:
    """
    Stage changes for commit. You can add all files or a specific file.

    Parameters:
        path (str): The path to the Git working directory.
        add_all (bool): If True, stage all changes (default: True).
        filename (str): If add_all is False, the file to stage.

    Returns:
        str: Success or error message.
    """
    if add_all:
        res = await git.add_all(path)
    elif filename:
        res = await git.add(filename=filename, path=path)
    else:
        return "❌ No file specified and add_all is False."

    files = "ALL" if add_all else filename
    return (
        f"✅ Staged: {files}"
        if res["code"] == 0
        else f"❌ Add failed: {res.get('message', 'Unknown error')}"
    )


# Get Git repo root
async def git_get_repo_root(path: str) -> str:
    """
    Return the root directory of the Git repository for the given file or directory.

    Parameters:
        path (str): Full path to a file or directory inside the Git repo.

    Returns:
        str: The path to the Git repository root or an error message.
    """
    dir_path = os.path.dirname(path)
    res = await git.show_top_level(dir_path)
    if res["code"] == 0 and res.get("path"):
        return f"📁 Repo root: {res['path']}"
    return f"❌ Not inside a Git repo. {res.get('message', '')}"


toolkit = Toolkit(
    name="git_toolkit",
    description="Tools for working with Git repositories.",
)
toolkit.add_tool(Tool(callable=git_clone, execute=True))
toolkit.add_tool(Tool(callable=git_status, read=True))
toolkit.add_tool(Tool(callable=git_log, read=True))
toolkit.add_tool(Tool(callable=git_pull, execute=True))
toolkit.add_tool(Tool(callable=git_push, execute=True))
toolkit.add_tool(Tool(callable=git_commit, execute=True))
toolkit.add_tool(Tool(callable=git_add, execute=True))
toolkit.add_tool(Tool(callable=git_get_repo_root, read=True))
