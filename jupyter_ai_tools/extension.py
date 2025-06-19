from jupyter_server_ai_tools.models import Tool

from . import git_tools, ynotebook_tools


def get_notebook_tools():
    return {
        Tool(callable=ynotebook_tools.delete_cell, delete=True),
        Tool(callable=ynotebook_tools.add_cell, write=True),
        Tool(callable=ynotebook_tools.write_to_cell, read=True, write=True),
        Tool(callable=ynotebook_tools.get_max_cell_index, read=True),
        Tool(callable=ynotebook_tools.read_cell, read=True),
        Tool(callable=ynotebook_tools.read_notebook, read=True)
    }

def get_git_tools():
    return {
        Tool(callable=git_tools.git_clone, write=True),
        Tool(callable=git_tools.git_status, read=True),
        Tool(callable=git_tools.git_log, read=True),
        Tool(callable=git_tools.git_pull, read=True, write=True),
        Tool(callable=git_tools.git_push, read=True, write=True),
        Tool(callable=git_tools.git_commit, write=True),
        Tool(callable=git_tools.git_add, write=True),
        Tool(callable=git_tools.git_get_repo_root, read=True)
    }
