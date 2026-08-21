"""Test server config for the tool E2E suite.

Boots JupyterLab (galata defaults) plus ``jupyter-server-mcp`` serving the
jupyter-ai default toolkit. Ports come from the ``jlpm start`` CLI args set in
``playwright.config.js`` (``--ServerApp.port`` / ``--MCPExtensionApp.mcp_port``).
"""

from jupyterlab.galata import configure_jupyter_server

configure_jupyter_server(c)  # noqa: F821

# --- jupyter-server-mcp: serve the jupyter-ai default toolkit -------------
# We register the tool list explicitly (instead of relying on the jupyter-ai
# entry point) so the suite does not need jupyter-ai installed. This list is
# jupyter_ai.DEFAULT_JUPYTER_SERVER_MCP_TOOLS verbatim.
c.MCPExtensionApp.use_tool_discovery = False  # noqa: F821
c.MCPExtensionApp.mcp_tools = [  # noqa: F821
    # notebook toolkit
    "jupyter_ai_tools.toolkits.notebook:read_notebook",
    "jupyter_ai_tools.toolkits.notebook:read_notebook_cells",
    "jupyter_ai_tools.toolkits.notebook:read_cell",
    "jupyter_ai_tools.toolkits.notebook:add_cell",
    "jupyter_ai_tools.toolkits.notebook:insert_cell",
    "jupyter_ai_tools.toolkits.notebook:delete_cell",
    "jupyter_ai_tools.toolkits.notebook:edit_cell",
    "jupyter_ai_tools.toolkits.notebook:select_cell",
    "jupyter_ai_tools.toolkits.notebook:get_cell_id_from_index",
    "jupyter_ai_tools.toolkits.notebook:get_active_notebook",
    "jupyter_ai_tools.toolkits.notebook:get_active_cell_id",
    "jupyter_ai_tools.toolkits.notebook:get_open_documents",
    "jupyter_ai_tools.toolkits.notebook:create_notebook",
    # jupyterlab toolkit
    "jupyter_ai_tools.toolkits.jupyterlab:open_file",
    "jupyter_ai_tools.toolkits.jupyterlab:run_cell",
    "jupyter_ai_tools.toolkits.jupyterlab:run_all_cells",
]
