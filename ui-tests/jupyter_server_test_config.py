"""Test server config for the tool E2E suite.

Boots JupyterLab (galata defaults) plus ``jupyter-server-mcp`` serving the
jupyter-ai default toolkit. Ports come from the ``jlpm start`` CLI args set in
``playwright.config.js`` (``--ServerApp.port`` / ``--MCPExtensionApp.mcp_port``).

The collaboration provider is selected by ``JAI_TRANSPORT`` (set by the nox
session): ``default`` (none), ``jcollab`` (jupyter_collaboration), or ``jsd``
(jupyter_server_documents). We enable the matching server extensions explicitly
so the environment is unambiguous; ``jupyter_server_fileid`` is required by both
RTC providers (it supplies ``settings['file_id_manager']``).
"""

import os

from jupyterlab.galata import configure_jupyter_server

configure_jupyter_server(c)  # noqa: F821

transport = os.environ.get("JAI_TRANSPORT", "default")

# Explicitly enable the transport's server extensions for the current matrix
# branch. (The RTC-free `default` leg enables nothing extra.) Use the traitlets
# LazyConfigValue.update() idiom -- do NOT dict()/reassign, which raises.
if transport == "jcollab":
    c.ServerApp.jpserver_extensions.update(  # noqa: F821
        {
            "jupyter_collaboration": True,
            "jupyter_server_ydoc": True,
            "jupyter_server_fileid": True,
        }
    )
elif transport == "jsd":
    c.ServerApp.jpserver_extensions.update(  # noqa: F821
        {"jupyter_server_documents": True, "jupyter_server_fileid": True}
    )

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
