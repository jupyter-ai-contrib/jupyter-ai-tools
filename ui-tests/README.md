# Tool E2E suite (`ui-tests`)

End-to-end tests for the tools jupyter-ai registers on the
`jupyter_server_mcp.tools` entry point (`DEFAULT_JUPYTER_SERVER_MCP_TOOLS`).

These tools only run correctly **inside the Jupyter Server process**, and
several of them drive the JupyterLab **frontend** (via
`jupyterlab-commands-toolkit`) or read **RTC awareness**. So the suite:

1. Boots a real JupyterLab (galata) with `jupyter-server-mcp` serving the
   default toolkit (see `jupyter_server_test_config.py`).
2. Opens the notebook **in the browser** — this creates the live YDoc room (on
   RTC legs), populates awareness, and lets the commands-toolkit frontend
   service `execute_command`.
3. Connects to the MCP server over streamable HTTP (the MCP TypeScript SDK) and
   **calls each tool exactly as an AI persona would**, then asserts the result
   (verifying mutations through the browser's notebook model).

This intentionally also exercises `jupyter-server-mcp` and
`jupyterlab-commands-toolkit` end to end.

## Transport matrix

Run via `nox` from the repo root (each leg is an isolated venv; the transport
is chosen purely by which package is installed):

```bash
nox -s "e2e(env='default')"    # no RTC provider (RTC-free target)
nox -s "e2e(env='jcollab')"    # jupyter_collaboration
nox -s "e2e(env='jsd')"        # jupyter_server_documents
```

CI runs all three as the **E2E** workflow, rendering as
`E2E / E2E tests (default)`, `E2E tests (+JCollab)`, `E2E tests (+JSD)`.

Local, single leg without nox:

```bash
cd ui-tests && jlpm install && jlpm playwright install chromium
JAI_TRANSPORT=default jlpm playwright test
```

## Expected results (failing tests are the baseline signal)

* **Read tools** (`read_notebook`, `read_notebook_cells`, `read_cell`,
  `get_cell_id_from_index`): pass on all legs (filesystem-backed, RTC-free).
* **Write tools** (`add_cell`, `insert_cell`, `delete_cell`, `edit_cell`): pass
  on RTC legs (live YDoc); on `default` they error in `utils.get_room`
  (`KeyError: 'jupyter_server_ydoc'`) — the RTC-free path is currently broken.
* **Awareness tools** (`get_active_notebook`, `get_open_documents`,
  `get_active_cell_id`): depend on collaboration awareness; expected to pass
  only where a provider populates it.
* **Frontend-command tools** (`open_file`, `run_cell`, `run_all_cells`,
  `select_cell`, `create_notebook`): dispatch to the browser via
  `execute_command`; they need a live frontend (which this suite provides).

Locally validated (default leg): all read tools green; write tools correctly
error on the RTC-free path.
