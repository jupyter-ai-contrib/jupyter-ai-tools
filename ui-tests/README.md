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
nox -s "e2e(env='jcollab')"    # jupyter_collaboration + jupyterlab-notebook-awareness
nox -s "e2e(env='jsd')"        # jupyter_server_documents + jupyterlab-notebook-awareness
```

The RTC legs also install `jupyterlab-notebook-awareness`, the frontend
extension that publishes the active cell id / notebook path into collaboration
awareness. The awareness tools (`get_active_cell_id`, `select_cell`,
`run_cell`) read those fields, so it is required for them to work.

CI runs all three as the **E2E** workflow, rendering as
`E2E / E2E tests (default)`, `E2E tests (+JCollab)`, `E2E tests (+JSD)`.

Local, single leg without nox:

```bash
cd ui-tests && jlpm install && jlpm playwright install chromium
JAI_TRANSPORT=default jlpm playwright test
```

## Structure

One spec file per tool (`read_notebook.spec.ts`, `add_cell.spec.ts`,
`get_active_notebook.spec.ts`, `run_cell.spec.ts`, ...). Shared setup lives in
`tests/base.ts` (a galata `test` extended with a worker-scoped `mcp` client),
`tests/mcp-client.ts`, and `tests/fixtures.ts`.

The notebook fixture (`buildNotebook`) is built *through JupyterLab* via galata
(`createNew` + `addCell`), not by uploading a hand-crafted .ipynb JSON, and the
real cell ids are read back from the live model. `tests/base.ts` also disables
galata's kernels/sessions API mocking: that route handler throws on
jupyter_server_documents' session/kernel responses during cell execution
(`Cannot read properties of null (reading 'id')`), which previously failed
`run_cell`/`run_all_cells` on `+JSD`. With mocking off, both RTC legs are green.

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

Locally validated: `+JCollab` and `+JSD` are green (20/20); `default` is the
RTC-free baseline (10 pass / 10 fail: read tools + `open_file` + `run_all_cells`
+ `create_notebook` pass; write/awareness tools + `run_cell` + `select_cell`
fail because there is no live YDoc room / awareness without a provider).

## Resolved: JSD `run_cell` / `run_all_cells`

These previously failed on `+JSD` with an uncaught browser
`TypeError: Cannot read properties of null (reading 'id')` during cell
execution. Root cause was **galata's own kernels/sessions API mocking**: its
route handler chokes on jupyter_server_documents' session/kernel responses (null
body) once execution triggers session activity. It was not the tool, not the
notebook construction, and not `jupyterlab-notebook-awareness`. Disabling that
mocking in `tests/base.ts` makes the real APIs flow and both RTC legs pass.
