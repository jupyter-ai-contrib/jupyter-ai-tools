"""E2E test matrix for the jupyter-ai default toolkit.

Mirrors the jupyter-ai-acp-client PR #178 transport matrix. These tools only
run correctly *inside the Jupyter Server process*, so the suite boots a real
JupyterLab (browser open, so the jupyterlab-commands-toolkit frontend and RTC
awareness are live), serves the tools via ``jupyter-server-mcp``, and drives
them from Playwright acting as an MCP client. This also exercises
``jupyter-server-mcp`` and ``jupyterlab-commands-toolkit`` end to end.

The transport is selected purely by which collaboration package is installed:

* ``default``  -- no RTC provider (the RTC-free / decoupling target).
* ``jcollab``  -- ``jupyter_collaboration`` installed; live YNotebook rooms.
* ``jsd``      -- ``jupyter_server_documents`` installed; live YNotebook rooms.

Run one leg locally, e.g.::

    nox -s "e2e(env='default')"
    nox -s "e2e(env='jcollab')"
    nox -s "e2e(env='jsd')"

Failing tests are expected and informative: they document where a tool is
coupled to RTC awareness or to the JupyterLab frontend.
"""

import os

import nox

# Prefer uv for fast, isolated env creation; fall back to virtualenv.
nox.options.default_venv_backend = "uv|virtualenv"

# env name -> extra packages that provide the transport.
# Floors mirror the validated jupyter-ai-router / acp-client RTC matrix.
_ENVS = {
    "default": [],
    "jcollab": ["jupyter_collaboration>=4,<5"],
    "jsd": ["jupyter_server_documents"],
}


@nox.session(python="3.10")
@nox.parametrize("env", list(_ENVS))
def e2e(session: nox.Session, env: str) -> None:
    """Run the tool E2E suite against one transport."""
    # The package under test: prebuilt wheel from a CI build job, else source.
    target = os.environ.get("E2E_WHEEL") or "."
    # A local jupyterlab-commands-toolkit checkout may override the PyPI dist.
    jlct = os.environ.get("JLCT_PATH", "jupyterlab-commands-toolkit")

    session.install(
        "jupyterlab>=4.0.0,<5",
        "jupyter-server-mcp",
        jlct,
        target,
        *_ENVS[env],
    )

    with session.chdir("ui-tests"):
        session.run("jlpm", "install", external=True)
        session.run("jlpm", "playwright", "install", "chromium", external=True)
        session.run(
            "jlpm",
            "playwright",
            "test",
            *session.posargs,
            external=True,
            env={"JAI_TRANSPORT": env},
        )
