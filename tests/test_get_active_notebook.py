"""Tests for get_active_notebook() on the RTC (global awareness) path."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jupyter_ai_tools.toolkits.notebook import get_active_notebook


@pytest.fixture(autouse=True)
def _assume_rtc_available():
    """These tests exercise get_active_notebook's RTC-mode logic (global awareness).

    Without a live server ``rtc_available()`` is False, which would route the
    call to jupyterlab-ai-commands instead; force it True so the tested code
    path runs. (RTC-free behavior is covered by the E2E suite.)
    """
    with patch("jupyter_ai_tools.toolkits.notebook.rtc_available", return_value=True):
        yield


# ── Helpers ──


def _state(username=None, **fields):
    """One client's global awareness state, as the frontend publishes it."""
    state = {"user": {"username": username}} if username else {}
    state.update(fields)
    return state


def _global_awareness(*states):
    """Patch get_global_awareness to return an awareness holding ``states``."""
    awareness = MagicMock()
    awareness.states = dict(enumerate(states, start=1))
    return patch(
        "jupyter_ai_tools.toolkits.notebook.get_global_awareness",
        new_callable=AsyncMock,
        return_value=awareness,
    )


# ── `notebookPath` published to global awareness ──


@pytest.mark.asyncio
async def test_prefers_notebook_path_while_a_chat_is_focused():
    """The case the field exists for: `current` is the chat and two notebooks are open."""
    with _global_awareness(
        _state(
            current="chat:untitled.chat",
            documents=["a.ipynb", "b.ipynb", "untitled.chat"],
            notebookPath="a.ipynb",
        )
    ):
        assert await get_active_notebook() == "a.ipynb"


@pytest.mark.asyncio
async def test_prefers_notebook_path_over_current_notebook():
    with _global_awareness(
        _state(
            current="notebook:b.ipynb",
            documents=["a.ipynb", "b.ipynb"],
            notebookPath="a.ipynb",
        )
    ):
        assert await get_active_notebook() == "a.ipynb"


@pytest.mark.asyncio
async def test_prefers_notebook_path_from_a_later_state():
    """A state carrying only the fallback fields (another tab of the same user, or
    the server's own state) must not answer ahead of the one publishing the field."""
    with _global_awareness(
        _state(current="notebook:b.ipynb", documents=["b.ipynb"]),
        _state(
            current="chat:untitled.chat",
            documents=["a.ipynb", "b.ipynb"],
            notebookPath="a.ipynb",
        ),
    ):
        assert await get_active_notebook() == "a.ipynb"


@pytest.mark.asyncio
async def test_notebook_path_is_filtered_by_username():
    with _global_awareness(
        _state("alice", notebookPath="alice.ipynb"),
        _state("bob", notebookPath="bob.ipynb"),
    ):
        assert await get_active_notebook(username="bob") == "bob.ipynb"


@pytest.mark.asyncio
async def test_does_not_answer_with_another_users_notebook_path():
    with _global_awareness(
        _state("alice", notebookPath="alice.ipynb"),
        _state("bob", current="chat:untitled.chat", documents=["a.ipynb", "b.ipynb"]),
    ):
        assert await get_active_notebook(username="bob") is None


@pytest.mark.asyncio
async def test_first_resolving_client_wins_without_username():
    """Without a username the first state that resolves wins, as before; a client
    whose focus is elsewhere can now be that state."""
    with _global_awareness(
        _state("alice", notebookPath="alice.ipynb"),
        _state("bob", current="notebook:bob.ipynb"),
    ):
        assert await get_active_notebook() == "alice.ipynb"


# ── The `current` / `documents` fallbacks, unchanged ──


@pytest.mark.asyncio
async def test_falls_back_to_current_when_notebook_path_is_absent():
    with _global_awareness(_state(current="notebook:a.ipynb", documents=["a.ipynb", "b.ipynb"])):
        assert await get_active_notebook() == "a.ipynb"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "notebook_path",
    [None, "", "notes.md", 42],
    ids=["null", "empty", "not_ipynb", "not_a_string"],
)
async def test_falls_back_to_current_when_notebook_path_is_unusable(notebook_path):
    """jupyterlab-notebook-awareness publishes null once no notebook is open; any
    other value that is not a notebook path is ignored rather than returned."""
    with _global_awareness(
        _state(
            current="notebook:a.ipynb",
            documents=["a.ipynb", "b.ipynb"],
            notebookPath=notebook_path,
        )
    ):
        assert await get_active_notebook() == "a.ipynb"


@pytest.mark.asyncio
async def test_falls_back_to_current_for_the_requested_user():
    """The fallbacks honour `username` too: alice's state could answer from either
    field, and comes first."""
    with _global_awareness(
        _state("alice", current="notebook:alice.ipynb", notebookPath="alice.ipynb"),
        _state("bob", current="notebook:bob.ipynb"),
    ):
        assert await get_active_notebook(username="bob") == "bob.ipynb"


@pytest.mark.asyncio
async def test_falls_back_to_the_only_open_notebook():
    with _global_awareness(_state(current="chat:untitled.chat", documents=["a.ipynb", "notes.md"])):
        assert await get_active_notebook() == "a.ipynb"


@pytest.mark.asyncio
async def test_cannot_choose_between_open_notebooks_without_notebook_path():
    """What remains unanswerable without a client publishing `notebookPath`: chat
    focused, two notebooks open."""
    with _global_awareness(_state(current="chat:untitled.chat", documents=["a.ipynb", "b.ipynb"])):
        assert await get_active_notebook() is None


@pytest.mark.asyncio
async def test_returns_none_without_global_awareness():
    with patch(
        "jupyter_ai_tools.toolkits.notebook.get_global_awareness",
        new_callable=AsyncMock,
        return_value=None,
    ):
        assert await get_active_notebook() is None


@pytest.mark.asyncio
async def test_returns_none_with_no_states():
    with _global_awareness():
        assert await get_active_notebook() is None
