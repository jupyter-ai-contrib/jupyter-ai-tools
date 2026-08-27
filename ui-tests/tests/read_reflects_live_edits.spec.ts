import { test, expect } from './base';
import { callTool } from './mcp-client';
import { buildNotebook } from './fixtures';

const TRANSPORT = process.env.JAI_TRANSPORT || 'default';

// Regression test for issue #39: RTC-free, the write tools mutate the live
// JupyterLab notebook model (via jupyterlab-ai-commands) WITHOUT saving to
// disk -- saving is left to the human. The read tools must therefore return
// the live in-memory content, NOT stale disk content.
//
// This is an RTC-free-only concern: with an RTC provider the server owns a
// live YDoc that is synced to the filesystem, so disk reads are already
// current. We only exercise (and assert) the live-read guarantee on the
// `default` (RTC-free) leg; the RTC legs are left unchanged.
test.describe('read tools reflect unsaved live edits (RTC-free, #39)', () => {
  test.skip(
    TRANSPORT !== 'default',
    'RTC-free-only: with RTC the live YDoc is synced to disk (#39)'
  );

  test('read_notebook returns live content after serial writes', async ({
    page,
    tmpPath,
    mcp
  }) => {
    const { path, cellIds } = await buildNotebook(page, tmpPath);

    // Serial writes to the live model, none of which save to disk.
    const edit = await callTool(mcp, 'edit_cell', {
      file_path: path,
      cell_id: cellIds[0],
      content: 'x = 42'
    });
    expect(edit.isError, edit.text).toBe(false);

    const add = await callTool(mcp, 'add_cell', {
      file_path: path,
      content: 'appended_marker = 123'
    });
    expect(add.isError, add.text).toBe(false);

    // The read must see the unsaved edits, not the on-disk 'x = 1'.
    const res = await callTool(mcp, 'read_notebook', { file_path: path });
    expect(res.isError, res.text).toBe(false);
    expect(res.text).toContain('x = 42');
    expect(res.text).toContain('appended_marker = 123');
    expect(res.text).not.toContain('x = 1');
  });

  test('read_notebook_cells returns live content after serial writes', async ({
    page,
    tmpPath,
    mcp
  }) => {
    const { path, cellIds } = await buildNotebook(page, tmpPath);

    const edit = await callTool(mcp, 'edit_cell', {
      file_path: path,
      cell_id: cellIds[0],
      content: 'x = 42'
    });
    expect(edit.isError, edit.text).toBe(false);

    const add = await callTool(mcp, 'add_cell', {
      file_path: path,
      content: 'appended_marker = 123'
    });
    expect(add.isError, add.text).toBe(false);

    const res = await callTool(mcp, 'read_notebook_cells', {
      notebook_path: path
    });
    expect(res.isError, res.text).toBe(false);
    expect(res.text).toContain('x = 42');
    expect(res.text).toContain('appended_marker = 123');
    expect(res.text).not.toContain('x = 1');
  });
});
