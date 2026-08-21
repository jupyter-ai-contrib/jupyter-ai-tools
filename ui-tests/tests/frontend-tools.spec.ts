import { expect, test } from '@jupyterlab/galata';
import type { Client } from '@modelcontextprotocol/sdk/client/index.js';

import { callTool, connectMcp } from './mcp-client';
import { CODE_CELL_1, createAndOpenNotebook } from './fixtures';

/**
 * Frontend-command tools dispatch to the JupyterLab frontend via
 * jupyterlab-commands-toolkit's ``execute_command`` (a Jupyter Event the
 * browser must service). The browser is open here, so these should succeed;
 * if no client were connected they would time out. ``execute_command`` returns
 * ``{"success": false, "error": "Command timed out..."}`` on timeout, so we
 * assert the absence of a timeout / an explicit success.
 */

let client: Client;

test.beforeAll(async () => {
  client = await connectMcp();
});

test.afterAll(async () => {
  await client?.close();
});

function assertNoTimeout(text: string): void {
  expect(text.toLowerCase()).not.toContain('timed out');
}

test.describe('frontend-command tools', () => {
  test('open_file opens a document', async ({ page, tmpPath }) => {
    const path = await createAndOpenNotebook(page, tmpPath);
    const res = await callTool(client, 'open_file', { file_path: path });
    expect(res.isError).toBe(false);
    assertNoTimeout(res.text);
    expect(res.text.toLowerCase()).toContain('success');
  });

  test('run_cell runs a single cell', async ({ page, tmpPath }) => {
    const path = await createAndOpenNotebook(page, tmpPath);
    const res = await callTool(client, 'run_cell', {
      cell_id: CODE_CELL_1,
      file_path: path
    });
    expect(res.isError).toBe(false);
    assertNoTimeout(res.text);
  });

  test('run_all_cells runs the whole notebook', async ({ page, tmpPath }) => {
    const path = await createAndOpenNotebook(page, tmpPath);
    const res = await callTool(client, 'run_all_cells', { file_path: path });
    expect(res.isError).toBe(false);
    assertNoTimeout(res.text);
  });

  test('select_cell moves the selection', async ({ page, tmpPath }) => {
    const path = await createAndOpenNotebook(page, tmpPath);
    const res = await callTool(client, 'select_cell', {
      cell_id: CODE_CELL_1,
      file_path: path
    });
    expect(res.isError).toBe(false);
    assertNoTimeout(res.text);
  });

  test('create_notebook creates a new notebook', async ({ tmpPath }) => {
    const res = await callTool(client, 'create_notebook', {
      file_path: `${tmpPath}/created_by_tool.ipynb`
    });
    expect(res.isError).toBe(false);
    assertNoTimeout(res.text);
  });
});
