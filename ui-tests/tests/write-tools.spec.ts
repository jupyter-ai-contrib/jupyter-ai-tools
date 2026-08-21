import { expect, test } from '@jupyterlab/galata';
import type { Client } from '@modelcontextprotocol/sdk/client/index.js';

import { callTool, connectMcp } from './mcp-client';
import { CODE_CELL_1, MD_CELL_2, createAndOpenNotebook } from './fixtures';

/**
 * Write tools mutate the live server-side YNotebook (or fall back to disk).
 * The notebook is open in the browser, so on RTC legs the browser is attached
 * to the same shared document -- we verify the mutation through the browser's
 * notebook model (the source of truth the user sees). On the ``default`` leg
 * there is no room, so these calls are expected to error.
 */

let client: Client;

test.beforeAll(async () => {
  client = await connectMcp();
});

test.afterAll(async () => {
  await client?.close();
});

test.describe('write tools (YDoc-backed)', () => {
  test('add_cell appends a cell', async ({ page, tmpPath }) => {
    const path = await createAndOpenNotebook(page, tmpPath);
    const res = await callTool(client, 'add_cell', {
      file_path: path,
      content: 'appended = True'
    });
    expect(res.isError).toBe(false);
    await expect.poll(async () => page.notebook.getCellCount()).toBe(4);
  });

  test('insert_cell inserts at an index', async ({ page, tmpPath }) => {
    const path = await createAndOpenNotebook(page, tmpPath);
    const res = await callTool(client, 'insert_cell', {
      file_path: path,
      content: 'inserted = True',
      insert_index: 1
    });
    expect(res.isError).toBe(false);
    await expect.poll(async () => page.notebook.getCellCount()).toBe(4);
  });

  test('delete_cell removes a cell by id', async ({ page, tmpPath }) => {
    const path = await createAndOpenNotebook(page, tmpPath);
    const res = await callTool(client, 'delete_cell', {
      file_path: path,
      cell_id: MD_CELL_2
    });
    expect(res.isError).toBe(false);
    await expect.poll(async () => page.notebook.getCellCount()).toBe(2);
  });

  test('edit_cell changes cell content', async ({ page, tmpPath }) => {
    const path = await createAndOpenNotebook(page, tmpPath);
    const res = await callTool(client, 'edit_cell', {
      file_path: path,
      cell_id: CODE_CELL_1,
      content: 'y = 2'
    });
    expect(res.isError).toBe(false);
    await expect
      .poll(async () => page.notebook.getCellTextInput(0))
      .toContain('y = 2');
  });
});
