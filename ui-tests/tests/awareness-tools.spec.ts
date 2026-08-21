import { expect, test } from '@jupyterlab/galata';
import type { Client } from '@modelcontextprotocol/sdk/client/index.js';

import { callTool, connectMcp } from './mcp-client';
import { createAndOpenNotebook } from './fixtures';

/**
 * Awareness-coupled query tools read JupyterLab collaboration awareness that is
 * populated by a live browser client. With the notebook open in the browser
 * these should report the active document/cell on RTC legs; on the ``default``
 * leg there is no awareness, so they are expected to fail.
 */

let client: Client;

test.beforeAll(async () => {
  client = await connectMcp();
});

test.afterAll(async () => {
  await client?.close();
});

test.describe('awareness tools', () => {
  test('get_active_notebook reports the open notebook', async ({
    page,
    tmpPath
  }) => {
    const path = await createAndOpenNotebook(page, tmpPath);
    const res = await callTool(client, 'get_active_notebook', {});
    expect(res.isError).toBe(false);
    expect(res.text).toContain(path);
  });

  test('get_open_documents lists the open notebook', async ({
    page,
    tmpPath
  }) => {
    await createAndOpenNotebook(page, tmpPath);
    const res = await callTool(client, 'get_open_documents', {});
    expect(res.isError).toBe(false);
    // Global awareness is shared across the session; assert it reports an open
    // notebook (all suites open a file named sample.ipynb) rather than an exact
    // per-test path.
    expect(res.text).toContain('sample.ipynb');
  });

  test('get_active_cell_id reports the selected cell', async ({
    page,
    tmpPath
  }) => {
    const path = await createAndOpenNotebook(page, tmpPath);
    await page.notebook.selectCells(0);
    const res = await callTool(client, 'get_active_cell_id', {
      notebook_path: path
    });
    expect(res.isError).toBe(false);
    expect(res.text.trim().length).toBeGreaterThan(0);
    expect(res.text.toLowerCase()).not.toContain('none');
  });
});
