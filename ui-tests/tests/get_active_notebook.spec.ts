import { test, expect } from './base';
import { callTool } from './mcp-client';
import { buildNotebook } from './fixtures';

// get_active_notebook: path of the active notebook, from global awareness.
test.describe('get_active_notebook', () => {
  test('reports the open notebook', async ({ page, tmpPath, mcp }) => {
    await buildNotebook(page, tmpPath);
    // jupyterlab-notebook-awareness publishes the active notebook into global
    // awareness asynchronously after open, so poll until it propagates.
    await expect
      .poll(async () => (await callTool(mcp, 'get_active_notebook', {})).text, {
        timeout: 15000
      })
      .toContain('sample.ipynb');
  });
});
