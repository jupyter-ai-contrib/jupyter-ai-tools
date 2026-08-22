import { test, expect } from './base';
import { callTool } from './mcp-client';
import { buildNotebook } from './fixtures';

// get_active_cell_id: selected cell id from the notebook room awareness
// (published by jupyterlab-notebook-awareness). RTC legs only.
test.describe('get_active_cell_id', () => {
  test('reports the selected cell', async ({ page, tmpPath, mcp }) => {
    const { path, cellIds } = await buildNotebook(page, tmpPath);
    await page.notebook.selectCells(0);
    // The selected cell id is published into the notebook room awareness
    // asynchronously, so poll until it propagates.
    await expect
      .poll(
        async () =>
          (await callTool(mcp, 'get_active_cell_id', { notebook_path: path }))
            .text,
        { timeout: 15000 }
      )
      .toContain(cellIds[0]);
  });
});
