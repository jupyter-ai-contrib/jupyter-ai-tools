import { test, expect } from './base';
import { callTool } from './mcp-client';
import { buildNotebook } from './fixtures';

// select_cell: move the selection to a cell. Verify the active cell actually
// changed to the target (works with or without RTC via move-cursor commands).
test.describe('select_cell', () => {
  test('moves the selection to a cell', async ({ page, tmpPath, mcp }) => {
    const { path, cellIds } = await buildNotebook(page, tmpPath);
    await page.notebook.selectCells(0);
    const res = await callTool(mcp, 'select_cell', {
      cell_id: cellIds[2],
      file_path: path
    });
    expect(res.isError, res.text).toBe(false);
    expect(res.text.toLowerCase(), res.text).not.toContain('timed out');
    // The active cell should now be the target cell.
    const active = await callTool(mcp, 'get_active_cell_id', {
      notebook_path: path
    });
    expect(active.text).toContain(cellIds[2]);
  });
});
