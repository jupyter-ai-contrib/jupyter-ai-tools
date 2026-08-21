import { test, expect } from './base';
import { callTool } from './mcp-client';
import { CODE_CELL_3, createAndOpenNotebook } from './fixtures';

// select_cell: move the selection to a cell (awareness + frontend command).
// Needs a current active cell, so select one first.
test.describe('select_cell', () => {
  test('moves the selection to a cell', async ({ page, tmpPath, mcp }) => {
    const path = await createAndOpenNotebook(page, tmpPath);
    await page.notebook.selectCells(0);
    const res = await callTool(mcp, 'select_cell', {
      cell_id: CODE_CELL_3,
      file_path: path
    });
    expect(res.isError, res.text).toBe(false);
    expect(res.text.toLowerCase(), res.text).not.toContain('timed out');
  });
});
