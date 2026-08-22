import { test, expect } from './base';
import { callTool } from './mcp-client';
import { buildNotebook } from './fixtures';

// edit_cell: change a cell's content and/or type (YDoc-backed under RTC;
// set-cell-content + change-cell-to-* commands when RTC-free).
test.describe('edit_cell', () => {
  test('changes a cell content', async ({ page, tmpPath, mcp }) => {
    const { path, cellIds } = await buildNotebook(page, tmpPath);
    const res = await callTool(mcp, 'edit_cell', {
      file_path: path,
      cell_id: cellIds[0],
      content: 'y = 2'
    });
    expect(res.isError, res.text).toBe(false);
    await expect
      .poll(async () => page.notebook.getCellTextInput(0))
      .toContain('y = 2');
  });

  test('changes a cell type', async ({ page, tmpPath, mcp }) => {
    const { path, cellIds } = await buildNotebook(page, tmpPath);
    const res = await callTool(mcp, 'edit_cell', {
      file_path: path,
      cell_id: cellIds[0],
      cell_type: 'markdown'
    });
    expect(res.isError, res.text).toBe(false);
    await expect
      .poll(async () => page.notebook.getCellType(0))
      .toBe('markdown');
  });
});
