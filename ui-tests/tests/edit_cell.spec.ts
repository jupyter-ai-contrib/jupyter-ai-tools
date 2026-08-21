import { test, expect } from './base';
import { callTool } from './mcp-client';
import { buildNotebook } from './fixtures';

// edit_cell: change a cell's content (YDoc-backed). RTC-free default errors.
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
});
