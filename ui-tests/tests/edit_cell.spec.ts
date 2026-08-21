import { test, expect } from './base';
import { callTool } from './mcp-client';
import { CODE_CELL_1, createAndOpenNotebook } from './fixtures';

// edit_cell: change a cell's content (YDoc-backed). RTC-free default errors.
test.describe('edit_cell', () => {
  test('changes a cell content', async ({ page, tmpPath, mcp }) => {
    const path = await createAndOpenNotebook(page, tmpPath);
    const res = await callTool(mcp, 'edit_cell', {
      file_path: path,
      cell_id: CODE_CELL_1,
      content: 'y = 2'
    });
    expect(res.isError, res.text).toBe(false);
    await expect
      .poll(async () => page.notebook.getCellTextInput(0))
      .toContain('y = 2');
  });
});
