import { test, expect } from './base';
import { callTool } from './mcp-client';
import { buildNotebook } from './fixtures';

// delete_cell: remove a cell by id (YDoc-backed). RTC-free default errors.
test.describe('delete_cell', () => {
  test('removes a cell by id', async ({ page, tmpPath, mcp }) => {
    const { path, cellIds } = await buildNotebook(page, tmpPath);
    const res = await callTool(mcp, 'delete_cell', {
      file_path: path,
      cell_id: cellIds[1]
    });
    expect(res.isError, res.text).toBe(false);
    await expect.poll(async () => page.notebook.getCellCount()).toBe(2);
  });
});
