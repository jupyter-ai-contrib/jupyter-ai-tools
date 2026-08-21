import { test, expect } from './base';
import { callTool } from './mcp-client';
import { createAndOpenNotebook } from './fixtures';

// insert_cell: insert a cell at an index (YDoc-backed). RTC-free default errors.
test.describe('insert_cell', () => {
  test('inserts a cell at an index', async ({ page, tmpPath, mcp }) => {
    const path = await createAndOpenNotebook(page, tmpPath);
    const res = await callTool(mcp, 'insert_cell', {
      file_path: path,
      content: 'inserted = True',
      insert_index: 1
    });
    expect(res.isError, res.text).toBe(false);
    await expect.poll(async () => page.notebook.getCellCount()).toBe(4);
  });
});
