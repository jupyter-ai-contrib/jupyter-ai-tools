import { test, expect } from './base';
import { callTool } from './mcp-client';
import { CODE_CELL_1, createAndOpenNotebook } from './fixtures';

// run_cell: select + execute a cell via the frontend command.
test.describe('run_cell', () => {
  test('runs a single cell', async ({ page, tmpPath, mcp }) => {
    const path = await createAndOpenNotebook(page, tmpPath);
    const res = await callTool(mcp, 'run_cell', {
      cell_id: CODE_CELL_1,
      file_path: path
    });
    expect(res.isError, res.text).toBe(false);
    expect(res.text.toLowerCase(), res.text).not.toContain('timed out');
  });
});
