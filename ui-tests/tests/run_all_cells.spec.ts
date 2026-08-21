import { test, expect } from './base';
import { callTool } from './mcp-client';
import { buildNotebook } from './fixtures';

// run_all_cells: execute every cell via the frontend command.
test.describe('run_all_cells', () => {
  test('runs the whole notebook', async ({ page, tmpPath, mcp }) => {
    const { path } = await buildNotebook(page, tmpPath);
    const res = await callTool(mcp, 'run_all_cells', { file_path: path });
    expect(res.isError, res.text).toBe(false);
    expect(res.text.toLowerCase(), res.text).not.toContain('timed out');
  });
});
