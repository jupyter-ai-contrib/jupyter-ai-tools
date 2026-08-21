import { test, expect } from './base';
import { callTool } from './mcp-client';
import { MD_CELL_2, writeNotebook } from './fixtures';

// read_notebook_cells: all cells, or a single cell by id (filesystem-backed)
test.describe('read_notebook_cells', () => {
  test('returns all cells', async ({ page, tmpPath, mcp }) => {
    const path = await writeNotebook(page, tmpPath);
    const res = await callTool(mcp, 'read_notebook_cells', {
      notebook_path: path
    });
    expect(res.isError, res.text).toBe(false);
    expect(res.text).toContain('x = 1');
    expect(res.text).toContain('print(x)');
  });

  test('returns a single cell by id', async ({ page, tmpPath, mcp }) => {
    const path = await writeNotebook(page, tmpPath);
    const res = await callTool(mcp, 'read_notebook_cells', {
      notebook_path: path,
      specific_cell_id: MD_CELL_2
    });
    expect(res.isError, res.text).toBe(false);
    expect(res.text).toContain('# Title');
  });
});
