import { test, expect } from './base';
import { callTool } from './mcp-client';
import { CODE_CELL_1, CODE_CELL_3, writeNotebook } from './fixtures';

// get_cell_id_from_index: cell UUID at an index (filesystem-backed)
test.describe('get_cell_id_from_index', () => {
  test('returns the id of the first cell', async ({ page, tmpPath, mcp }) => {
    const path = await writeNotebook(page, tmpPath);
    const res = await callTool(mcp, 'get_cell_id_from_index', {
      file_path: path,
      cell_index: 0
    });
    expect(res.isError, res.text).toBe(false);
    expect(res.text).toContain(CODE_CELL_1);
  });

  test('returns the id of the third cell', async ({ page, tmpPath, mcp }) => {
    const path = await writeNotebook(page, tmpPath);
    const res = await callTool(mcp, 'get_cell_id_from_index', {
      file_path: path,
      cell_index: 2
    });
    expect(res.isError, res.text).toBe(false);
    expect(res.text).toContain(CODE_CELL_3);
  });
});
