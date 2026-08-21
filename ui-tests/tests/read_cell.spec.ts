import { test, expect } from './base';
import { callTool } from './mcp-client';
import { CODE_CELL_1, writeNotebook } from './fixtures';

// read_cell: one cell as markdown, by uuid or numeric index (filesystem-backed)
test.describe('read_cell', () => {
  test('reads a cell by id', async ({ page, tmpPath, mcp }) => {
    const path = await writeNotebook(page, tmpPath);
    const res = await callTool(mcp, 'read_cell', {
      file_path: path,
      cell_id: CODE_CELL_1
    });
    expect(res.isError, res.text).toBe(false);
    expect(res.text).toContain('x = 1');
  });

  test('reads a cell by numeric index string', async ({
    page,
    tmpPath,
    mcp
  }) => {
    const path = await writeNotebook(page, tmpPath);
    const res = await callTool(mcp, 'read_cell', {
      file_path: path,
      cell_id: '2'
    });
    expect(res.isError, res.text).toBe(false);
    expect(res.text).toContain('print(x)');
  });
});
