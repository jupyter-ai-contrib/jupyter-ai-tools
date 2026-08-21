import { test, expect } from './base';
import { callTool } from './mcp-client';
import { buildNotebook } from './fixtures';

// read_cell: one cell as markdown, by uuid or numeric index (filesystem-backed)
test.describe('read_cell', () => {
  test('reads a cell by id', async ({ page, tmpPath, mcp }) => {
    const { path, cellIds } = await buildNotebook(page, tmpPath);
    const res = await callTool(mcp, 'read_cell', {
      file_path: path,
      cell_id: cellIds[0]
    });
    expect(res.isError, res.text).toBe(false);
    expect(res.text).toContain('x = 1');
  });

  test('reads a cell by numeric index string', async ({
    page,
    tmpPath,
    mcp
  }) => {
    const { path } = await buildNotebook(page, tmpPath);
    const res = await callTool(mcp, 'read_cell', {
      file_path: path,
      cell_id: '2'
    });
    expect(res.isError, res.text).toBe(false);
    expect(res.text).toContain('print(x)');
  });
});
