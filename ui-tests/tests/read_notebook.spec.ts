import { test, expect } from './base';
import { callTool } from './mcp-client';
import { buildNotebook } from './fixtures';

// read_notebook: whole notebook as markdown (filesystem-backed, RTC-independent)
test.describe('read_notebook', () => {
  test('returns the notebook as markdown', async ({ page, tmpPath, mcp }) => {
    const { path } = await buildNotebook(page, tmpPath);
    const res = await callTool(mcp, 'read_notebook', { file_path: path });
    expect(res.isError, res.text).toBe(false);
    expect(res.text).toContain('x = 1');
    expect(res.text).toContain('# Title');
    expect(res.text).toContain('print(x)');
  });

  test('includes each cell id in document order', async ({
    page,
    tmpPath,
    mcp
  }) => {
    const { path, cellIds } = await buildNotebook(page, tmpPath);
    const res = await callTool(mcp, 'read_notebook', { file_path: path });
    expect(res.isError, res.text).toBe(false);
    // Every real cell id appears in the markdown...
    for (const id of cellIds) {
      expect(res.text).toContain(id);
    }
    // ...and in the same order as the notebook's cells.
    const positions = cellIds.map(id => res.text.indexOf(id));
    const sorted = [...positions].sort((a, b) => a - b);
    expect(positions).toEqual(sorted);
  });
});
