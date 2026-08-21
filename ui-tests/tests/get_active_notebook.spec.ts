import { test, expect } from './base';
import { callTool } from './mcp-client';
import { buildNotebook } from './fixtures';

// get_active_notebook: path of the active notebook, from global awareness.
test.describe('get_active_notebook', () => {
  test('reports the open notebook', async ({ page, tmpPath, mcp }) => {
    await buildNotebook(page, tmpPath);
    const res = await callTool(mcp, 'get_active_notebook', {});
    expect(res.isError, res.text).toBe(false);
    expect(res.text).toContain('sample.ipynb');
  });
});
