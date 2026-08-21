import { test, expect } from './base';
import { callTool } from './mcp-client';
import { createAndOpenNotebook } from './fixtures';

// get_active_notebook: path of the active notebook, from global awareness.
// Requires a collaboration provider populating awareness (RTC legs).
test.describe('get_active_notebook', () => {
  test('reports the open notebook', async ({ page, tmpPath, mcp }) => {
    await createAndOpenNotebook(page, tmpPath);
    const res = await callTool(mcp, 'get_active_notebook', {});
    expect(res.isError, res.text).toBe(false);
    expect(res.text).toContain('sample.ipynb');
  });
});
