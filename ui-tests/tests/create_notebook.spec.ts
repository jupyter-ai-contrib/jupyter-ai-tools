import { test, expect } from './base';
import { callTool } from './mcp-client';

// create_notebook: create + open a new notebook. The disk-write half is
// RTC-free; the open half goes through the frontend command.
test.describe('create_notebook', () => {
  test('creates a new notebook', async ({ tmpPath, mcp }) => {
    const res = await callTool(mcp, 'create_notebook', {
      file_path: `${tmpPath}/created_by_tool.ipynb`
    });
    expect(res.isError, res.text).toBe(false);
    expect(res.text.toLowerCase()).not.toContain('timed out');
  });
});
