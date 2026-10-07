import { test, expect } from './base';
import { callTool } from './mcp-client';

// create_notebook: create + open a new notebook. The disk-write half is
// RTC-free; the open half goes through the frontend command, so the test
// needs `page` to connect a JupyterLab web client.
test.describe('create_notebook', () => {
  test('creates a new notebook', async ({ page, tmpPath, mcp }) => {
    const res = await callTool(mcp, 'create_notebook', {
      file_path: `${tmpPath}/created_by_tool.ipynb`
    });
    expect(res.isError, res.text).toBe(false);
    expect(res.text.toLowerCase()).not.toContain('timed out');
    expect(res.text).toContain('Successfully created and opened notebook');
  });
});
