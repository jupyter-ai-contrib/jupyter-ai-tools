import { test, expect } from './base';
import { callTool } from './mcp-client';
import { buildNotebook } from './fixtures';

// open_file: open a document in the main area via the frontend command.
test.describe('open_file', () => {
  test('opens a document', async ({ page, tmpPath, mcp }) => {
    const { path } = await buildNotebook(page, tmpPath);
    const res = await callTool(mcp, 'open_file', { file_path: path });
    expect(res.isError, res.text).toBe(false);
    expect(res.text.toLowerCase()).not.toContain('timed out');
    expect(res.text.toLowerCase()).toContain('success');
  });
});
