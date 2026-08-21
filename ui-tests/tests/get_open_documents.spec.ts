import { test, expect } from './base';
import { callTool } from './mcp-client';
import { buildNotebook } from './fixtures';

// get_open_documents: open documents (excluding .chat), from global awareness.
// Shared across the session, so assert on the notebook name.
test.describe('get_open_documents', () => {
  test('lists the open notebook', async ({ page, tmpPath, mcp }) => {
    await buildNotebook(page, tmpPath);
    const res = await callTool(mcp, 'get_open_documents', {});
    expect(res.isError, res.text).toBe(false);
    expect(res.text).toContain('sample.ipynb');
  });
});
