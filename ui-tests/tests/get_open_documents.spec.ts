import { test, expect } from './base';
import { callTool } from './mcp-client';
import { createAndOpenNotebook } from './fixtures';

// get_open_documents: open documents (excluding .chat), from global awareness.
// RTC legs only. Global awareness is shared across the session, so assert on
// the notebook name rather than an exact per-test path.
test.describe('get_open_documents', () => {
  test('lists the open notebook', async ({ page, tmpPath, mcp }) => {
    await createAndOpenNotebook(page, tmpPath);
    const res = await callTool(mcp, 'get_open_documents', {});
    expect(res.isError, res.text).toBe(false);
    expect(res.text).toContain('sample.ipynb');
  });
});
