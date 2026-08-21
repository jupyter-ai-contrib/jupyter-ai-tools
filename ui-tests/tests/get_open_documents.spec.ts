import { test, expect } from './base';
import { callTool } from './mcp-client';
import { buildNotebook } from './fixtures';

const RTC = (process.env.JAI_TRANSPORT ?? 'default') !== 'default';

// get_open_documents: open documents (excluding .chat), from global awareness.
// There is no jupyterlab-ai-commands (RTC-free) equivalent, so this is skipped
// on the `default` leg and only exercised where a provider populates awareness.
test.describe('get_open_documents', () => {
  test('lists the open notebook', async ({ page, tmpPath, mcp }) => {
    test.skip(
      !RTC,
      'get_open_documents has no jupyterlab-ai-commands (RTC-free) equivalent'
    );
    await buildNotebook(page, tmpPath);
    const res = await callTool(mcp, 'get_open_documents', {});
    expect(res.isError, res.text).toBe(false);
    expect(res.text).toContain('sample.ipynb');
  });
});
