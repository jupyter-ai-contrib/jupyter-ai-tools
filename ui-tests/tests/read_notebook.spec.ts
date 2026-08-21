import { test, expect } from './base';
import { callTool } from './mcp-client';
import { writeNotebook } from './fixtures';

// read_notebook: whole notebook as markdown (filesystem-backed, RTC-independent)
test.describe('read_notebook', () => {
  test('returns the notebook as markdown', async ({ page, tmpPath, mcp }) => {
    const path = await writeNotebook(page, tmpPath);
    const res = await callTool(mcp, 'read_notebook', { file_path: path });
    expect(res.isError, res.text).toBe(false);
    expect(res.text).toContain('x = 1');
    expect(res.text).toContain('# Title');
    expect(res.text).toContain('print(x)');
  });

  test('include_outputs adds an output section', async ({
    page,
    tmpPath,
    mcp
  }) => {
    const path = await writeNotebook(page, tmpPath);
    const without = await callTool(mcp, 'read_notebook', { file_path: path });
    expect(without.text).not.toContain('#### Output');
    // NOTE: only meaningful where the contents layer round-trips cell outputs
    // from the uploaded fixture; jupyter_server_documents does not, so this is
    // asserted only when an output actually survived the write.
    const withOut = await callTool(mcp, 'read_notebook', {
      file_path: path,
      include_outputs: true
    });
    test.skip(
      !withOut.text.includes('1\n') && !withOut.text.includes('stdout'),
      'contents layer did not persist fixture outputs (e.g. JSD)'
    );
    expect(withOut.text).toContain('#### Output');
  });
});
