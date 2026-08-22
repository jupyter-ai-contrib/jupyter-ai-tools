import { test, expect } from './base';
import { callTool } from './mcp-client';
import { buildNotebook } from './fixtures';

// Count open main-area document widgets whose context path matches `path`.
async function openTabCount(page: any, path: string): Promise<number> {
  return page.evaluate((p: string) => {
    const app = (window as any).jupyterapp;
    let n = 0;
    for (const w of app.shell.widgets('main')) {
      const ctx = (w as any).context;
      if (ctx && ctx.path === p) {
        n++;
      }
    }
    return n;
  }, path);
}

// Path of the currently-active main-area widget (or null).
async function currentPath(page: any): Promise<string | null> {
  return page.evaluate(() => {
    const app = (window as any).jupyterapp;
    const w: any = app.shell.currentWidget;
    return w?.context?.path ?? null;
  });
}

// open_file: open a document in the main area via the frontend command.
test.describe('open_file', () => {
  test('opens a document', async ({ page, tmpPath, mcp }) => {
    const { path } = await buildNotebook(page, tmpPath);
    const res = await callTool(mcp, 'open_file', { file_path: path });
    expect(res.isError, res.text).toBe(false);
    expect(res.text.toLowerCase()).not.toContain('timed out');
    expect(res.text.toLowerCase()).toContain('success');
  });

  // docmanager:open calls openOrReveal, so opening an already-open file should
  // reveal its existing tab rather than error or spawn a duplicate.
  test('is idempotent when called twice', async ({ page, tmpPath, mcp }) => {
    const { path } = await buildNotebook(page, tmpPath);

    const first = await callTool(mcp, 'open_file', { file_path: path });
    expect(first.isError, first.text).toBe(false);
    expect(await openTabCount(page, path)).toBe(1);

    const second = await callTool(mcp, 'open_file', { file_path: path });
    expect(second.isError, second.text).toBe(false);
    // Still one tab -- no duplicate, no error.
    expect(await openTabCount(page, path)).toBe(1);
    expect(await currentPath(page)).toBe(path);
  });

  // Opening nb1, then nb2, then nb1 again should reveal (focus) nb1's existing
  // tab rather than open a second copy.
  test('reveals an already-open document', async ({ page, tmpPath, mcp }) => {
    // Build nb1 the proper way (created + opened).
    const { path: nb1 } = await buildNotebook(page, tmpPath, 'nb1.ipynb');

    // Create nb2 on disk via the contents API (a second galata createNew hangs
    // on the kernel dialog). Include a kernelspec so opening it does not pop a
    // "Select Kernel" modal -- that dialog would steal focus and mask the
    // reveal behavior under test.
    const nb2 = `${tmpPath}/nb2.ipynb`;
    await page.evaluate(async (p: string) => {
      const app = (window as any).jupyterapp;
      await app.serviceManager.contents.save(p, {
        type: 'notebook',
        format: 'json',
        content: {
          cells: [],
          metadata: { kernelspec: { name: 'python3', display_name: 'Python 3' } },
          nbformat: 4,
          nbformat_minor: 5
        }
      });
    }, nb2);
    const openNb2 = await callTool(mcp, 'open_file', { file_path: nb2 });
    expect(openNb2.isError, openNb2.text).toBe(false);
    await expect.poll(() => currentPath(page)).toBe(nb2);

    // Re-open nb1 while nb2 is the active tab: docmanager:open -> openOrReveal
    // reuses the existing widget (no duplicate) and brings it to the front.
    const res = await callTool(mcp, 'open_file', { file_path: nb1 });
    expect(res.isError, res.text).toBe(false);
    await expect.poll(() => currentPath(page)).toBe(nb1);
    expect(await openTabCount(page, nb1)).toBe(1);
    expect(await openTabCount(page, nb2)).toBe(1);
  });
});
