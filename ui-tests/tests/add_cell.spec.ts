import { test, expect } from './base';
import { callTool } from './mcp-client';
import { buildNotebook } from './fixtures';

// add_cell: add a cell above/below a reference cell (YDoc-backed). Verified via
// the browser notebook model. On the RTC-free default leg this errors.
test.describe('add_cell', () => {
  test('appends a cell at the end', async ({ page, tmpPath, mcp }) => {
    const { path } = await buildNotebook(page, tmpPath);
    const res = await callTool(mcp, 'add_cell', {
      file_path: path,
      content: 'appended = True'
    });
    expect(res.isError, res.text).toBe(false);
    await expect.poll(async () => page.notebook.getCellCount()).toBe(4);
  });

  test('adds a markdown cell above a reference cell', async ({
    page,
    tmpPath,
    mcp
  }) => {
    const { path, cellIds } = await buildNotebook(page, tmpPath);
    const res = await callTool(mcp, 'add_cell', {
      file_path: path,
      content: '## note',
      cell_id: cellIds[0],
      add_above: true,
      cell_type: 'markdown'
    });
    expect(res.isError, res.text).toBe(false);
    await expect.poll(async () => page.notebook.getCellCount()).toBe(4);
  });

  test('saves the new cell to disk', async ({ page, tmpPath, mcp }) => {
    const { path } = await buildNotebook(page, tmpPath);
    const res = await callTool(mcp, 'add_cell', {
      file_path: path,
      content: 'saved = True'
    });
    expect(res.isError, res.text).toBe(false);
    await expect
      .poll(async () => {
        const response = await page.request.get(
          `/api/contents/${path}?content=1`
        );
        const model = await response.json();
        return model.content.cells.map((cell: any) => cell.source);
      })
      .toContain('saved = True');
  });
});
