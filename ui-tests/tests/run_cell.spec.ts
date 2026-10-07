import { test, expect } from './base';
import { callTool } from './mcp-client';
import { buildNotebook } from './fixtures';

// run_cell: select + execute a cell via the frontend command.
test.describe('run_cell', () => {
  test('runs a single cell', async ({ page, tmpPath, mcp }) => {
    const { path, cellIds } = await buildNotebook(page, tmpPath);
    const res = await callTool(mcp, 'run_cell', {
      cell_id: cellIds[0],
      file_path: path
    });
    expect(res.isError, res.text).toBe(false);
    expect(res.text.toLowerCase(), res.text).not.toContain('timed out');
  });

  // write_outputs: false runs the cell in the notebook kernel on the server
  // and returns the outputs, without a change to the open notebook. The run
  // must see an edit that is not saved yet (with RTC, the edit is in the YDoc).
  test('runs a cell without writing the outputs', async ({
    page,
    tmpPath,
    mcp
  }) => {
    const { path, cellIds } = await buildNotebook(page, tmpPath);
    const args = { file_path: path, write_outputs: false };
    const first = await callTool(mcp, 'run_cell', { ...args, cell_id: cellIds[0] });
    expect(first.isError, first.text).toBe(false);
    const edit = await callTool(mcp, 'edit_cell', {
      file_path: path,
      cell_id: cellIds[2],
      content: 'print(x + 1)'
    });
    expect(edit.isError, edit.text).toBe(false);
    const res = await callTool(mcp, 'run_cell', { ...args, cell_id: cellIds[2] });
    expect(res.isError, res.text).toBe(false);
    expect(res.text).toContain('did not change');
    expect(res.text).toMatch(/"text":\s*"2\\n"/);

    const outputCounts = await page.evaluate(() => {
      const cells = (window as any).jupyterapp.shell.currentWidget.content
        .model.cells;
      const counts: number[] = [];
      for (let i = 0; i < cells.length; i++) {
        counts.push(cells.get(i).outputs?.length ?? 0);
      }
      return counts;
    });
    expect(outputCounts).toEqual([0, 0, 0]);
  });
});
