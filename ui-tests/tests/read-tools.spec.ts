import { expect, test } from '@jupyterlab/galata';
import type { Client } from '@modelcontextprotocol/sdk/client/index.js';

import { callTool, connectMcp, listToolNames } from './mcp-client';
import { CODE_CELL_1, MD_CELL_2, writeNotebook } from './fixtures';

let client: Client;

test.beforeAll(async () => {
  client = await connectMcp();
});

test.afterAll(async () => {
  await client?.close();
});

test.describe('read tools (filesystem-backed, expected RTC-independent)', () => {
  test('MCP server exposes all 16 default-toolkit tools', async () => {
    const names = await listToolNames(client);
    for (const t of [
      'read_notebook',
      'read_notebook_cells',
      'read_cell',
      'add_cell',
      'insert_cell',
      'delete_cell',
      'edit_cell',
      'select_cell',
      'get_cell_id_from_index',
      'get_active_notebook',
      'get_active_cell_id',
      'get_open_documents',
      'create_notebook',
      'open_file',
      'run_cell',
      'run_all_cells'
    ]) {
      expect(names).toContain(t);
    }
  });

  test('read_notebook', async ({ page, tmpPath }) => {
    const path = await writeNotebook(page, tmpPath);
    const res = await callTool(client, 'read_notebook', { file_path: path });
    expect(res.isError).toBe(false);
    expect(res.text).toContain('x = 1');
    expect(res.text).toContain('# Title');
    expect(res.text).toContain('print(x)');
  });

  test('read_notebook_cells', async ({ page, tmpPath }) => {
    const path = await writeNotebook(page, tmpPath);
    const all = await callTool(client, 'read_notebook_cells', {
      notebook_path: path
    });
    expect(all.isError).toBe(false);
    expect(all.text).toContain('x = 1');

    const one = await callTool(client, 'read_notebook_cells', {
      notebook_path: path,
      specific_cell_id: MD_CELL_2
    });
    expect(one.isError).toBe(false);
    expect(one.text).toContain('# Title');
  });

  test('read_cell', async ({ page, tmpPath }) => {
    const path = await writeNotebook(page, tmpPath);
    const byId = await callTool(client, 'read_cell', {
      file_path: path,
      cell_id: CODE_CELL_1
    });
    expect(byId.isError).toBe(false);
    expect(byId.text).toContain('x = 1');

    const byIndex = await callTool(client, 'read_cell', {
      file_path: path,
      cell_id: '2'
    });
    expect(byIndex.isError).toBe(false);
    expect(byIndex.text).toContain('print(x)');
  });

  test('get_cell_id_from_index', async ({ page, tmpPath }) => {
    const path = await writeNotebook(page, tmpPath);
    const res = await callTool(client, 'get_cell_id_from_index', {
      file_path: path,
      cell_index: 0
    });
    expect(res.isError).toBe(false);
    expect(res.text).toContain(CODE_CELL_1);
  });
});
