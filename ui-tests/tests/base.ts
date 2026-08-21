/**
 * Shared galata test extended with a worker-scoped MCP client.
 *
 * Every per-tool spec imports `test`/`expect` from here and receives an `mcp`
 * client connected to jupyter-server-mcp.
 *
 * We also disable galata's kernels/sessions API mocking: its route handler
 * throws on jupyter_server_documents' session/kernel responses during cell
 * execution ("Cannot read properties of null (reading 'id')"). With mocking
 * off the real APIs are used, so execution tools work under JSD too.
 */
import { test as galataTest, expect } from '@jupyterlab/galata';
import type { Client } from '@modelcontextprotocol/sdk/client/index.js';

import { connectMcp } from './mcp-client';

export const test = galataTest.extend<
  { kernels: null; sessions: null },
  { mcp: Client }
>({
  kernels: async ({}, use) => {
    await use(null);
  },
  sessions: async ({}, use) => {
    await use(null);
  },
  mcp: [
    async ({}, use) => {
      const client = await connectMcp();
      await use(client);
      await client.close();
    },
    { scope: 'worker' }
  ]
});

export { expect };
