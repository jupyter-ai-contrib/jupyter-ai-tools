/**
 * Shared galata test extended with a worker-scoped MCP client.
 *
 * Every per-tool spec imports `test`/`expect` from here and receives an `mcp`
 * client connected to jupyter-server-mcp.
 */
import { test as galataTest, expect } from '@jupyterlab/galata';
import type { Client } from '@modelcontextprotocol/sdk/client/index.js';

import { connectMcp } from './mcp-client';

export const test = galataTest.extend<object, { mcp: Client }>({
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
