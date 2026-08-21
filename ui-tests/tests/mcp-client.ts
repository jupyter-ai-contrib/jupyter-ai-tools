/**
 * Minimal MCP client for the E2E suite.
 *
 * jupyter-server-mcp serves FastMCP over streamable HTTP on its own port
 * (no Jupyter auth). We connect with the official MCP TypeScript SDK and call
 * the jupyter-ai tools exactly as an AI persona would.
 */
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';

export interface ToolResult {
  raw: any;
  isError: boolean;
  text: string;
  structured: any;
}

export function mcpUrl(): URL {
  return new URL(process.env.JAI_MCP_URL || 'http://127.0.0.1:3101/mcp');
}

export async function connectMcp(): Promise<Client> {
  const client = new Client({ name: 'jai-tools-e2e', version: '0.0.0' });
  const transport = new StreamableHTTPClientTransport(mcpUrl());
  await client.connect(transport);
  return client;
}

export async function listToolNames(client: Client): Promise<string[]> {
  const res: any = await client.listTools();
  return (res.tools || []).map((t: any) => t.name);
}

export async function callTool(
  client: Client,
  name: string,
  args: Record<string, any> = {}
): Promise<ToolResult> {
  const res: any = await client.callTool({ name, arguments: args });
  const content = Array.isArray(res.content) ? res.content : [];
  const text = content
    .filter((c: any) => c.type === 'text')
    .map((c: any) => c.text)
    .join('\n');
  return {
    raw: res,
    isError: Boolean(res.isError),
    text,
    structured: res.structuredContent
  };
}
