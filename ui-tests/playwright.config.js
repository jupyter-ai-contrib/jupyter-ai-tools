const baseConfig = require('@jupyterlab/galata/lib/playwright-config');

// Pin a random port pair once into the environment (Playwright re-requires the
// config in every worker, so the values must be stable across workers).
if (!process.env.JAI_TEST_PORT) {
  process.env.JAI_TEST_PORT = String(8989 + Math.floor(Math.random() * 800));
}
const PORT = Number(process.env.JAI_TEST_PORT);
const MCP_PORT = PORT + 100;

// The MCP server (jupyter-server-mcp) listens on its own port with no auth.
// Specs read this to connect an MCP client.
process.env.JAI_MCP_URL = `http://127.0.0.1:${MCP_PORT}/mcp`;

module.exports = {
  ...baseConfig,
  timeout: 90 * 1000,
  // Run serially: all specs share one JupyterLab server + notebook workspace.
  workers: 1,
  use: {
    ...(baseConfig.use || {}),
    baseURL: `http://localhost:${PORT}`
  },
  webServer: {
    command: `jlpm start --ServerApp.port=${PORT} --MCPExtensionApp.mcp_port=${MCP_PORT}`,
    url: `http://localhost:${PORT}/lab`,
    timeout: 120 * 1000,
    reuseExistingServer: false
  }
};
