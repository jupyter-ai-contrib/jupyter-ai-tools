/**
 * Shared fixtures for the tool E2E suite.
 */
import type { IJupyterLabPageFixture } from '@jupyterlab/galata';

export const CODE_CELL_1 = 'cell-code-0001';
export const MD_CELL_2 = 'cell-md-0002';
export const CODE_CELL_3 = 'cell-code-0003';

// Set by the nox session; 'default' when run standalone.
export const TRANSPORT = process.env.JAI_TRANSPORT || 'default';
export const IS_RTC = TRANSPORT === 'jcollab' || TRANSPORT === 'jsd';

/** A minimal but realistic nbformat 4.5 notebook with three cells. */
export function sampleNotebook(): any {
  return {
    cells: [
      {
        cell_type: 'code',
        id: CODE_CELL_1,
        metadata: {},
        execution_count: null,
        outputs: [],
        source: 'x = 1'
      },
      {
        cell_type: 'markdown',
        id: MD_CELL_2,
        metadata: {},
        source: '# Title'
      },
      {
        cell_type: 'code',
        id: CODE_CELL_3,
        metadata: {},
        execution_count: null,
        outputs: [],
        source: 'print(x)'
      }
    ],
    metadata: {
      kernelspec: {
        display_name: 'Python 3 (ipykernel)',
        language: 'python',
        name: 'python3'
      },
      language_info: { name: 'python', version: '3.10' }
    },
    nbformat: 4,
    nbformat_minor: 5
  };
}

/**
 * Write the sample notebook (via the contents API) into the test's temp
 * directory and return its server-relative path. No browser open -- enough for
 * the filesystem-backed read tools.
 */
export async function writeNotebook(
  page: IJupyterLabPageFixture,
  tmpPath: string,
  name = 'sample.ipynb'
): Promise<string> {
  const filePath = `${tmpPath}/${name}`;
  await page.contents.uploadContent(
    JSON.stringify(sampleNotebook()),
    'text',
    filePath
  );
  return filePath;
}

/**
 * Write the sample notebook and open it in the browser. Opening it in
 * JupyterLab is what makes a live YDoc room exist (on RTC legs) and what lets
 * the jupyterlab-commands-toolkit frontend service ``execute_command`` calls.
 * Returns the server-relative path to pass to the tools.
 */
export async function createAndOpenNotebook(
  page: IJupyterLabPageFixture,
  tmpPath: string,
  name = 'sample.ipynb'
): Promise<string> {
  const filePath = await writeNotebook(page, tmpPath, name);
  const opened = await page.notebook.openByPath(filePath);
  if (!opened) {
    throw new Error(`Failed to open notebook: ${filePath}`);
  }
  return filePath;
}
