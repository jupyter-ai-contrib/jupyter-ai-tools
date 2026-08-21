/**
 * Shared fixtures for the tool E2E suite.
 *
 * The notebook is built the *proper* way -- through JupyterLab via galata
 * (`createNew` + `addCell`) -- rather than uploading a hand-crafted .ipynb
 * JSON. This gives a correctly-initialized YNotebook / kernel / active cell,
 * which the collaboration providers (and jupyterlab-notebook-awareness) expect.
 */
import type { IJupyterLabPageFixture } from '@jupyterlab/galata';

const TRANSPORT = process.env.JAI_TRANSPORT || 'default';

export interface BuiltNotebook {
  /** Server-relative path to the saved notebook. */
  path: string;
  /** Real nbformat cell ids, in order: [code "x = 1", md "# Title", code "print(x)"]. */
  cellIds: string[];
}

/**
 * Create + open a 3-cell notebook through JupyterLab (not by uploading JSON),
 * save it, and return its path plus the real cell ids read from the live model.
 */
export async function buildNotebook(
  page: IJupyterLabPageFixture,
  tmpPath: string,
  name = 'sample.ipynb'
): Promise<BuiltNotebook> {
  await page.filebrowser.openDirectory(tmpPath);

  await page.notebook.createNew(name, { kernel: 'python3' });
  await page.notebook.setCell(0, 'code', 'x = 1');
  await page.notebook.addCell('markdown', '# Title');
  await page.notebook.addCell('code', 'print(x)');

  // Persist to disk for the filesystem-read tools. galata's save() (context
  // .save()) hangs under jupyter_server_documents, but JSD autosaves the YDoc
  // to disk; on the other legs save explicitly.
  if (TRANSPORT === 'jsd') {
    await page.waitForTimeout(3000);
  } else {
    await page.notebook.save();
  }

  const path = `${tmpPath}/${name}`;

  const cellIds = await page.evaluate(() => {
    const app = (window as any).jupyterapp;
    const panel: any = app.shell.currentWidget;
    const cells = panel.content.model.cells;
    const ids: string[] = [];
    for (let i = 0; i < cells.length; i++) {
      ids.push(cells.get(i).id);
    }
    return ids;
  });

  return { path, cellIds };
}
