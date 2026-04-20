import { spawn, spawnSync } from 'node:child_process';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';

const scriptPath = fileURLToPath(import.meta.url);
const repoRoot = path.resolve(path.dirname(scriptPath), '..', '..');
const chromePath = resolveChromePath();
const chromePort = 18774;
const docsJsPath = path.join(repoRoot, 'docs', '_static', 'external-links.js');
const outDir = path.join(os.tmpdir(), 'gwexpy-17-18-external-links');
const args = process.argv.slice(2);

export function resolveChromePath() {
  const configuredPath =
    process.env.CHROME_BIN || process.env.GOOGLE_CHROME_BIN || process.env.BROWSER_BIN;
  if (configuredPath) {
    return configuredPath;
  }

  for (const candidate of ['google-chrome', 'chromium', 'chromium-browser', 'chrome']) {
    const result = spawnSync('which', [candidate], { encoding: 'utf8' });
    if (result.status === 0) {
      const resolved = result.stdout.trim();
      if (resolved) {
        return resolved;
      }
    }
  }

  throw new Error(
    'Could not resolve a Chrome/Chromium executable. Set CHROME_BIN, GOOGLE_CHROME_BIN, or BROWSER_BIN, or add google-chrome/chromium to PATH.',
  );
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchJson(url, init = {}, timeoutMs = 15000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} for ${url}`);
    }
    return await response.json();
  } finally {
    clearTimeout(timer);
  }
}

async function waitForDevtools(timeoutMs = 15000) {
  const start = Date.now();
  for (;;) {
    try {
      return await fetchJson(`http://127.0.0.1:${chromePort}/json/version`);
    } catch (error) {
      if (Date.now() - start > timeoutMs) {
        throw error;
      }
      await wait(100);
    }
  }
}

class CdpSession {
  constructor(wsUrl) {
    this.ws = new WebSocket(wsUrl);
    this.nextId = 1;
    this.pending = new Map();
    this.eventWaiters = new Map();
    this.ws.addEventListener('message', (event) => {
      const message = JSON.parse(event.data.toString());
      if (message.id) {
        const pending = this.pending.get(message.id);
        if (!pending) {
          return;
        }
        this.pending.delete(message.id);
        if (message.error) {
          pending.reject(new Error(message.error.message));
        } else {
          pending.resolve(message.result);
        }
        return;
      }
      const waiters = this.eventWaiters.get(message.method);
      if (!waiters || waiters.length === 0) {
        return;
      }
      for (const waiter of [...waiters]) {
        if (!waiter.filter || waiter.filter(message.params)) {
          waiter.resolve(message.params);
          waiters.splice(waiters.indexOf(waiter), 1);
          break;
        }
      }
    });
  }

  async open() {
    if (this.ws.readyState === WebSocket.OPEN) {
      return;
    }
    await new Promise((resolve, reject) => {
      this.ws.addEventListener('open', () => resolve(), { once: true });
      this.ws.addEventListener('error', (error) => reject(error), { once: true });
    });
  }

  send(method, params = {}) {
    const id = this.nextId++;
    this.ws.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
    });
  }

  waitFor(method, filter) {
    return new Promise((resolve) => {
      const waiters = this.eventWaiters.get(method) ?? [];
      waiters.push({ resolve, filter });
      this.eventWaiters.set(method, waiters);
    });
  }

  close() {
    this.ws.close();
  }
}

function spawnChrome(profileDir) {
  return spawn(
    chromePath,
    [
      '--headless=new',
      '--no-sandbox',
      '--disable-gpu',
      '--disable-dev-shm-usage',
      '--allow-file-access-from-files',
      `--remote-debugging-port=${chromePort}`,
      `--user-data-dir=${profileDir}`,
      'about:blank',
    ],
    { stdio: ['ignore', 'pipe', 'pipe'] },
  );
}

async function openPage(url) {
  return await fetchJson(
    `http://127.0.0.1:${chromePort}/json/new?${encodeURIComponent(url)}`,
    { method: 'PUT' },
  );
}

async function closePage(targetId) {
  const response = await fetch(`http://127.0.0.1:${chromePort}/json/close/${targetId}`, { method: 'PUT' });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} while closing target ${targetId}`);
  }
}

async function buildFixture() {
  await fs.rm(outDir, { recursive: true, force: true });
  await fs.mkdir(outDir, { recursive: true });
  const fixturePath = path.join(outDir, 'fixture.html');
  const html = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>GWexpy external link probe</title>
  </head>
  <body>
    <main>
      <a id="external-link" href="https://example.com/docs">External link</a>
      <a id="internal-relative" href="local-page.html">Relative internal link</a>
      <a id="fragment-link" href="#section">Fragment link</a>
      <section id="section">Section</section>
    </main>
    <script src="${docsJsPath}"></script>
  </body>
</html>`;
  await fs.writeFile(fixturePath, html, 'utf8');
  return fixturePath;
}

async function evaluateFixture(session) {
  await session.send('Page.enable');
  await session.send('Runtime.enable');

  const loadEvent = session.waitFor('Page.loadEventFired');
  await session.send('Page.reload', { ignoreCache: true });
  await loadEvent;
  await wait(300);

  const evaluation = await session.send('Runtime.evaluate', {
    expression: `(() => {
      const read = (id) => {
        const node = document.getElementById(id);
        return {
          id,
          href: node.getAttribute('href'),
          target: node.getAttribute('target'),
          rel: node.getAttribute('rel'),
        };
      };
      return {
        external: read('external-link'),
        internalRelative: read('internal-relative'),
        fragment: read('fragment-link'),
      };
    })()`,
    returnByValue: true,
  });
  return evaluation.result.value;
}

async function evaluateBuiltPage(session) {
  await session.send('Page.enable');
  await session.send('Runtime.enable');

  const loadEvent = session.waitFor('Page.loadEventFired');
  await session.send('Page.reload', { ignoreCache: true });
  await loadEvent;
  await wait(400);

  const evaluation = await session.send('Runtime.evaluate', {
    expression: `(() => {
      const links = [...document.querySelectorAll('a[href]')].map((link) => {
        const rawHref = link.getAttribute('href');
        let isHttp = false;
        let isExternal = false;
        try {
          const url = new URL(link.href, document.baseURI);
          isHttp = url.protocol === 'http:' || url.protocol === 'https:';
          isExternal = isHttp && (
            window.location.protocol === 'file:' ? true : url.origin !== window.location.origin
          );
        } catch (err) {
          isHttp = false;
          isExternal = false;
        }
        const isInternalDocLink = !isExternal && (
          rawHref.startsWith('#') ||
          (!rawHref.startsWith('http://') &&
           !rawHref.startsWith('https://') &&
           !rawHref.startsWith('mailto:') &&
           !rawHref.startsWith('ftp:'))
        );
        return {
          text: (link.textContent || '').trim(),
          rawHref,
          target: link.getAttribute('target'),
          rel: link.getAttribute('rel'),
          isExternal,
          isInternalDocLink,
        };
      });
      return {
        title: document.title,
        externalLinks: links.filter((link) => link.isExternal),
        internalDocLinks: links.filter((link) => link.isInternalDocLink),
      };
    })()`,
    returnByValue: true,
  });
  return evaluation.result.value;
}

function assertResults(results) {
  const errors = [];
  if (results.external.target !== '_blank') {
    errors.push(`external link target expected "_blank", got ${results.external.target}`);
  }
  const externalRelTokens = new Set((results.external.rel ?? '').split(/\s+/).filter(Boolean));
  if (!externalRelTokens.has('noopener')) {
    errors.push(`external link rel expected to include "noopener", got ${results.external.rel}`);
  }
  if (results.internalRelative.target !== null) {
    errors.push(`relative internal link target expected null, got ${results.internalRelative.target}`);
  }
  if (results.fragment.target !== null) {
    errors.push(`fragment link target expected null, got ${results.fragment.target}`);
  }
  if (errors.length > 0) {
    const error = new Error(errors.join('\n'));
    error.results = results;
    throw error;
  }
}

function assertBuiltPage(results) {
  const errors = [];
  const badExternal = results.externalLinks.filter(
    (link) =>
      link.target !== '_blank' ||
      !new Set((link.rel ?? '').split(/\s+/).filter(Boolean)).has('noopener'),
  );
  const badInternal = results.internalDocLinks.filter((link) => link.target !== null);

  if (results.externalLinks.length === 0) {
    errors.push('expected at least one external http(s) link on built page');
  }
  if (results.internalDocLinks.length === 0) {
    errors.push('expected at least one internal doc link or fragment on built page');
  }
  if (badExternal.length > 0) {
    errors.push(`external links missing target/rel: ${JSON.stringify(badExternal.slice(0, 5), null, 2)}`);
  }
  if (badInternal.length > 0) {
    errors.push(`internal doc links should not get target: ${JSON.stringify(badInternal.slice(0, 5), null, 2)}`);
  }
  if (errors.length > 0) {
    const error = new Error(errors.join('\n'));
    error.results = results;
    throw error;
  }
}

function parseBuiltPages() {
  const pages = [];
  for (let i = 0; i < args.length; i += 1) {
    if (args[i] === '--built-page' && args[i + 1]) {
      pages.push(args[i + 1]);
      i += 1;
    }
  }
  return pages;
}

async function main() {
  const builtPages = parseBuiltPages();
  const fixturePath = builtPages.length === 0 ? await buildFixture() : null;
  const profileDir = await fs.mkdtemp(path.join(os.tmpdir(), 'gwexpy-17-18-chrome-'));
  const chrome = spawnChrome(profileDir);
  try {
    await waitForDevtools();
    if (builtPages.length === 0) {
      const target = await openPage(pathToFileURL(fixturePath).href);
      const session = new CdpSession(target.webSocketDebuggerUrl);
      await session.open();
      const results = await evaluateFixture(session);
      session.close();
      await closePage(target.id);
      await fs.writeFile(path.join(outDir, 'results.json'), JSON.stringify(results, null, 2));
      assertResults(results);
      console.log(JSON.stringify(results, null, 2));
      return;
    }

    const pageResults = [];
    for (const pagePath of builtPages) {
      const target = await openPage(pathToFileURL(pagePath).href);
      const session = new CdpSession(target.webSocketDebuggerUrl);
      await session.open();
      const results = await evaluateBuiltPage(session);
      pageResults.push({ pagePath, results });
      session.close();
      await closePage(target.id);
      assertBuiltPage(results);
    }
    await fs.writeFile(path.join(outDir, 'built-page-results.json'), JSON.stringify(pageResults, null, 2));
    console.log(JSON.stringify(pageResults, null, 2));
  } finally {
    chrome.kill('SIGKILL');
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === scriptPath) {
  main().catch(async (error) => {
    await fs.mkdir(outDir, { recursive: true });
    const payload = {
      message: error.message,
      results: error.results ?? null,
    };
    await fs.writeFile(path.join(outDir, 'error.json'), JSON.stringify(payload, null, 2));
    console.error(error.message);
    process.exit(1);
  });
}
