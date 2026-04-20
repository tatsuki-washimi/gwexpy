import test from 'node:test';
import assert from 'node:assert/strict';

test('importing the module has no top-level chrome resolution side effect', async () => {
  const previous = {
    CHROME_BIN: process.env.CHROME_BIN,
    GOOGLE_CHROME_BIN: process.env.GOOGLE_CHROME_BIN,
    BROWSER_BIN: process.env.BROWSER_BIN,
    PATH: process.env.PATH,
  };

  delete process.env.CHROME_BIN;
  delete process.env.GOOGLE_CHROME_BIN;
  delete process.env.BROWSER_BIN;
  process.env.PATH = '';

  try {
    const mod = await import('../../scripts/validation/check_external_link_targets.mjs');
    assert.equal(typeof mod.resolveChromePath, 'function');
  } finally {
    process.env.CHROME_BIN = previous.CHROME_BIN;
    process.env.GOOGLE_CHROME_BIN = previous.GOOGLE_CHROME_BIN;
    process.env.BROWSER_BIN = previous.BROWSER_BIN;
    process.env.PATH = previous.PATH;
  }
});

const { resolveChromePath } = await import('../../scripts/validation/check_external_link_targets.mjs');

test('resolveChromePath prefers CHROME_BIN from env', () => {
  const previous = {
    CHROME_BIN: process.env.CHROME_BIN,
    GOOGLE_CHROME_BIN: process.env.GOOGLE_CHROME_BIN,
    BROWSER_BIN: process.env.BROWSER_BIN,
    PATH: process.env.PATH,
  };

  process.env.CHROME_BIN = '/tmp/custom-chrome';
  delete process.env.GOOGLE_CHROME_BIN;
  delete process.env.BROWSER_BIN;
  process.env.PATH = '';

  try {
    assert.equal(resolveChromePath(), '/tmp/custom-chrome');
  } finally {
    process.env.CHROME_BIN = previous.CHROME_BIN;
    process.env.GOOGLE_CHROME_BIN = previous.GOOGLE_CHROME_BIN;
    process.env.BROWSER_BIN = previous.BROWSER_BIN;
    process.env.PATH = previous.PATH;
  }
});

test('resolveChromePath accepts GOOGLE_CHROME_BIN from env', () => {
  const previous = {
    CHROME_BIN: process.env.CHROME_BIN,
    GOOGLE_CHROME_BIN: process.env.GOOGLE_CHROME_BIN,
    BROWSER_BIN: process.env.BROWSER_BIN,
    PATH: process.env.PATH,
  };

  delete process.env.CHROME_BIN;
  process.env.GOOGLE_CHROME_BIN = '/tmp/google-chrome';
  delete process.env.BROWSER_BIN;
  process.env.PATH = '';

  try {
    assert.equal(resolveChromePath(), '/tmp/google-chrome');
  } finally {
    process.env.CHROME_BIN = previous.CHROME_BIN;
    process.env.GOOGLE_CHROME_BIN = previous.GOOGLE_CHROME_BIN;
    process.env.BROWSER_BIN = previous.BROWSER_BIN;
    process.env.PATH = previous.PATH;
  }
});
