import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONF_PATH = ROOT / "docs" / "conf.py"
JS_PATH = ROOT / "docs" / "_static" / "external-links.js"


def _load_conf_module(name: str):
    spec = importlib.util.spec_from_file_location(name, CONF_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run_node(script: str) -> subprocess.CompletedProcess[str]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for docs external-links JS tests")

    return subprocess.run(
        [node, "-e", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_docs_conf_registers_external_links_js(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("NBS_EXECUTE", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)

    conf = _load_conf_module("gwexpy_docs_conf_external_links")

    assert "external-links.js" in conf.html_js_files


def test_external_links_js_marks_only_cross_origin_http_links():
    script = f"""
    const mod = require({json.dumps(str(JS_PATH))});

    function makeAnchor(href, attrs = {{}}) {{
      const state = {{ href, ...attrs }};
      return {{
        getAttribute(name) {{
          return Object.prototype.hasOwnProperty.call(state, name) ? state[name] : null;
        }},
        setAttribute(name, value) {{
          state[name] = String(value);
        }},
      }};
    }}

    const anchors = [
      makeAnchor("https://example.com/reference"),
      makeAnchor("/internal/page"),
      makeAnchor("#section"),
      makeAnchor("mailto:test@example.com"),
      makeAnchor("https://tatsuki-washimi.github.io/gwexpy/docs/web/en/index.html"),
      makeAnchor("https://third.example/path", {{ rel: "external ugc" }}),
    ];

    mod.enhanceExternalLinks(
      {{
        querySelectorAll() {{
          return anchors;
        }},
      }},
      "https://tatsuki-washimi.github.io/gwexpy/docs/web/en/page.html",
    );

    console.log(JSON.stringify(anchors.map((anchor) => ({{
      href: anchor.getAttribute("href"),
      target: anchor.getAttribute("target"),
      rel: anchor.getAttribute("rel"),
    }}))));
    """

    result = _run_node(script)
    assert result.returncode == 0, result.stderr

    links = json.loads(result.stdout)

    assert links[0]["target"] == "_blank"
    assert set(links[0]["rel"].split()) == {"noopener", "noreferrer"}

    assert links[1]["target"] is None
    assert links[1]["rel"] is None
    assert links[2]["target"] is None
    assert links[2]["rel"] is None
    assert links[3]["target"] is None
    assert links[3]["rel"] is None
    assert links[4]["target"] is None
    assert links[4]["rel"] is None

    assert links[5]["target"] == "_blank"
    assert set(links[5]["rel"].split()) == {
        "external",
        "ugc",
        "noopener",
        "noreferrer",
    }
