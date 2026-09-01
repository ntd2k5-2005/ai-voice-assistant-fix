from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


ASSETS = Path(__file__).resolve().parents[1] / "voice_app" / "web_assets"
HTML = (ASSETS / "index.html").read_text(encoding="utf-8")
APP_JS = (ASSETS / "app.js").read_text(encoding="utf-8")
WORKLET_JS = ASSETS / "pcm-player-worklet.js"


def test_all_literal_dom_ids_used_by_app_js_exist_in_html():
    ids = set(re.findall(r'id="([^"]+)"', HTML))
    referenced = set(re.findall(r'\$\("([A-Za-z0-9_-]+)"\)', APP_JS))
    missing = sorted(referenced - ids)
    assert not missing, f"app.js references missing DOM ids: {missing}"


def test_tab_buttons_target_real_tab_panels():
    ids = set(re.findall(r'id="([^"]+)"', HTML))
    targets = re.findall(r'class="tab(?: active)?"[^>]+data-panel="([^"]+)"', HTML)
    assert targets == ["tab-chat", "tab-knowledge", "tab-system"]
    assert all(target in ids for target in targets)
    assert HTML.count('class="tab-panel') == 3
    assert 'document.querySelectorAll(".tab-panel")' in APP_JS
    assert 'dataset?.panel' in APP_JS


def test_legacy_broken_tab_selectors_are_gone():
    assert 'querySelectorAll(".panel")' not in APP_JS
    assert 'panel-${btn.dataset.tab}' not in APP_JS
    assert 'panel-${button.dataset.tab}' not in APP_JS


def test_frontend_assets_use_cache_busting():
    assert "/assets/app.js?v=10-rebuild-1" in HTML
    assert "/assets/style.css?v=10-rebuild-1" in HTML
    assert "pcm-player-worklet.js?${ASSET_VERSION}" in APP_JS


def test_javascript_syntax_when_node_is_available():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    for asset in (ASSETS / "app.js", WORKLET_JS):
        subprocess.run([node, "--check", str(asset)], check=True, capture_output=True, text=True)


def test_worklet_reports_drained_if_stream_ends_while_rebuffering():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    harness = f"""
const fs = require('fs');
const vm = require('vm');
global.sampleRate = 48000;
global.AudioWorkletProcessor = class {{
  constructor() {{
    this.port = {{
      onmessage: null,
      messages: [],
      postMessage(msg) {{ this.messages.push(msg); }}
    }};
  }}
}};
global.registerProcessor = (_name, cls) => {{ global.WorkletClass = cls; }};
vm.runInThisContext(fs.readFileSync({str(WORKLET_JS)!r}, 'utf8'));
const player = new global.WorkletClass();
player.startedOnce = true;
player.playing = false;
player.ended = true;
player.ring.clear();
player.process([], [[new Float32Array(128)]]);
if (!player.port.messages.some((msg) => msg.type === 'drained')) process.exit(2);
"""
    subprocess.run([node, "-e", harness], check=True, capture_output=True, text=True)
