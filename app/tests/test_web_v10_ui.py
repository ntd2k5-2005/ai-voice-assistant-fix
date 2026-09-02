from __future__ import annotations

import re
import json
import shutil
import subprocess
from pathlib import Path

import pytest


ASSETS = Path(__file__).resolve().parents[1] / "voice_app" / "web_assets"
HTML = (ASSETS / "index.html").read_text(encoding="utf-8")
APP_JS = (ASSETS / "app.js").read_text(encoding="utf-8")
WORKLET_JS = ASSETS / "pcm-player-worklet.js"
MIC_WORKLET_JS = ASSETS / "mic-capture-worklet.js"


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
    assert "/assets/app.js?v=10-final-1" in HTML
    assert "/assets/style.css?v=10-final-1" in HTML
    assert "pcm-player-worklet.js?${ASSET_VERSION}" in APP_JS
    assert "mic-capture-worklet.js?${ASSET_VERSION}" in APP_JS


def test_javascript_syntax_when_node_is_available():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    for asset in (ASSETS / "app.js", WORKLET_JS, MIC_WORKLET_JS):
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


def test_worklet_synthetic_playout_regressions():
    """Exercise the real worklet for fast, realtime, bursty and slow producers."""
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

function run(name, intervalMs, chunkMs, count) {{
  const player = new global.WorkletClass();
  player.port.onmessage({{data: {{type: 'config', prebufferMs: 700, rebufferMs: 900}}}});
  const arrivals = [];
  for (let i = 0; i < count; i++) arrivals.push({{at: i * intervalMs, frames: Math.round(48000 * chunkMs / 1000), seq: i + 1}});
  let next = 0;
  let ended = false;
  let nowMs = 0;
  const quantumMs = 128 / 48000 * 1000;
  for (let callback = 0; callback < 20000; callback++) {{
    while (next < arrivals.length && arrivals[next].at <= nowMs + 1e-7) {{
      const item = arrivals[next++];
      const buffer = new ArrayBuffer(item.frames * 2);
      player.port.onmessage({{data: {{type: 'pcm', buffer, sequence: item.seq, frameCount: item.frames, epoch: 0}}}});
    }}
    if (!ended && next === arrivals.length && nowMs >= arrivals[arrivals.length - 1].at) {{
      ended = true;
      player.port.onmessage({{data: {{type: 'end'}}}});
    }}
    player.process([], [[new Float32Array(128)]]);
    nowMs += quantumMs;
    if (player.port.messages.some(msg => msg.type === 'drained')) break;
  }}
  const consumedSeq = player.port.messages.filter(msg => msg.type === 'consuming').map(msg => msg.sequence);
  const producedAudioMs = count * chunkMs;
  const productionElapsedMs = (count - 1) * intervalMs;
  const rate = producedAudioMs / productionElapsedMs;
  return {{
    name,
    rate,
    status: rate < 1 ? 'insufficient' : 'headroom-observed',
    underruns: player.underrunCount,
    accepted: player.cumulativeAcceptedFrames,
    consumed: player.cumulativeConsumedFrames,
    consumedSeq,
    drained: player.port.messages.some(msg => msg.type === 'drained'),
    ringSize: player.ring.size,
    spanCount: player.chunkSpans.length,
  }};
}}

const rows = [
  run('faster', 100, 200, 50),
  run('realtime', 100, 100, 100),
  run('bursty-headroom', 800, 1000, 10),
  run('slower', 100, 80, 100),
];
process.stdout.write(JSON.stringify(rows));
"""
    result = subprocess.run(
        [node, "-e", harness], check=True, capture_output=True, text=True
    )
    rows = {row["name"]: row for row in json.loads(result.stdout)}

    for name in ("faster", "realtime", "bursty-headroom"):
        assert rows[name]["underruns"] == 0, rows[name]
        assert rows[name]["status"] == "headroom-observed"

    assert rows["slower"]["underruns"] > 0
    assert rows["slower"]["status"] == "insufficient"

    for row in rows.values():
        assert row["drained"] is True
        assert row["accepted"] == row["consumed"]
        assert row["ringSize"] == 0 and row["spanCount"] == 0
        assert row["consumedSeq"] == list(range(1, len(row["consumedSeq"]) + 1))


def test_worklet_epoch_flush_rejects_stale_pcm():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    harness = f"""
const fs = require('fs');
const vm = require('vm');
global.sampleRate = 48000;
global.AudioWorkletProcessor = class {{
  constructor() {{ this.port = {{onmessage: null, messages: [], postMessage(msg) {{ this.messages.push(msg); }}}}; }}
}};
global.registerProcessor = (_name, cls) => {{ global.WorkletClass = cls; }};
vm.runInThisContext(fs.readFileSync({str(WORKLET_JS)!r}, 'utf8'));
const player = new global.WorkletClass();
player.port.onmessage({{data: {{type: 'config', epoch: 7}}}});
player.port.onmessage({{data: {{type: 'pcm', epoch: 6, sequence: 1, buffer: new ArrayBuffer(2048)}}}});
if (player.ring.size !== 0 || !player.port.messages.some(x => x.type === 'stale')) process.exit(2);
player.port.onmessage({{data: {{type: 'pcm', epoch: 7, sequence: 2, buffer: new ArrayBuffer(2048)}}}});
if (player.ring.size === 0) process.exit(3);
player.port.onmessage({{data: {{type: 'flush', epoch: 8}}}});
if (player.ring.size !== 0 || player.epoch !== 8 || !player.port.messages.some(x => x.type === 'flushed')) process.exit(4);
"""
    subprocess.run([node, "-e", harness], check=True, capture_output=True, text=True)


def test_real_vad_logic_endpoints_speech_and_rejects_impulse():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    harness = f"""
const fs = require('fs');
const vm = require('vm');
const context = {{
  console, performance, Blob, setTimeout, clearTimeout,
  location: {{protocol: 'http:', host: 'localhost'}},
  navigator: {{}},
  sessionStorage: {{getItem() {{return null;}}, setItem() {{}}, removeItem() {{}}}},
  document: {{readyState: 'loading', addEventListener() {{}}, getElementById() {{return null;}}, querySelectorAll() {{return [];}}}},
  window: {{}},
  fetch: async () => ({{ok: true, json: async () => ({{conversation_id: 'c'}})}}),
}};
vm.createContext(context);
vm.runInContext(fs.readFileSync({str(ASSETS / "app.js")!r}, 'utf8'), context);
vm.runInContext(
  'audioCtx = {{sampleRate: 48000}};' +
  'vadEnabled = true;' +
  'vadConfig = {{...vadConfig, threshold_db: -42, activation_frames: 3, pre_roll_ms: 100, min_speech_ms: 200, min_silence_ms: 300}};' +
  'sendAudioBlob = async (blob) => {{ globalThis.sentBytes = blob.size; }};' +
  'const hot = new Float32Array(1024).fill(0.1); const quiet = new Float32Array(1024);' +
  'handleMicFrame(hot); handleMicFrame(quiet); handleMicFrame(quiet);' +
  'globalThis.impulseSent = globalThis.sentBytes || 0;' +
  'for (let i = 0; i < 12; i++) handleMicFrame(hot);' +
  'for (let i = 0; i < 16; i++) handleMicFrame(quiet);',
  context
);
setTimeout(() => {{
  if (context.impulseSent !== 0) process.exit(2);
  if (!(context.sentBytes > 44)) process.exit(3);
}}, 20);
"""
    subprocess.run([node, "-e", harness], check=True, capture_output=True, text=True)
