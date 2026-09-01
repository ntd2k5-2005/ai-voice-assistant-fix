const ASSET_VERSION = "v10-rebuild-2";
const $ = (id) => document.getElementById(id);

let audioCtx = null;
let pcmNode = null;
let currentTurn = null;
let currentControlSocket = null;
let currentAudioSocket = null;
let turnStartedAt = 0;
let prebufferMs = 700;
let rebufferMs = 900;
let mediaRecorder = null;
let micStream = null;
let micChunks = [];
let lastRawMetrics = {};
let serverTurnDone = false;
let turnBusy = false;
let pcmTraceBySequence = new Map();

const browserMetrics = () => ({
  first_pcm_received_ms: null,
  first_audio_scheduled_ms: null,
  first_audio_started_estimate_ms: null,
  audio_context_sample_rate: null,
  underrun_count: 0,
  underrun_frames: 0,
  rebuffer_count: 0,
  max_schedule_gap_ms: 0,
  min_buffer_ahead_ms: null,
  max_buffer_ahead_ms: null,
  final_buffer_ahead_ms: null,
  media_transport: "binary-websocket-pcm-v1",
  pcm_sequence_errors: 0,
  max_browser_receive_gap_ms: 0,
  producer_rate: null,
  producer_throughput_status: null,
  pcm_trace: [],
});
let bm = browserMetrics();

function nowTurnMs() {
  return performance.now() - turnStartedAt;
}

function fmtMs(value) {
  return Number.isFinite(value) ? `${Math.round(value)} ms` : "—";
}

function setText(id, value) {
  const el = $(id);
  if (el) el.textContent = value;
}

function setStatus(text, kind = "") {
  const el = $("status");
  if (!el) return;
  el.textContent = text;
  el.className = `status ${kind}`.trim();
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  }[char]));
}

function renderMarkdownLite(text) {
  let value = escapeHtml(text || "");
  value = value
    .replace(/^###\s+(.+)$/gm, "<h3>$1</h3>")
    .replace(/^##\s+(.+)$/gm, "<h2>$1</h2>")
    .replace(/^#\s+(.+)$/gm, "<h1>$1</h1>");
  value = value
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/^[-•○]\s+(.+)$/gm, "• $1");
  return value.replace(/\n/g, "<br>");
}

function setTurnBusy(value) {
  turnBusy = Boolean(value);
  const sendBtn = $("sendBtn");
  const audioFile = $("audioFile");
  if (sendBtn) sendBtn.disabled = turnBusy;
  if (audioFile) audioFile.disabled = turnBusy;
  const micBtn = $("micBtn");
  if (micBtn && !(mediaRecorder && mediaRecorder.state === "recording")) {
    micBtn.disabled = turnBusy;
  }
}

function resetPlayback() {
  bm = browserMetrics();
  pcmTraceBySequence = new Map();
  setText("bufferAhead", "0 ms");
  const bar = $("bufferBar");
  if (bar) bar.style.width = "0%";
  setText("mBrowserPcm", "—");
  setText("mUnderrun", "0");
  if (pcmNode) pcmNode.port.postMessage({ type: "reset" });
}

async function ensureAudioWorklet(sampleRate = 48000) {
  if (!window.AudioContext && !window.webkitAudioContext) {
    throw new Error("Trình duyệt không hỗ trợ Web Audio API");
  }
  if (!window.AudioWorkletNode) {
    throw new Error("Trình duyệt không hỗ trợ AudioWorklet");
  }

  if (!audioCtx) {
    const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AudioContextCtor({ sampleRate });
    await audioCtx.audioWorklet.addModule(`/assets/pcm-player-worklet.js?${ASSET_VERSION}`);
    pcmNode = new AudioWorkletNode(audioCtx, "pcm-player-v10", {
      numberOfInputs: 0,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });
    pcmNode.connect(audioCtx.destination);
    pcmNode.port.onmessage = (event) => handleWorkletMessage(event.data || {});
  }

  if (audioCtx.state !== "running") await audioCtx.resume();
  bm.audio_context_sample_rate = audioCtx.sampleRate;
  if (audioCtx.sampleRate !== sampleRate) {
    console.warn(`AudioContext sampleRate ${audioCtx.sampleRate} != source ${sampleRate}`);
  }
  pcmNode.port.postMessage({ type: "config", prebufferMs, rebufferMs });
  return audioCtx;
}

function handleWorkletMessage(msg) {
  if (!audioCtx) return;

  if (msg.type === "started") {
    if (bm.first_audio_scheduled_ms === null) bm.first_audio_scheduled_ms = nowTurnMs();
    if (bm.first_audio_started_estimate_ms === null) bm.first_audio_started_estimate_ms = nowTurnMs();
    setStatus("🔊 Đang phát realtime bằng AudioWorklet…", "talking");
  } else if (msg.type === "rebuffering") {
    setStatus("⏳ TTS chưa theo kịp · đang đệm lại âm thanh…", "talking");
  } else if (msg.type === "resumed") {
    setStatus("🔊 Đã đủ đệm · tiếp tục phát…", "talking");
  }

  if (msg.type === "accepted" || msg.type === "consuming") {
    const row = pcmTraceBySequence.get(Number(msg.sequence));
    if (row) {
      if (msg.type === "accepted") row.t8_worklet_accepted_ms = nowTurnMs();
      else row.t9_consumption_started_ms = nowTurnMs();
      row.worklet_buffered_frames = Number(msg.bufferedFrames || 0);
    }
  }

  if (["metrics", "started", "rebuffering", "resumed"].includes(msg.type)) {
    const frames = Number(msg.bufferedFrames || 0);
    const aheadMs = frames / audioCtx.sampleRate * 1000;
    setText("bufferAhead", `${Math.round(aheadMs)} ms`);
    const bar = $("bufferBar");
    if (bar) bar.style.width = `${Math.min(100, aheadMs / 20)}%`;

    if (msg.startedOnce || ["started", "resumed", "rebuffering"].includes(msg.type)) {
      bm.min_buffer_ahead_ms = bm.min_buffer_ahead_ms === null
        ? aheadMs
        : Math.min(bm.min_buffer_ahead_ms, aheadMs);
      bm.max_buffer_ahead_ms = bm.max_buffer_ahead_ms === null
        ? aheadMs
        : Math.max(bm.max_buffer_ahead_ms, aheadMs);
    }

    bm.underrun_count = Number(msg.underrunCount ?? bm.underrun_count ?? 0);
    bm.underrun_frames = Number(msg.underrunFrames ?? bm.underrun_frames ?? 0);
    bm.rebuffer_count = Number(msg.rebufferCount ?? bm.rebuffer_count ?? 0);
    bm.max_schedule_gap_ms = Number(msg.maxUnderrunFrames || 0) / audioCtx.sampleRate * 1000;
    setText("mUnderrun", `${bm.underrun_count} / rebuffer ${bm.rebuffer_count}`);
  }

  if (msg.type === "drained") {
    bm.final_buffer_ahead_ms = 0;
    if (currentTurn) void postBrowserMetrics(currentTurn);
    if (serverTurnDone) setStatus("✅ Hoàn tất · audio đã phát hết.");
  }
}

async function streamAudio(turnId, sampleRate) {
  await ensureAudioWorklet(sampleRate);
  pcmNode.port.postMessage({ type: "config", prebufferMs, rebufferMs });

  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const url = `${scheme}://${location.host}/ws/turn/${encodeURIComponent(turnId)}/audio`;
  let expectedSequence = 1;
  let previousReceiveMs = null;
  let cumulativeFrames = 0;
  let firstProducerNs = null;
  let latestProducerNs = null;

  await new Promise((resolve, reject) => {
    const ws = new WebSocket(url);
    currentAudioSocket = ws;
    ws.binaryType = "arraybuffer";
    let ended = false;

    ws.onmessage = (event) => {
      if (typeof event.data === "string") {
        let message;
        try { message = JSON.parse(event.data); }
        catch (err) { reject(err); return; }
        if (message.type === "end") {
          ended = true;
          pcmNode.port.postMessage({ type: "end" });
          ws.close(1000);
          resolve();
        }
        return;
      }

      const receivedMs = nowTurnMs();
      const buffer = event.data;
      if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 72) {
        reject(new Error("Khung PCM WebSocket không hợp lệ"));
        return;
      }
      const view = new DataView(buffer);
      const magic = String.fromCharCode(
        view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3)
      );
      const version = view.getUint16(4, true);
      const headerBytes = view.getUint16(6, true);
      if (magic !== "V10P" || version !== 1 || headerBytes < 72 || headerBytes > buffer.byteLength) {
        reject(new Error("Sai protocol PCM WebSocket"));
        return;
      }

      const sequence = view.getUint32(8, true);
      const unitIndex = view.getUint32(12, true);
      const unitChunkIndex = view.getUint32(16, true);
      const frameCount = view.getUint32(20, true);
      const stageNs = [];
      for (let offset = 24; offset <= 64; offset += 8) stageNs.push(view.getBigUint64(offset, true));
      if (sequence !== expectedSequence) bm.pcm_sequence_errors += 1;
      expectedSequence = sequence + 1;
      const payload = buffer.slice(headerBytes);
      if (payload.byteLength !== frameCount * 2) {
        reject(new Error(`PCM frame ${sequence} sai kích thước`));
        return;
      }

      if (bm.first_pcm_received_ms === null) {
        bm.first_pcm_received_ms = receivedMs;
        setText("mBrowserPcm", fmtMs(receivedMs));
      }
      const arrivalGapMs = previousReceiveMs === null ? 0 : receivedMs - previousReceiveMs;
      previousReceiveMs = receivedMs;
      bm.max_browser_receive_gap_ms = Math.max(bm.max_browser_receive_gap_ms, arrivalGapMs);
      cumulativeFrames += frameCount;
      if (stageNs[0] > 0n) {
        if (firstProducerNs === null) firstProducerNs = stageNs[0];
        latestProducerNs = stageNs[0];
      }
      if (firstProducerNs !== null && latestProducerNs > firstProducerNs) {
        const elapsedMs = Number(latestProducerNs - firstProducerNs) / 1e6;
        bm.producer_rate = (cumulativeFrames / sampleRate * 1000) / elapsedMs;
      }

      const row = {
        sequence,
        unit_index: unitIndex,
        unit_chunk_index: unitChunkIndex,
        frame_count: frameCount,
        audio_duration_ms: frameCount / sampleRate * 1000,
        arrival_gap_ms: arrivalGapMs,
        t0_model_yield_ns: stageNs[0].toString(),
        t1_service_yield_ns: stageNs[1].toString(),
        t2_client_receive_ns: stageNs[2].toString(),
        t3_media_enqueue_ns: stageNs[3].toString(),
        t4_web_audio_enqueue_ns: stageNs[4].toString(),
        t5_websocket_send_ns: stageNs[5].toString(),
        t6_browser_received_ms: receivedMs,
      };
      pcmTraceBySequence.set(sequence, row);
      bm.pcm_trace.push(row);
      row.t7_posted_to_worklet_ms = nowTurnMs();
      pcmNode.port.postMessage(
        { type: "pcm", buffer: payload, sequence, frameCount },
        [payload],
      );
    };
    ws.onerror = () => reject(new Error("WebSocket audio lỗi"));
    ws.onclose = (event) => {
      if (currentAudioSocket === ws) currentAudioSocket = null;
      if (!ended && event.code !== 1000) reject(new Error(`WebSocket audio đóng (${event.code})`));
    };
  });

  bm.producer_throughput_status = bm.producer_rate === null
    ? "unknown"
    : bm.producer_rate < 1 ? "insufficient" : "headroom-observed";
  await postBrowserMetrics(turnId);
}

async function postBrowserMetrics(turnId) {
  if (audioCtx && bm.final_buffer_ahead_ms === null) {
    const text = $("bufferAhead")?.textContent || "0";
    bm.final_buffer_ahead_ms = Number.parseFloat(text) || 0;
  }
  try {
    await fetch(`/api/turn/${encodeURIComponent(turnId)}/browser-metrics`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(bm),
    });
  } catch (err) {
    console.debug("browser metrics upload failed", err);
  }
}

function handleEvent(event, turnId) {
  const kind = event.type;
  if (kind === "keepalive") return;

  if (kind === "stage") {
    const map = {
      stt: "🎤 Đang nhận dạng giọng nói…",
      rag: "🔎 Đang truy xuất kho tri thức…",
      llm: "🧠 LLM đang trả lời…",
    };
    setStatus(map[event.stage] || event.message || "⏳ Đang xử lý…");
  } else if (kind === "transcript_ready") {
    setText("transcript", event.transcript || "—");
    $("transcript")?.classList.remove("muted");
  } else if (kind === "prepared") {
    setText("transcript", event.transcript || "—");
    $("transcript")?.classList.remove("muted");
    setStatus("🧠 Đã có ngữ cảnh · LLM đang stream…");
  } else if (kind === "llm_delta") {
    const answer = $("answer");
    if (answer) {
      answer.innerHTML = renderMarkdownLite(event.answer || "");
      answer.classList.remove("muted");
    }
  } else if (kind === "speech_unit_queued") {
    if ($("mSentence")?.textContent === "—") setText("mSentence", fmtMs(event.queued_ms));
  } else if (kind === "tts_unit_start") {
    if ($("mTtsStart")?.textContent === "—") setText("mTtsStart", fmtMs(event.started_ms));
  } else if (kind === "pcm") {
    if ($("mFirstPcm")?.textContent === "—") setText("mFirstPcm", fmtMs(event.server_arrival_ms));
  } else if (kind === "llm_done") {
    setText("mLlmDone", fmtMs(event.meta?.pipeline_done_ms));
  } else if (kind === "tts_done") {
    lastRawMetrics = event.meta || {};
    setText("rawMetrics", JSON.stringify({ server: lastRawMetrics, browser: bm }, null, 2));
    if (event.final_audio) {
      const replay = $("replay");
      if (replay) replay.src = `/api/turn/${encodeURIComponent(turnId)}/replay.wav`;
      $("replayWrap")?.classList.remove("hidden");
    }
  } else if (kind === "done") {
    serverTurnDone = true;
    lastRawMetrics = event.details?.tts || lastRawMetrics;
    setText("rawMetrics", JSON.stringify({
      server: lastRawMetrics,
      browser: bm,
      timings: event.timings,
    }, null, 2));
    const ahead = Number.parseFloat($("bufferAhead")?.textContent || "0") || 0;
    setStatus(
      ahead > 20
        ? `✅ Xử lý xong · đang phát ${Math.round(ahead)} ms audio còn lại…`
        : `✅ Hoàn tất · ${Math.round(event.timings?.total_ms || 0)} ms`,
    );
    void postBrowserMetrics(turnId);
  } else if (kind === "error") {
    setStatus(`❌ ${event.stage}: ${event.error}`, "error");
  }
}

function websocketUrl(turnId) {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${location.host}/ws/turn/${encodeURIComponent(turnId)}`;
}

function connectControl(turnId) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(websocketUrl(turnId));
    currentControlSocket = ws;
    let closedByServer = false;
    let settled = false;

    const settleReject = (error) => {
      if (settled) return;
      settled = true;
      reject(error);
    };
    const settleResolve = () => {
      if (settled) return;
      settled = true;
      resolve();
    };

    const openTimer = setTimeout(() => {
      try { ws.close(4000, "open timeout"); } catch (_) {}
      settleReject(new Error("WebSocket control plane không kết nối được sau 15 giây"));
    }, 15000);

    ws.onopen = () => clearTimeout(openTimer);
    ws.onmessage = (message) => {
      try {
        const event = JSON.parse(message.data);
        if (event.type === "stream_closed") {
          closedByServer = true;
          ws.close(1000, "done");
          return;
        }
        handleEvent(event, turnId);
      } catch (err) {
        console.error("control frame", err, message.data);
      }
    };
    ws.onerror = () => {
      clearTimeout(openTimer);
      settleReject(new Error("WebSocket control plane lỗi"));
    };
    ws.onclose = (event) => {
      clearTimeout(openTimer);
      if (currentControlSocket === ws) currentControlSocket = null;
      if (closedByServer || event.code === 1000) settleResolve();
      else settleReject(new Error(`WebSocket đóng bất thường (${event.code})`));
    };
  });
}

async function startTurn(fetcher) {
  if (turnBusy) {
    setStatus("⏳ Lượt hiện tại chưa xong.");
    return;
  }

  setTurnBusy(true);
  turnStartedAt = performance.now();
  serverTurnDone = false;
  currentTurn = null;
  resetPlayback();

  $("replayWrap")?.classList.add("hidden");
  $("replay")?.removeAttribute("src");
  setText("answer", "");
  $("answer")?.classList.add("muted");
  setText("transcript", "—");
  setText("mSentence", "—");
  setText("mTtsStart", "—");
  setText("mFirstPcm", "—");
  setText("mLlmDone", "—");
  setText("rawMetrics", "{}");
  setStatus("⏳ Đang tạo lượt hội thoại…");

  try {
    const resp = await fetcher();
    if (!resp.ok) throw new Error(await resp.text());
    const info = await resp.json();
    currentTurn = info.turn_id;
    prebufferMs = info.prebuffer_ms || 700;
    rebufferMs = info.rebuffer_ms || 900;

    await ensureAudioWorklet(info.sample_rate || 48000);
    pcmNode.port.postMessage({ type: "config", prebufferMs, rebufferMs });

    const results = await Promise.allSettled([
      connectControl(currentTurn),
      streamAudio(currentTurn, info.sample_rate || 48000),
    ]);
    const failed = results.find((result) => result.status === "rejected");
    if (failed) throw failed.reason;
  } finally {
    setTurnBusy(false);
  }
}

function commonForm(form) {
  form.append("rag_mode", $("ragMode")?.value || "dense");
  form.append("rerank", $("rerank")?.checked ? "true" : "false");
  form.append("top_k", $("topK")?.value || "5");
  form.append("voice_index", $("voiceIndex")?.value || "2");
  return form;
}

async function sendText() {
  const query = $("question")?.value.trim() || "";
  if (!query) {
    setStatus("⚠️ Nhập câu hỏi trước khi gửi.");
    return;
  }
  try {
    await startTurn(() => fetch("/api/turn/text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        rag_mode: $("ragMode")?.value || "dense",
        rerank: Boolean($("rerank")?.checked),
        top_k: Number($("topK")?.value || 5),
        voice_index: Number($("voiceIndex")?.value || 2),
      }),
    }));
  } catch (err) {
    setStatus(`❌ ${err.message}`, "error");
  }
}

async function sendAudioBlob(blob, filename = "recording.webm") {
  try {
    await startTurn(() => {
      const form = commonForm(new FormData());
      form.append("audio", blob, filename);
      return fetch("/api/turn/voice", { method: "POST", body: form });
    });
  } catch (err) {
    setStatus(`❌ ${err.message}`, "error");
  }
}

async function toggleMic() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    setText("micBtn", "🎙️ Ghi âm");
    return;
  }

  if (turnBusy) {
    setStatus("⏳ Lượt hiện tại chưa xong.");
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Trình duyệt không hỗ trợ microphone hoặc trang chưa chạy qua HTTPS");
  }
  if (!window.MediaRecorder) {
    throw new Error("Trình duyệt không hỗ trợ MediaRecorder");
  }

  await ensureAudioWorklet(48000);
  micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  micChunks = [];
  mediaRecorder = new MediaRecorder(micStream);
  mediaRecorder.ondataavailable = (event) => {
    if (event.data.size) micChunks.push(event.data);
  };
  mediaRecorder.onstop = async () => {
    micStream?.getTracks().forEach((track) => track.stop());
    const blob = new Blob(micChunks, { type: mediaRecorder.mimeType || "audio/webm" });
    setStatus("🎤 Đã dừng ghi âm · tự chạy pipeline…");
    await sendAudioBlob(blob, "recording.webm");
  };
  mediaRecorder.start();
  setText("micBtn", "⏹ Dừng và gửi");
  setStatus("🔴 Đang ghi âm…");
}

async function refreshHealth() {
  try {
    const response = await fetch("/health/ready", { cache: "no-store" });
    if (!response.ok) throw new Error(`Health HTTP ${response.status}`);
    const data = await response.json();
    setText("healthJson", JSON.stringify(data, null, 2));
    const ok = Boolean(data.ready && data.v10?.tts_realtime_contract_ok);
    const perf = data.v10?.tts_performance_probe;
    setText("healthBadge", ok
      ? (perf === "degraded" ? "V10 ready · TTS chậm" : "V10 realtime ready")
      : "Service chưa sẵn sàng");
    const badge = $("healthBadge");
    if (badge) badge.className = `badge ${ok ? "ok" : "bad"}`;
  } catch (err) {
    setText("healthBadge", "Health lỗi");
    const badge = $("healthBadge");
    if (badge) badge.className = "badge bad";
    console.error(err);
  }
}

async function refreshKnowledge() {
  const status = $("knowledgeStatus");
  try {
    const response = await fetch("/api/knowledge/documents", { cache: "no-store" });
    if (!response.ok) throw new Error(`Knowledge HTTP ${response.status}`);
    const docs = await response.json();
    const root = $("knowledgeList");
    if (!root) return;
    root.innerHTML = "";

    if (!Array.isArray(docs) || docs.length === 0) {
      root.innerHTML = '<div class="empty-state">Kho tri thức đang trống.</div>';
      return;
    }

    for (const doc of docs) {
      const item = document.createElement("div");
      item.className = "doc";

      const left = document.createElement("div");
      left.innerHTML = `<strong>${escapeHtml(doc.document_id || "")}</strong><br>`
        + `<small>${escapeHtml(doc.filename || "")} · ${Number(doc.chunk_count || 0)} chunks · ${escapeHtml(doc.status || "")}</small>`;

      const del = document.createElement("button");
      del.type = "button";
      del.textContent = "Xóa";
      del.onclick = async () => {
        if (!confirm(`Xóa ${doc.document_id}?`)) return;
        del.disabled = true;
        try {
          const response = await fetch(`/api/knowledge/documents/${encodeURIComponent(doc.document_id)}`, {
            method: "DELETE",
          });
          if (!response.ok) throw new Error(await response.text());
          await refreshKnowledge();
        } catch (err) {
          if (status) status.textContent = `❌ ${err.message}`;
        } finally {
          del.disabled = false;
        }
      };

      item.append(left, del);
      root.appendChild(item);
    }
  } catch (err) {
    if (status) status.textContent = `❌ ${err.message}`;
  }
}

async function uploadKnowledge() {
  const input = $("knowledgeFile");
  const file = input?.files?.[0];
  if (!file) {
    setText("knowledgeStatus", "⚠️ Chọn tài liệu trước khi nạp.");
    return;
  }

  const button = $("knowledgeUploadBtn");
  if (button) button.disabled = true;
  setText("knowledgeStatus", `⏳ Đang nạp ${file.name}…`);

  try {
    const form = new FormData();
    form.append("file", file, file.name);
    const requestedId = $("knowledgeId")?.value.trim();
    if (requestedId) form.append("document_id", requestedId);

    const response = await fetch("/api/knowledge/upload", { method: "POST", body: form });
    if (!response.ok) throw new Error(await response.text());
    setText("knowledgeStatus", `✅ Đã nạp ${file.name}.`);
    if (input) input.value = "";
    await refreshKnowledge();
  } catch (err) {
    setText("knowledgeStatus", `❌ ${err.message}`);
  } finally {
    if (button) button.disabled = false;
  }
}

function activateTab(button) {
  const panelId = button?.dataset?.panel;
  if (!panelId) return;

  document.querySelectorAll(".tab").forEach((tab) => {
    const active = tab === button;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
  });

  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === panelId);
  });

  if (panelId === "tab-knowledge") void refreshKnowledge();
  if (panelId === "tab-system") void refreshHealth();
}

function bindClick(id, handler) {
  const el = $(id);
  if (!el) {
    console.error(`Missing UI element #${id}`);
    return;
  }
  el.addEventListener("click", handler);
}

function bootstrap() {
  bindClick("sendBtn", () => void sendText());
  bindClick("micBtn", () => void toggleMic().catch((err) => setStatus(`❌ ${err.message}`, "error")));
  bindClick("knowledgeUploadBtn", () => void uploadKnowledge());
  bindClick("knowledgeRefreshBtn", () => void refreshKnowledge());
  bindClick("healthRefreshBtn", () => void refreshHealth());

  const audioFile = $("audioFile");
  if (audioFile) {
    audioFile.addEventListener("change", () => {
      const file = audioFile.files?.[0];
      if (file) void sendAudioBlob(file, file.name);
    });
  }

  const question = $("question");
  if (question) {
    question.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && event.ctrlKey) void sendText();
    });
  }

  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => activateTab(button));
  });

  const activeTab = document.querySelector(".tab.active") || document.querySelector(".tab");
  if (activeTab) activateTab(activeTab);
  void refreshHealth();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", bootstrap, { once: true });
} else {
  bootstrap();
}
