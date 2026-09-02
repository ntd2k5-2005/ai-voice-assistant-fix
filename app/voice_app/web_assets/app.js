const ASSET_VERSION = "v10-final-1";
const $ = (id) => document.getElementById(id);

let audioCtx = null;
let pcmNode = null;
let conversationId = null;
let currentTurn = null;
let currentGeneration = 0;
let playbackEpoch = 0;
let requestToken = 0;
let currentControlSocket = null;
let currentAudioSocket = null;
let turnStartedAt = 0;
let prebufferMs = 700;
let rebufferMs = 900;
let serverTurnDone = false;
let turnBusy = false;
let cancelledTurns = new Set();
let pcmTraceBySequence = new Map();
let assistantNodes = new Map();
let lastRawMetrics = {};

let micStream = null;
let micSource = null;
let micCaptureNode = null;
let micMuteGain = null;
let pttRecording = false;
let pttChunks = [];
let vadEnabled = false;
let vadSpeech = false;
let vadChunks = [];
let vadSpeechFrames = 0;
let vadVoicedFrames = 0;
let vadSilenceFrames = 0;
let vadHotFrames = 0;
let vadPreRoll = [];
let vadPreRollFrames = 0;
let vadNoiseFloorDb = -60;
let pendingBargeIn = null;
let micModuleLoaded = false;
let lastBargeDetectedAt = null;
let vadConfig = {
  threshold_db: -42,
  pre_roll_ms: 300,
  min_speech_ms: 250,
  min_silence_ms: 650,
  activation_frames: 3,
  barge_in_debounce_ms: 180,
  max_utterance_ms: 120000,
};

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
  stale_packets_dropped: 0,
  control_reconnect_count: 0,
  audio_reconnect_count: 0,
  barge_in_detected_ms: null,
  playback_flush_ms: null,
  pcm_trace: [],
});
let bm = browserMetrics();

function nowTurnMs() { return performance.now() - turnStartedAt; }
function fmtMs(value) { return Number.isFinite(value) ? `${Math.round(value)} ms` : "—"; }
function setText(id, value) { const el = $(id); if (el) el.textContent = value; }

function friendlyError(error) {
  const raw = String(error?.message || error || "Lỗi không xác định");
  try {
    const parsed = JSON.parse(raw);
    return parsed.detail || raw;
  } catch (_) {
    return raw.replace(/^Error:\s*/i, "");
  }
}

function setStatus(text, state = "idle") {
  const el = $("status");
  if (!el) return;
  el.textContent = text;
  el.dataset.state = state;
  el.className = `status ${state === "error" ? "error" : ""}`.trim();
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
  }[char]));
}

function renderMarkdownLite(text) {
  return escapeHtml(text || "")
    .replace(/^###\s+(.+)$/gm, "<h3>$1</h3>")
    .replace(/^##\s+(.+)$/gm, "<h2>$1</h2>")
    .replace(/^#\s+(.+)$/gm, "<h1>$1</h1>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/^[-•◦]\s+(.+)$/gm, "• $1")
    .replace(/\n/g, "<br>");
}

function scrollChat() {
  const log = $("chatLog");
  if (log) log.scrollTop = log.scrollHeight;
}

function appendMessage(role, text, turnId = "") {
  $("chatLog")?.querySelector(".welcome")?.remove();
  const root = document.createElement("article");
  root.className = `message ${role}`;
  if (turnId) root.dataset.turnId = turnId;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = renderMarkdownLite(text || (role === "assistant" ? "…" : ""));
  const meta = document.createElement("div");
  meta.className = "message-meta";
  meta.textContent = role === "user" ? "Bạn" : "Eureka";
  root.append(bubble, meta);
  $("chatLog")?.append(root);
  if (role === "assistant" && turnId) assistantNodes.set(turnId, root);
  scrollChat();
  return root;
}

function updateAssistant(turnId, text) {
  let root = assistantNodes.get(turnId);
  if (!root) root = appendMessage("assistant", "", turnId);
  const bubble = root.querySelector(".bubble");
  if (bubble) bubble.innerHTML = renderMarkdownLite(text || "…");
  scrollChat();
}

function renderSources(turnId, sources) {
  const root = assistantNodes.get(turnId);
  if (!root) return;
  root.querySelector(".sources")?.remove();
  if (!Array.isArray(sources) || !sources.length) return;
  const wrap = document.createElement("div");
  wrap.className = "sources";
  for (const source of sources) {
    const item = document.createElement("span");
    item.className = "source";
    const location = source.page != null ? ` · trang ${source.page}` : "";
    item.textContent = `[${source.rank || "?"}] ${source.source || source.document_id || "Nguồn"}${location}`;
    item.title = source.excerpt || "";
    wrap.append(item);
  }
  root.append(wrap);
}

function setTurnBusy(value) {
  turnBusy = Boolean(value);
  if ($("sendBtn")) $("sendBtn").disabled = turnBusy;
  if ($("audioFile")) $("audioFile").disabled = turnBusy;
  if ($("interruptBtn")) $("interruptBtn").disabled = !turnBusy;
  if ($("micBtn")) $("micBtn").disabled = turnBusy && !vadEnabled && !pttRecording;
}

function resetPlayback(epoch = ++playbackEpoch) {
  bm = browserMetrics();
  pcmTraceBySequence = new Map();
  setText("bufferAhead", "0 ms");
  setText("mBrowserPcm", "—");
  setText("mUnderrun", "0 / 0");
  const bar = $("bufferBar");
  if (bar) bar.style.width = "0%";
  if (pcmNode) pcmNode.port.postMessage({ type: "reset", epoch });
  return epoch;
}

function flushPlayback(detectedAt = performance.now()) {
  playbackEpoch += 1;
  lastBargeDetectedAt = detectedAt;
  bm.barge_in_detected_ms = nowTurnMs();
  if (pcmNode) pcmNode.port.postMessage({ type: "flush", epoch: playbackEpoch });
  setText("bufferAhead", "0 ms");
  const bar = $("bufferBar");
  if (bar) bar.style.width = "0%";
  return playbackEpoch;
}

async function ensureAudioWorklet(sampleRate = 48000) {
  if (!window.AudioContext && !window.webkitAudioContext) throw new Error("Trình duyệt không hỗ trợ Web Audio API");
  if (!window.AudioWorkletNode) throw new Error("Trình duyệt không hỗ trợ AudioWorklet");
  if (!audioCtx) {
    const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AudioContextCtor({ sampleRate });
    await audioCtx.audioWorklet.addModule(`/assets/pcm-player-worklet.js?${ASSET_VERSION}`);
    pcmNode = new AudioWorkletNode(audioCtx, "pcm-player-v10", {
      numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [1],
    });
    pcmNode.connect(audioCtx.destination);
    pcmNode.port.onmessage = (event) => handleWorkletMessage(event.data || {});
  }
  if (audioCtx.state !== "running") await audioCtx.resume();
  bm.audio_context_sample_rate = audioCtx.sampleRate;
  pcmNode.port.postMessage({ type: "config", prebufferMs, rebufferMs, epoch: playbackEpoch });
  return audioCtx;
}

function handleWorkletMessage(msg) {
  if (!audioCtx) return;
  if (msg.type === "flushed" && lastBargeDetectedAt !== null) {
    bm.playback_flush_ms = performance.now() - lastBargeDetectedAt;
    setText("mBargeIn", fmtMs(bm.playback_flush_ms));
    lastBargeDetectedAt = null;
  } else if (msg.type === "stale") {
    bm.stale_packets_dropped += 1;
  } else if (msg.type === "overflow") {
    setStatus("Bộ đệm trình duyệt đã đầy; hãy dừng lượt và thử lại.", "error");
  } else if (msg.type === "started") {
    if (bm.first_audio_scheduled_ms === null) bm.first_audio_scheduled_ms = nowTurnMs();
    if (bm.first_audio_started_estimate_ms === null) bm.first_audio_started_estimate_ms = nowTurnMs();
    setStatus("Đang nói… Bạn có thể ngắt lời khi dùng VAD.", "speaking");
  } else if (msg.type === "rebuffering") {
    setStatus("TTS đang đệm lại âm thanh…", "speaking");
  } else if (msg.type === "resumed") {
    setStatus("Đang tiếp tục phát câu trả lời…", "speaking");
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
    if ($("bufferBar")) $("bufferBar").style.width = `${Math.min(100, aheadMs / 20)}%`;
    if (msg.startedOnce || ["started", "resumed", "rebuffering"].includes(msg.type)) {
      bm.min_buffer_ahead_ms = bm.min_buffer_ahead_ms === null ? aheadMs : Math.min(bm.min_buffer_ahead_ms, aheadMs);
      bm.max_buffer_ahead_ms = bm.max_buffer_ahead_ms === null ? aheadMs : Math.max(bm.max_buffer_ahead_ms, aheadMs);
    }
    bm.underrun_count = Number(msg.underrunCount ?? bm.underrun_count);
    bm.underrun_frames = Number(msg.underrunFrames ?? bm.underrun_frames);
    bm.rebuffer_count = Number(msg.rebufferCount ?? bm.rebuffer_count);
    bm.max_schedule_gap_ms = Number(msg.maxUnderrunFrames || 0) / audioCtx.sampleRate * 1000;
    setText("mUnderrun", `${bm.underrun_count} / ${bm.rebuffer_count}`);
  }
  if (msg.type === "drained") {
    bm.final_buffer_ahead_ms = 0;
    if (currentTurn) void postBrowserMetrics(currentTurn);
    if (serverTurnDone) setStatus(vadEnabled ? "Hoàn tất · VAD đang chờ bạn nói." : "Hoàn tất.", "idle");
  }
}

function owns(turnId, generation) {
  return currentTurn === turnId && currentGeneration === generation && !cancelledTurns.has(turnId);
}

function websocketUrl(turnId, suffix = "", after = null) {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const query = after == null ? "" : `?after=${encodeURIComponent(after)}`;
  return `${scheme}://${location.host}/ws/turn/${encodeURIComponent(turnId)}${suffix}${query}`;
}

async function streamAudio(turnId, generation, epoch, sampleRate) {
  await ensureAudioWorklet(sampleRate);
  let expectedSequence = 1;
  let previousReceiveMs = null;
  let cumulativeFrames = 0;
  let firstProducerNs = null;
  let latestProducerNs = null;

  for (let attempt = 0; attempt < 3 && owns(turnId, generation); attempt++) {
    try {
      const ended = await new Promise((resolve, reject) => {
        const ws = new WebSocket(websocketUrl(turnId, "/audio", expectedSequence - 1));
        currentAudioSocket = ws;
        ws.binaryType = "arraybuffer";
        let cleanEnd = false;
        ws.onmessage = (event) => {
          if (!owns(turnId, generation)) return;
          if (typeof event.data === "string") {
            try {
              const message = JSON.parse(event.data);
              if (message.type === "end") {
                cleanEnd = true;
                pcmNode.port.postMessage({ type: "end", epoch });
                ws.close(1000, "done");
                resolve(true);
              }
            } catch (error) { reject(error); }
            return;
          }
          const buffer = event.data;
          if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 72) { reject(new Error("Khung PCM không hợp lệ")); return; }
          const view = new DataView(buffer);
          const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
          const version = view.getUint16(4, true);
          const headerBytes = view.getUint16(6, true);
          if (magic !== "V10P" || version !== 1 || headerBytes < 72 || headerBytes > buffer.byteLength) {
            reject(new Error("Sai protocol PCM")); return;
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
          if (payload.byteLength !== frameCount * 2) { reject(new Error(`PCM frame ${sequence} sai kích thước`)); return; }
          const receivedMs = nowTurnMs();
          if (bm.first_pcm_received_ms === null) { bm.first_pcm_received_ms = receivedMs; setText("mBrowserPcm", fmtMs(receivedMs)); }
          const arrivalGapMs = previousReceiveMs === null ? 0 : receivedMs - previousReceiveMs;
          previousReceiveMs = receivedMs;
          bm.max_browser_receive_gap_ms = Math.max(bm.max_browser_receive_gap_ms, arrivalGapMs);
          cumulativeFrames += frameCount;
          if (stageNs[0] > 0n) {
            if (firstProducerNs === null) firstProducerNs = stageNs[0];
            latestProducerNs = stageNs[0];
          }
          if (firstProducerNs !== null && latestProducerNs > firstProducerNs) {
            bm.producer_rate = (cumulativeFrames / sampleRate * 1000) / (Number(latestProducerNs - firstProducerNs) / 1e6);
          }
          const row = {
            sequence, unit_index: unitIndex, unit_chunk_index: unitChunkIndex, frame_count: frameCount,
            audio_duration_ms: frameCount / sampleRate * 1000, arrival_gap_ms: arrivalGapMs,
            t0_model_yield_ns: stageNs[0].toString(), t1_service_yield_ns: stageNs[1].toString(),
            t2_client_receive_ns: stageNs[2].toString(), t3_media_enqueue_ns: stageNs[3].toString(),
            t4_web_audio_enqueue_ns: stageNs[4].toString(), t5_websocket_send_ns: stageNs[5].toString(),
            t6_browser_received_ms: receivedMs,
          };
          pcmTraceBySequence.set(sequence, row);
          bm.pcm_trace.push(row);
          row.t7_posted_to_worklet_ms = nowTurnMs();
          if (owns(turnId, generation) && epoch === playbackEpoch) {
            pcmNode.port.postMessage({ type: "pcm", buffer: payload, sequence, frameCount, epoch }, [payload]);
          } else {
            bm.stale_packets_dropped += 1;
          }
        };
        ws.onerror = () => reject(new Error("WebSocket audio lỗi"));
        ws.onclose = (event) => {
          if (currentAudioSocket === ws) currentAudioSocket = null;
          if (!cleanEnd && event.code !== 1000) reject(new Error(`WebSocket audio đóng (${event.code})`));
        };
      });
      if (ended) break;
    } catch (error) {
      if (!owns(turnId, generation)) return;
      if (attempt === 2) throw error;
      bm.audio_reconnect_count += 1;
      await new Promise((resolve) => setTimeout(resolve, 200 * (attempt + 1)));
    }
  }
  bm.producer_throughput_status = bm.producer_rate === null ? "unknown" : bm.producer_rate < 1 ? "insufficient" : "headroom-observed";
  await postBrowserMetrics(turnId);
}

function handleEvent(event, turnId, generation) {
  if (!owns(turnId, generation) || event.type === "keepalive") return;
  const kind = event.type;
  if (kind === "stage") {
    const map = {
      stt: ["Đang nhận dạng giọng nói…", "transcribing"],
      rag: ["Đang truy xuất kho tri thức…", "retrieving"],
      llm: ["Đang tạo câu trả lời…", "thinking"],
    };
    const item = map[event.stage] || [event.message || "Đang xử lý…", "thinking"];
    setStatus(item[0], item[1]);
  } else if (kind === "transcript_ready") {
    const hasUser = $("chatLog")?.querySelector(`.message.user[data-turn-id="${turnId}"]`);
    if (!hasUser) {
      const node = appendMessage("user", event.transcript || "", turnId);
      node.dataset.turnId = turnId;
    }
  } else if (kind === "prepared") {
    updateAssistant(turnId, "…");
    renderSources(turnId, event.sources || []);
    setStatus("Đã có ngữ cảnh · LLM đang stream…", "thinking");
  } else if (kind === "llm_delta") {
    updateAssistant(turnId, event.answer || "");
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
    if (event.final_audio) {
      $("replay").src = `/api/turn/${encodeURIComponent(turnId)}/replay.wav`;
      $("replayWrap")?.classList.remove("hidden");
    }
  } else if (kind === "done") {
    serverTurnDone = true;
    updateAssistant(turnId, event.answer || "");
    renderSources(turnId, event.sources || []);
    lastRawMetrics = event.details?.tts || lastRawMetrics;
    setText("rawMetrics", JSON.stringify({ server: lastRawMetrics, browser: bm, timings: event.timings }, null, 2));
    setStatus("Xử lý xong · đang phát phần âm thanh còn lại…", "speaking");
    void postBrowserMetrics(turnId);
  } else if (kind === "cancelled" || kind === "interrupted") {
    setStatus(kind === "interrupted" ? "Đã ngắt lượt cũ · đang nghe câu mới…" : "Đã dừng lượt trả lời.", "interrupted");
  } else if (kind === "error") {
    setStatus(`${event.stage}: ${event.error}`, "error");
  }
}

async function connectControl(turnId, generation) {
  let cursor = 0;
  for (let attempt = 0; attempt < 4 && owns(turnId, generation); attempt++) {
    try {
      const closed = await new Promise((resolve, reject) => {
        const ws = new WebSocket(websocketUrl(turnId, "", cursor));
        currentControlSocket = ws;
        let clean = false;
        const timer = setTimeout(() => { try { ws.close(4000, "timeout"); } catch (_) {} reject(new Error("Control WebSocket quá hạn")); }, 15000);
        ws.onopen = () => clearTimeout(timer);
        ws.onmessage = (message) => {
          try {
            const event = JSON.parse(message.data);
            if (event.event_sequence) cursor = Math.max(cursor, Number(event.event_sequence));
            if (event.type === "stream_closed") { clean = true; ws.close(1000, "done"); resolve(true); return; }
            handleEvent(event, turnId, generation);
          } catch (error) { reject(error); }
        };
        ws.onerror = () => { clearTimeout(timer); reject(new Error("Control WebSocket lỗi")); };
        ws.onclose = (event) => {
          clearTimeout(timer);
          if (currentControlSocket === ws) currentControlSocket = null;
          if (clean || event.code === 1000) resolve(true);
          else reject(new Error(`Control WebSocket đóng (${event.code})`));
        };
      });
      if (closed) return;
    } catch (error) {
      if (!owns(turnId, generation)) return;
      if (attempt === 3) throw error;
      bm.control_reconnect_count += 1;
      await new Promise((resolve) => setTimeout(resolve, 200 * (attempt + 1)));
    }
  }
}

async function postBrowserMetrics(turnId) {
  if (!turnId) return;
  try {
    await fetch(`/api/turn/${encodeURIComponent(turnId)}/browser-metrics`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(bm),
    });
  } catch (error) { console.debug("browser metrics upload failed", error); }
}

async function ensureConversation() {
  const saved = sessionStorage.getItem("eureka.conversation");
  if (saved) {
    const response = await fetch(`/api/conversations/${encodeURIComponent(saved)}`, { cache: "no-store" });
    if (response.ok) {
      const data = await response.json();
      conversationId = data.conversation_id;
      restoreConversation(data);
      return conversationId;
    }
  }
  const response = await fetch("/api/conversations", { method: "POST" });
  if (!response.ok) throw new Error(await response.text());
  const data = await response.json();
  conversationId = data.conversation_id;
  sessionStorage.setItem("eureka.conversation", conversationId);
  setText("conversationLabel", conversationId.slice(0, 12));
  return conversationId;
}

function restoreConversation(data) {
  setText("conversationLabel", data.conversation_id.slice(0, 12));
  const turns = Array.isArray(data.turns) ? data.turns : [];
  if (!turns.length) return;
  $("chatLog").innerHTML = "";
  for (const turn of turns) {
    if (turn.user_text) appendMessage("user", turn.user_text, turn.turn_id);
    if (turn.assistant_text) {
      appendMessage("assistant", turn.assistant_text, turn.turn_id);
      renderSources(turn.turn_id, turn.sources || []);
    }
  }
}

async function interruptCurrentTurn(reason = "user_cancelled", detectedAt = null) {
  const turnId = currentTurn;
  if (!turnId) return;
  cancelledTurns.add(turnId);
  requestToken += 1;
  flushPlayback(detectedAt || performance.now());
  currentControlSocket?.close(4001, reason);
  currentAudioSocket?.close(4001, reason);
  currentControlSocket = null;
  currentAudioSocket = null;
  currentTurn = null;
  setTurnBusy(false);
  try {
    const response = await fetch(`/api/turn/${encodeURIComponent(turnId)}/cancel`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ reason }),
    });
    if (!response.ok && response.status !== 409) throw new Error(await response.text());
  } catch (error) {
    setStatus(`Không thể hủy lượt: ${friendlyError(error)}`, "error");
  }
}

function resetTurnTelemetry(epoch) {
  resetPlayback(epoch);
  serverTurnDone = false;
  $("replayWrap")?.classList.add("hidden");
  $("replay")?.removeAttribute("src");
  for (const id of ["mSentence", "mTtsStart", "mFirstPcm", "mLlmDone", "mBargeIn"]) setText(id, "—");
  setText("rawMetrics", "{}");
}

async function startTurn(fetcher, userText = "") {
  if (turnBusy) throw new Error("Lượt hiện tại chưa kết thúc");
  await ensureConversation();
  const token = ++requestToken;
  setTurnBusy(true);
  turnStartedAt = performance.now();
  currentTurn = null;
  currentGeneration = 0;
  const epoch = ++playbackEpoch;
  resetTurnTelemetry(epoch);
  setStatus("Đang tạo lượt hội thoại…", "thinking");

  try {
    const response = await fetcher(conversationId);
    if (!response.ok) throw new Error(await response.text());
    const info = await response.json();
    if (token !== requestToken) return;
    currentTurn = info.turn_id;
    currentGeneration = info.generation;
    conversationId = info.conversation_id;
    sessionStorage.setItem("eureka.conversation", conversationId);
    setText("conversationLabel", conversationId.slice(0, 12));
    if (userText) {
      const node = appendMessage("user", userText, currentTurn);
      node.dataset.turnId = currentTurn;
    }
    appendMessage("assistant", "…", currentTurn);
    prebufferMs = info.prebuffer_ms || 700;
    rebufferMs = info.rebuffer_ms || 900;
    if (info.vad) vadConfig = { ...vadConfig, ...info.vad };
    await ensureAudioWorklet(info.sample_rate || 48000);
    pcmNode.port.postMessage({ type: "config", prebufferMs, rebufferMs, epoch });
    const results = await Promise.allSettled([
      connectControl(currentTurn, currentGeneration),
      streamAudio(currentTurn, currentGeneration, epoch, info.sample_rate || 48000),
    ]);
    if (token !== requestToken || cancelledTurns.has(info.turn_id)) return;
    const failed = results.find((result) => result.status === "rejected");
    if (failed) throw failed.reason;
  } finally {
    if (token === requestToken) {
      setTurnBusy(false);
      currentTurn = null;
    }
  }
}

function commonForm(form, id) {
  form.append("conversation_id", id);
  form.append("rag_mode", $("ragMode")?.value || "dense");
  form.append("rerank", $("rerank")?.checked ? "true" : "false");
  form.append("top_k", $("topK")?.value || "5");
  form.append("voice_index", $("voiceIndex")?.value || "2");
  return form;
}

async function sendText() {
  const query = $("question")?.value.trim() || "";
  if (!query) { setStatus("Hãy nhập câu hỏi trước khi gửi.", "error"); return; }
  $("question").value = "";
  try {
    await startTurn((id) => fetch("/api/turn/text", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        conversation_id: id, query, rag_mode: $("ragMode")?.value || "dense",
        rerank: Boolean($("rerank")?.checked), top_k: Number($("topK")?.value || 5),
        voice_index: Number($("voiceIndex")?.value || 2),
      }),
    }), query);
  } catch (error) { setStatus(friendlyError(error), "error"); }
}

async function sendAudioBlob(blob, filename = "recording.wav") {
  try {
    await startTurn((id) => {
      const form = commonForm(new FormData(), id);
      form.append("audio", blob, filename);
      return fetch("/api/turn/voice", { method: "POST", body: form });
    });
  } catch (error) { setStatus(friendlyError(error), "error"); }
}

function concatFrames(chunks) {
  const length = chunks.reduce((sum, frame) => sum + frame.length, 0);
  const result = new Float32Array(length);
  let offset = 0;
  for (const frame of chunks) { result.set(frame, offset); offset += frame.length; }
  return result;
}

function wavBlob(chunks, sampleRate) {
  const samples = concatFrames(chunks);
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const write = (offset, value) => { for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i)); };
  write(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true); write(8, "WAVE");
  write(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  write(36, "data"); view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const value = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, value < 0 ? value * 32768 : value * 32767, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

async function ensureMicCapture() {
  await ensureAudioWorklet(48000);
  if (micCaptureNode && micStream?.active) return;
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("Microphone cần HTTPS hoặc localhost");
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  if (!micModuleLoaded) {
    await audioCtx.audioWorklet.addModule(`/assets/mic-capture-worklet.js?${ASSET_VERSION}`);
    micModuleLoaded = true;
  }
  micSource = audioCtx.createMediaStreamSource(micStream);
  micCaptureNode = new AudioWorkletNode(audioCtx, "eureka-mic-capture", {
    numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1],
  });
  micMuteGain = audioCtx.createGain();
  micMuteGain.gain.value = 0;
  micSource.connect(micCaptureNode).connect(micMuteGain).connect(audioCtx.destination);
  micCaptureNode.port.onmessage = (event) => {
    const message = event.data || {};
    if (message.type === "frame" && message.samples) handleMicFrame(message.samples);
  };
}

function stopMicCapture() {
  micStream?.getTracks().forEach((track) => track.stop());
  try { micSource?.disconnect(); micCaptureNode?.disconnect(); micMuteGain?.disconnect(); } catch (_) {}
  micStream = null; micSource = null; micCaptureNode = null; micMuteGain = null;
}

function frameDb(samples) {
  let energy = 0;
  for (let i = 0; i < samples.length; i++) energy += samples[i] * samples[i];
  return 20 * Math.log10(Math.sqrt(energy / Math.max(1, samples.length)) + 1e-8);
}

function pushPreRoll(samples) {
  vadPreRoll.push(samples.slice());
  vadPreRollFrames += samples.length;
  const maxFrames = Math.round(audioCtx.sampleRate * vadConfig.pre_roll_ms / 1000);
  while (vadPreRollFrames > maxFrames && vadPreRoll.length) {
    vadPreRollFrames -= vadPreRoll.shift().length;
  }
}

function handleMicFrame(samples) {
  if (pttRecording) pttChunks.push(samples.slice());
  if (!vadEnabled) return;
  const db = frameDb(samples);
  const effectiveThreshold = Math.max(vadConfig.threshold_db, Math.min(-28, vadNoiseFloorDb + 10));
  const hot = db >= effectiveThreshold;

  if (!vadSpeech) {
    pushPreRoll(samples);
    if (!hot) vadNoiseFloorDb = vadNoiseFloorDb * 0.98 + db * 0.02;
    vadHotFrames = hot ? vadHotFrames + 1 : Math.max(0, vadHotFrames - 1);
    if (vadHotFrames >= vadConfig.activation_frames) {
      vadSpeech = true;
      vadChunks = vadPreRoll.map((frame) => frame.slice());
      vadSpeechFrames = vadChunks.reduce((sum, frame) => sum + frame.length, 0);
      vadVoicedFrames = samples.length;
      vadSilenceFrames = 0;
      vadPreRoll = [];
      vadPreRollFrames = 0;
      vadHotFrames = 0;
      const detectedAt = performance.now();
      setStatus("Đang nghe…", "listening");
      if (currentTurn && !pendingBargeIn) {
        pendingBargeIn = interruptCurrentTurn("barge_in", detectedAt).finally(() => { pendingBargeIn = null; });
      }
    }
    return;
  }

  vadChunks.push(samples.slice());
  vadSpeechFrames += samples.length;
  if (hot) vadVoicedFrames += samples.length;
  vadSilenceFrames = hot ? 0 : vadSilenceFrames + samples.length;
  const minSpeechFrames = audioCtx.sampleRate * vadConfig.min_speech_ms / 1000;
  const minSilenceFrames = audioCtx.sampleRate * vadConfig.min_silence_ms / 1000;
  const maxUtteranceFrames = audioCtx.sampleRate * vadConfig.max_utterance_ms / 1000;
  if (vadSilenceFrames >= minSilenceFrames || vadSpeechFrames >= maxUtteranceFrames) {
    const chunks = vadChunks;
    const valid = vadVoicedFrames >= minSpeechFrames;
    vadSpeech = false; vadChunks = []; vadSpeechFrames = 0; vadVoicedFrames = 0; vadSilenceFrames = 0;
    if (!valid) {
      setStatus("Bỏ qua tiếng động quá ngắn · VAD đang nghe…", "listening");
      return;
    }
    setStatus("Đã hết câu · đang gửi nhận dạng…", "transcribing");
    void (async () => {
      if (pendingBargeIn) await pendingBargeIn;
      await sendAudioBlob(wavBlob(chunks, audioCtx.sampleRate), "vad-recording.wav");
    })();
  }
}

async function toggleMic() {
  if (vadEnabled) {
    setStatus("VAD đang tự nghe; hãy nói tự nhiên hoặc chuyển sang Push-to-talk.", "listening");
    return;
  }
  if (!pttRecording) {
    if (turnBusy) return;
    await ensureMicCapture();
    pttChunks = [];
    pttRecording = true;
    setText("micBtn", "⏹ Dừng và gửi");
    setStatus("Đang nghe…", "listening");
    return;
  }
  pttRecording = false;
  setText("micBtn", "🎙️ Bắt đầu nói");
  const chunks = pttChunks;
  pttChunks = [];
  stopMicCapture();
  if (!chunks.length) { setStatus("Không thu được âm thanh.", "error"); return; }
  setStatus("Đang gửi nhận dạng…", "transcribing");
  await sendAudioBlob(wavBlob(chunks, audioCtx.sampleRate), "ptt-recording.wav");
}

async function changeMicMode() {
  vadEnabled = $("micMode")?.value === "vad";
  if (pttRecording) { pttRecording = false; pttChunks = []; }
  if (vadEnabled) {
    try {
      await ensureMicCapture();
      setText("micBtn", "🎧 VAD đang bật");
      setStatus("VAD đang chờ giọng nói…", "listening");
    } catch (error) {
      vadEnabled = false;
      $("micMode").value = "ptt";
      setStatus(friendlyError(error), "error");
    }
  } else {
    stopMicCapture();
    setText("micBtn", "🎙️ Bắt đầu nói");
    setStatus("Sẵn sàng.", "idle");
  }
}

async function newConversation() {
  if (currentTurn) await interruptCurrentTurn("conversation_reset");
  const previous = conversationId;
  if (previous) await fetch(`/api/conversations/${encodeURIComponent(previous)}`, { method: "DELETE" });
  conversationId = null;
  assistantNodes = new Map();
  sessionStorage.removeItem("eureka.conversation");
  $("chatLog").innerHTML = '<div class="welcome"><strong>Cuộc trò chuyện mới</strong><span>Lịch sử cũ đã được xóa khỏi phiên chạy.</span></div>';
  await ensureConversation();
  setStatus(vadEnabled ? "VAD đang chờ giọng nói…" : "Sẵn sàng.", vadEnabled ? "listening" : "idle");
}

async function refreshHealth() {
  try {
    const response = await fetch("/health/ready", { cache: "no-store" });
    if (!response.ok) throw new Error(`Health HTTP ${response.status}`);
    const data = await response.json();
    setText("healthJson", JSON.stringify(data, null, 2));
    if (data.v10?.vad) vadConfig = { ...vadConfig, ...data.v10.vad };
    const ok = Boolean(data.ready && data.v10?.tts_realtime_contract_ok);
    const perf = data.v10?.tts_performance_probe;
    setText("healthBadge", ok ? (perf === "degraded" ? "Ready · TTS degraded" : "Realtime ready") : "Service chưa sẵn sàng");
    $("healthBadge").className = `badge ${ok ? "ok" : "bad"}`;
  } catch (error) {
    setText("healthBadge", "Health lỗi");
    $("healthBadge").className = "badge bad";
  }
}

async function refreshKnowledge() {
  try {
    const response = await fetch("/api/knowledge/documents", { cache: "no-store" });
    if (!response.ok) throw new Error(await response.text());
    const docs = await response.json();
    const root = $("knowledgeList");
    root.innerHTML = "";
    if (!Array.isArray(docs) || !docs.length) { root.innerHTML = '<div class="empty-state">Kho tri thức đang trống.</div>'; return; }
    for (const doc of docs) {
      const item = document.createElement("div");
      item.className = "doc";
      const left = document.createElement("div");
      left.innerHTML = `<strong>${escapeHtml(doc.document_id || "")}</strong><br><small>${escapeHtml(doc.filename || "")} · ${Number(doc.chunk_count || 0)} chunks · ${escapeHtml(doc.status || "")}</small>`;
      const del = document.createElement("button");
      del.type = "button"; del.textContent = "Xóa";
      del.onclick = async () => {
        if (!confirm(`Xóa ${doc.document_id}?`)) return;
        del.disabled = true;
        try {
          const result = await fetch(`/api/knowledge/documents/${encodeURIComponent(doc.document_id)}`, { method: "DELETE" });
          if (!result.ok) throw new Error(await result.text());
          await refreshKnowledge();
        } catch (error) { setText("knowledgeStatus", friendlyError(error)); }
        finally { del.disabled = false; }
      };
      item.append(left, del); root.append(item);
    }
  } catch (error) { setText("knowledgeStatus", friendlyError(error)); }
}

async function uploadKnowledge() {
  const file = $("knowledgeFile")?.files?.[0];
  if (!file) { setText("knowledgeStatus", "Hãy chọn tài liệu."); return; }
  $("knowledgeUploadBtn").disabled = true;
  setText("knowledgeStatus", `Đang nạp ${file.name}…`);
  try {
    const form = new FormData();
    form.append("file", file, file.name);
    const id = $("knowledgeId")?.value.trim();
    if (id) form.append("document_id", id);
    const response = await fetch("/api/knowledge/upload", { method: "POST", body: form });
    if (!response.ok) throw new Error(await response.text());
    setText("knowledgeStatus", `Đã nạp ${file.name}.`);
    $("knowledgeFile").value = "";
    await refreshKnowledge();
  } catch (error) { setText("knowledgeStatus", friendlyError(error)); }
  finally { $("knowledgeUploadBtn").disabled = false; }
}

function activateTab(button) {
  const panelId = button?.dataset?.panel;
  if (!panelId) return;
  document.querySelectorAll(".tab").forEach((tab) => {
    const active = tab === button;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === panelId));
  if (panelId === "tab-knowledge") void refreshKnowledge();
  if (panelId === "tab-system") void refreshHealth();
}

function bindClick(id, handler) {
  const element = $(id);
  if (!element) throw new Error(`Missing UI element #${id}`);
  element.addEventListener("click", handler);
}

async function bootstrap() {
  bindClick("sendBtn", () => void sendText());
  bindClick("micBtn", () => void toggleMic().catch((error) => setStatus(friendlyError(error), "error")));
  bindClick("interruptBtn", () => void interruptCurrentTurn("user_cancelled"));
  bindClick("resetConversationBtn", () => void newConversation().catch((error) => setStatus(friendlyError(error), "error")));
  bindClick("knowledgeUploadBtn", () => void uploadKnowledge());
  bindClick("knowledgeRefreshBtn", () => void refreshKnowledge());
  bindClick("healthRefreshBtn", () => void refreshHealth());
  $("micMode").addEventListener("change", () => void changeMicMode());
  $("audioFile").addEventListener("change", () => {
    const file = $("audioFile").files?.[0];
    if (file) void sendAudioBlob(file, file.name);
  });
  $("question").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && event.ctrlKey) void sendText();
  });
  document.querySelectorAll(".tab").forEach((button) => button.addEventListener("click", () => activateTab(button)));
  activateTab(document.querySelector(".tab.active") || document.querySelector(".tab"));
  try { await ensureConversation(); } catch (error) { setStatus(friendlyError(error), "error"); }
  void refreshHealth();
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", () => void bootstrap(), { once: true });
else void bootstrap();
