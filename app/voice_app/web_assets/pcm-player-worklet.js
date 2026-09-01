class FloatRingBuffer {
  constructor(capacity) {
    this.capacity = Math.max(1024, capacity | 0);
    this.data = new Float32Array(this.capacity);
    this.readIndex = 0;
    this.writeIndex = 0;
    this.size = 0;
  }

  clear() {
    this.readIndex = 0;
    this.writeIndex = 0;
    this.size = 0;
  }

  _grow(required) {
    if (required <= this.capacity) return;
    let nextCapacity = this.capacity;
    while (nextCapacity < required) nextCapacity *= 2;
    const next = new Float32Array(nextCapacity);
    const first = Math.min(this.size, this.capacity - this.readIndex);
    next.set(this.data.subarray(this.readIndex, this.readIndex + first), 0);
    if (first < this.size) next.set(this.data.subarray(0, this.size - first), first);
    this.data = next;
    this.capacity = nextCapacity;
    this.readIndex = 0;
    this.writeIndex = this.size;
  }

  push(samples) {
    if (!samples || !samples.length) return;
    this._grow(this.size + samples.length + 1);
    const first = Math.min(samples.length, this.capacity - this.writeIndex);
    this.data.set(samples.subarray(0, first), this.writeIndex);
    if (first < samples.length) this.data.set(samples.subarray(first), 0);
    this.writeIndex = (this.writeIndex + samples.length) % this.capacity;
    this.size += samples.length;
  }

  popInto(out) {
    const take = Math.min(out.length, this.size);
    if (!take) return 0;
    const first = Math.min(take, this.capacity - this.readIndex);
    out.set(this.data.subarray(this.readIndex, this.readIndex + first), 0);
    if (first < take) out.set(this.data.subarray(0, take - first), first);
    this.readIndex = (this.readIndex + take) % this.capacity;
    this.size -= take;
    return take;
  }
}

class PCMPlayerV10 extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ring = new FloatRingBuffer(Math.round(sampleRate * 30));
    this.prebufferFrames = Math.round(sampleRate * 0.7);
    this.rebufferFrames = Math.round(sampleRate * 0.9);
    this.startedOnce = false;
    this.playing = false;
    this.ended = false;
    this.underrunCount = 0;
    this.underrunFrames = 0;
    this.currentUnderrunFrames = 0;
    this.maxUnderrunFrames = 0;
    this.rebufferCount = 0;
    this.callbackCount = 0;
    this.drainedSent = false;
    this.chunkSpans = [];
    this.cumulativeAcceptedFrames = 0;
    this.cumulativeConsumedFrames = 0;

    this.port.onmessage = (event) => {
      const msg = event.data || {};
      if (msg.type === "config") {
        this.prebufferFrames = Math.max(1, Math.round(sampleRate * (Number(msg.prebufferMs || 700) / 1000)));
        this.rebufferFrames = Math.max(1, Math.round(sampleRate * (Number(msg.rebufferMs || 900) / 1000)));
      } else if (msg.type === "pcm") {
        const bytes = new Uint8Array(msg.buffer || new ArrayBuffer(0));
        const aligned = bytes.byteLength - (bytes.byteLength % 2);
        if (!aligned) return;
        const view = new DataView(bytes.buffer, bytes.byteOffset, aligned);
        const samples = new Float32Array(aligned / 2);
        for (let i = 0; i < samples.length; i++) samples[i] = view.getInt16(i * 2, true) / 32768;
        this.ring.push(samples);
        const sequence = Number(msg.sequence || 0);
        this.chunkSpans.push({sequence, remaining: samples.length, started: false});
        this.cumulativeAcceptedFrames += samples.length;
        this.port.postMessage({
          type: "accepted",
          sequence,
          acceptedFrames: samples.length,
          bufferedFrames: this.ring.size,
          cumulativeAcceptedFrames: this.cumulativeAcceptedFrames,
        });
      } else if (msg.type === "end") {
        this.ended = true;
      } else if (msg.type === "reset") {
        this.ring.clear();
        this.startedOnce = false;
        this.playing = false;
        this.ended = false;
        this.underrunCount = 0;
        this.underrunFrames = 0;
        this.currentUnderrunFrames = 0;
        this.maxUnderrunFrames = 0;
        this.rebufferCount = 0;
        this.drainedSent = false;
        this.chunkSpans = [];
        this.cumulativeAcceptedFrames = 0;
        this.cumulativeConsumedFrames = 0;
      }
    };
  }

  _emitMetrics(force = false) {
    this.callbackCount += 1;
    if (!force && this.callbackCount % 8 !== 0) return;
    this.port.postMessage({
      type: "metrics",
      bufferedFrames: this.ring.size,
      underrunCount: this.underrunCount,
      underrunFrames: this.underrunFrames,
      maxUnderrunFrames: this.maxUnderrunFrames,
      rebufferCount: this.rebufferCount,
      playing: this.playing,
      startedOnce: this.startedOnce,
      ended: this.ended,
      cumulativeAcceptedFrames: this.cumulativeAcceptedFrames,
      cumulativeConsumedFrames: this.cumulativeConsumedFrames,
    });
  }

  _consumeSpans(frameCount) {
    let remaining = frameCount;
    while (remaining > 0 && this.chunkSpans.length) {
      const span = this.chunkSpans[0];
      if (!span.started) {
        span.started = true;
        this.port.postMessage({
          type: "consuming",
          sequence: span.sequence,
          bufferedFrames: this.ring.size,
          cumulativeConsumedFrames: this.cumulativeConsumedFrames,
        });
      }
      const take = Math.min(remaining, span.remaining);
      span.remaining -= take;
      remaining -= take;
      this.cumulativeConsumedFrames += take;
      if (span.remaining === 0) this.chunkSpans.shift();
    }
  }

  process(_inputs, outputs) {
    const out = outputs[0][0];
    if (!out) return true;
    out.fill(0);

    if (!this.playing) {
      // If the network stream ended while we were rebuffering and no PCM
      // remains, report drained immediately instead of waiting forever for a
      // threshold that can never be reached.
      if (this.ended && this.ring.size === 0) {
        if (!this.drainedSent) {
          this.drainedSent = true;
          this.port.postMessage({type: "drained"});
        }
        this._emitMetrics();
        return true;
      }

      const threshold = this.startedOnce ? this.rebufferFrames : this.prebufferFrames;
      if (this.ring.size >= threshold || (this.ended && this.ring.size > 0)) {
        this.playing = true;
        if (!this.startedOnce) {
          this.startedOnce = true;
          this.port.postMessage({type: "started", bufferedFrames: this.ring.size});
        } else {
          this.port.postMessage({type: "resumed", bufferedFrames: this.ring.size});
        }
        this.currentUnderrunFrames = 0;
      } else {
        // While rebuffering after playback has already started, every silent
        // render quantum is part of the same audible underrun gap. Count it so
        // browser metrics remain comparable with what the user actually hears.
        if (this.startedOnce && !this.ended) {
          this.currentUnderrunFrames += out.length;
          this.underrunFrames += out.length;
          this.maxUnderrunFrames = Math.max(this.maxUnderrunFrames, this.currentUnderrunFrames);
        }
        this._emitMetrics();
        return true;
      }
    }

    const written = this.ring.popInto(out);
    this._consumeSpans(written);
    const missing = out.length - written;
    if (missing > 0 && !this.ended) {
      if (this.currentUnderrunFrames === 0) {
        this.underrunCount += 1;
        this.rebufferCount += 1;
      }
      this.currentUnderrunFrames += missing;
      this.underrunFrames += missing;
      this.maxUnderrunFrames = Math.max(this.maxUnderrunFrames, this.currentUnderrunFrames);
      // V10 stops consuming after a real underrun and waits for a fresh cushion.
      this.playing = false;
      this.port.postMessage({
        type: "rebuffering",
        bufferedFrames: this.ring.size,
        underrunCount: this.underrunCount,
        rebufferCount: this.rebufferCount,
      });
    } else if (missing === 0) {
      this.currentUnderrunFrames = 0;
    }

    if (this.ended && this.ring.size === 0 && written === 0 && !this.drainedSent) {
      this.drainedSent = true;
      this.port.postMessage({type: "drained"});
    }
    this._emitMetrics(missing > 0);
    return true;
  }
}

registerProcessor("pcm-player-v10", PCMPlayerV10);
