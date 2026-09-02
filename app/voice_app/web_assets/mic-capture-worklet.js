class MicCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.enabled = true;
    this.pending = new Float32Array(1024);
    this.offset = 0;
    this.port.onmessage = (event) => {
      const message = event.data || {};
      if (message.type === "enabled") this.enabled = Boolean(message.value);
      if (message.type === "reset") this.offset = 0;
    };
  }

  process(inputs, outputs) {
    const output = outputs[0]?.[0];
    if (output) output.fill(0);
    const input = inputs[0]?.[0];
    if (!this.enabled || !input?.length) return true;

    let cursor = 0;
    while (cursor < input.length) {
      const take = Math.min(input.length - cursor, this.pending.length - this.offset);
      this.pending.set(input.subarray(cursor, cursor + take), this.offset);
      cursor += take;
      this.offset += take;
      if (this.offset === this.pending.length) {
        const frame = this.pending;
        this.pending = new Float32Array(1024);
        this.offset = 0;
        this.port.postMessage({ type: "frame", samples: frame }, [frame.buffer]);
      }
    }
    return true;
  }
}

registerProcessor("eureka-mic-capture", MicCaptureProcessor);
