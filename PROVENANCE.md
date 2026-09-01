# Source provenance

This monorepo is an integration snapshot. The module baselines were copied
from the following repositories/revisions before source-level integration fixes:

| Module | Upstream | Baseline commit |
|---|---|---|
| STT | https://github.com/YuluWusu/Speech-to-Text-Long | `bab59e7289e8ab0a82f8e33b92dc8d9704c6b9f4` |
| RAG | https://github.com/ntd2k5-2005/rag-service | `0492dca4b493644aab715c19debf97762839638d` |
| LLM | https://github.com/tsuk1shima-cs/LLM-Model | `dafecbec16b5c2b5b025e5f65fca6b115d352ffe` |
| TTS | https://github.com/PhatTran-71/API-TTS | `8fff60262a05d5fb462def6115de8324daca2b37` |
| App/FE | https://github.com/ntd2k5-2005/fe-voice-chatbot | `d13a13b372ad19f1ef61f1de7ef6df49fc879eb8` |

Keep fixes in module folders. Use Git branches for changes (`fix/stt-runtime`,
`fix/llm-ollama`, ...), not one permanent branch per module.

## VieNeu-TTS runtime V9

Realtime TTS environment is reproduced from `pnnbao97/VieNeu-TTS` commit `36c4b501b0634a8f59805e6b529a058fbd30190b` using its `uv.lock` via `uv sync --frozen`.


## V10 integration note

V10 changes are local integration/media orchestration changes. STT, RAG and LLM upstream provenance remains unchanged. VieNeu realtime runtime remains pinned by `services/tts/runtime.json`.
