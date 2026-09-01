# PhoWhisper-base Vietnamese STT API

## Run

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Swagger:
`http://localhost:8000/docs`

Main endpoint:
`POST /transcribe`

Example:

```bash
curl -X POST "http://localhost:8000/transcribe" -F "file=@audio.wav"
```

The API automatically uses CUDA when available, otherwise CPU.

Team benchmark reference:
VIVOS test — WER 8.40%, CER 3.80%, RTF 0.0541 on Colab Tesla T4.


## Integration hardening

This monorepo carries the Colab-proven fixes directly in source:
- model input floating tensors are cast to the model dtype/device;
- integer/bool tensors preserve dtype;
- decoder max token budget reserves target-position prefix space;
- runtime dependencies are pinned to the tested family.
