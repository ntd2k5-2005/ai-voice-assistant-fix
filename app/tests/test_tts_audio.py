import numpy as np
from scipy.io import wavfile

from voice_app.clients.tts import decode_raw_pcm, write_wav


def test_decode_float32_and_write_wav(tmp_path):
    original = np.linspace(-0.5, 0.5, 480, dtype=np.float32)
    decoded = decode_raw_pcm(
        original.astype("<f4").tobytes(),
        dtype="float32",
        channels=1,
    )
    assert decoded.dtype == np.dtype("<f4")
    assert np.allclose(decoded, original)

    path = tmp_path / "a.wav"
    write_wav(path, 48000, decoded)
    sr, loaded = wavfile.read(path)
    assert sr == 48000
    assert loaded.shape == original.shape
