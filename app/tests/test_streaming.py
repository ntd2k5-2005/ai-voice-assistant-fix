import numpy as np

from voice_app.streaming import PCMPlaybackBuffer, SentenceChunker, pcm_to_wav_bytes


def test_sentence_chunker_incremental_complete_sentences():
    chunker = SentenceChunker(
        min_chars=5,
        soft_chars=80,
        hard_chars=140,
    )
    output = []
    output += chunker.push("Xin chào bạn. Đây là ")
    output += chunker.push("câu thứ hai!")
    output += chunker.flush()
    assert output == ["Xin chào bạn.", "Đây là câu thứ hai!"]


def test_sentence_chunker_does_not_cut_normal_sentence_at_soft_limit():
    chunker = SentenceChunker(min_chars=10, soft_chars=40, hard_chars=140)
    first = (
        "Đây là một câu hoàn chỉnh khá dài nhưng chưa kết thúc, vì vậy hệ thống "
        "không được cắt lưng chừng chỉ vì đã vượt ngưỡng mềm"
    )
    assert len(first) > 40
    assert chunker.push(first) == []
    output = chunker.push(" và chỉ phát khi câu thực sự kết thúc.")
    assert output == [first + " và chỉ phát khi câu thực sự kết thúc."]


def test_sentence_chunker_avoids_common_abbreviation_boundary():
    chunker = SentenceChunker(min_chars=5, soft_chars=100, hard_chars=180)
    # Important incremental case: the first delta may end exactly at "PGS.".
    assert chunker.push("PGS.") == []
    assert chunker.push(" TS.") == []
    output = chunker.push(" Nguyễn Văn A đang trình bày. Câu tiếp theo.")
    assert output == [
        "PGS. TS. Nguyễn Văn A đang trình bày.",
        "Câu tiếp theo.",
    ]


def test_sentence_chunker_long_run_on_prefers_strong_clause_boundary():
    chunker = SentenceChunker(min_chars=20, soft_chars=70, hard_chars=150)
    text = (
        "Phần đầu giải thích đầy đủ ý nghĩa của quy trình và vẫn đang tiếp tục; "
        "phần sau bổ sung thêm thông tin để câu không kéo dài vô hạn"
    )
    output = chunker.push(text)
    assert output
    assert output[0].endswith(";")
    assert "phần sau" not in output[0]


def test_sentence_chunker_hard_guard_only_for_pathological_run_on():
    chunker = SentenceChunker(min_chars=10, soft_chars=30, hard_chars=60)
    text = " ".join(["khôngdấucâu"] * 12)
    output = chunker.push(text)
    assert output
    assert len(output[0]) <= 60


def test_pcm_to_wav_bytes_float32():
    raw = np.linspace(
        -0.5,
        0.5,
        480,
        dtype="<f4",
    ).tobytes()
    wav = pcm_to_wav_bytes(
        raw,
        dtype="float32",
        channels=1,
        sample_rate=48000,
    )
    assert wav[:4] == b"RIFF"
    assert b"WAVE" in wav[:16]


def test_pcm_playback_buffer_coalesces_across_pushes():
    buffer = PCMPlaybackBuffer(
        sample_rate=1000,
        channels=1,
        dtype="float32",
        target_ms=1000,
    )
    frame = b"\x00\x00\x00\x00"
    assert buffer.push(frame * 250) == []
    assert buffer.push(frame * 250) == []
    assert buffer.pending_ms == 500.0
    assert buffer.push(frame * 250) == []
    ready = buffer.push(frame * 250)
    assert len(ready) == 1
    assert buffer.duration_ms(ready[0]) == 1000.0
    assert buffer.pending_ms == 0.0


def test_pcm_playback_buffer_flushes_short_tail_once():
    buffer = PCMPlaybackBuffer(
        sample_rate=1000,
        channels=1,
        dtype="float32",
        target_ms=1250,
    )
    frame = b"\x00\x00\x00\x00"
    assert buffer.push(frame * 500) == []
    tail = buffer.flush()
    assert len(tail) == 1
    assert buffer.duration_ms(tail[0]) == 500.0
    assert buffer.flush() == []
