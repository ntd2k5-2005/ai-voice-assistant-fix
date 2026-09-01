from voice_app.realtime.speech_units import AdaptiveSpeechUnitChunker, SpeechUnitProfile, markdown_to_speech


def test_markdown_to_speech_keeps_content_but_removes_presentation_marks():
    text = "## Nhóm\n- **Khoa Công nghệ:** [HUIT](https://huit.edu.vn)."
    spoken = markdown_to_speech(text)
    assert "##" not in spoken and "**" not in spoken and "https://" not in spoken
    assert "Nhóm" in spoken and "Khoa Công nghệ" in spoken and "HUIT" in spoken


def test_v10_uses_smaller_first_profile_then_larger_steady_profile():
    c = AdaptiveSpeechUnitChunker(
        first=SpeechUnitProfile(20, 35, 55),
        steady=SpeechUnitProfile(35, 70, 100),
    )
    out = c.push("Đây là mệnh đề đầu tiên đủ dài, và đây là phần tiếp theo còn đang sinh thêm nội dung")
    assert out
    assert len(out[0]) <= 55
    assert c.emitted_count >= 1
    assert c.active_profile.max_chars == 100


def test_ordered_list_marker_is_not_spoken_as_own_unit():
    c = AdaptiveSpeechUnitChunker(first=SpeechUnitProfile(10,30,60), steady=SpeechUnitProfile(10,30,60))
    out = c.push("1. Nhóm khoa học tự nhiên và kỹ thuật.")
    assert out == ["Nhóm khoa học tự nhiên và kỹ thuật."]


def test_scheduler_get_exits_after_cancellation_without_sentinel():
    import threading
    import time
    from voice_app.realtime.speech_units import SpeechUnitScheduler

    cancel = threading.Event()
    scheduler = SpeechUnitScheduler(maxsize=1, started_at=time.perf_counter(), cancel_event=cancel)
    result = []

    worker = threading.Thread(target=lambda: result.append(scheduler.get()), daemon=True)
    worker.start()
    time.sleep(0.06)
    cancel.set()
    scheduler.close()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert result == [None]
