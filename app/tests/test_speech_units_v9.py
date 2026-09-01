from voice_app.realtime.speech_units import SpeechUnitChunker, markdown_to_speech


def test_markdown_to_speech_keeps_content_but_removes_presentation_marks():
    text = "## Nhóm\n- **Khoa Công nghệ:** [HUIT](https://huit.edu.vn)."
    spoken = markdown_to_speech(text)
    assert "##" not in spoken and "**" not in spoken and "https://" not in spoken
    assert "Nhóm" in spoken and "Khoa Công nghệ" in spoken and "HUIT" in spoken


def test_long_sentence_emits_clause_before_full_sentence_is_finished():
    c = SpeechUnitChunker(min_chars=30, target_chars=55, max_chars=90)
    out = c.push("Trường Đại học Công Thương Thành phố Hồ Chí Minh, là một cơ sở đào tạo")
    assert out
    assert out[0].endswith(",")
    assert "cơ sở đào tạo" not in out[0]


def test_short_complete_sentence_emits_immediately():
    c = SpeechUnitChunker(min_chars=40, target_chars=70, max_chars=100)
    assert c.push("Xin chào.") == ["Xin chào."]


def test_ordered_list_marker_is_not_emitted_as_its_own_unit():
    c = SpeechUnitChunker(min_chars=10, target_chars=30, max_chars=60)
    out = c.push("1. Nhóm khoa học tự nhiên và kỹ thuật.")
    assert out == ["Nhóm khoa học tự nhiên và kỹ thuật."]
