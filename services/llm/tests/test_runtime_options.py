from llm.config import Settings
from llm.service import LLMService, _active_processor


def test_explicit_ollama_placement_is_forwarded_to_every_request():
    service = LLMService.__new__(LLMService)
    service.settings = Settings(
        ollama_placement="cpu",
        ollama_num_gpu=0,
        ollama_num_ctx=4096,
        ollama_num_thread=8,
    )
    options = service._options(None, None, None)
    assert options["num_gpu"] == 0
    assert options["num_ctx"] == 4096
    assert options["num_thread"] == 8


def test_unconfigured_optional_runtime_values_are_not_invented():
    service = LLMService.__new__(LLMService)
    service.settings = Settings()
    options = service._options(None, None, None)
    assert "num_gpu" not in options
    assert "num_ctx" not in options
    assert "num_thread" not in options


def test_active_processor_reports_actual_ollama_memory_placement():
    assert _active_processor(None) is None
    assert _active_processor({"size": 100, "size_vram": 0}) == "cpu"
    assert _active_processor({"size": 100, "size_vram": 40}) == "cpu+gpu"
    assert _active_processor({"size": 100, "size_vram": 100}) == "gpu"
