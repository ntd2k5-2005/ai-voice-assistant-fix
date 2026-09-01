def test_liveness(client):
    r = client.get("/health/live")
    assert r.status_code == 200
    assert r.json()["status"] == "alive"


def test_invalid_chat_is_422(client):
    r = client.post("/v1/chat", json={"messages": [], "max_tokens": 100})
    assert r.status_code == 422


def test_missing_generate_prompt_is_422(client):
    r = client.post(
        "/v1/generate",
        json={"temperature": 0.7, "max_tokens": 100},
    )
    assert r.status_code == 422


def test_invalid_generate_temperature_is_422(client):
    r = client.post(
        "/v1/generate",
        json={"prompt": "x", "temperature": 3.0, "max_tokens": 100},
    )
    assert r.status_code == 422


def test_readiness_contract_exists(client):
    # 200 when Ollama/model is available, otherwise clean 503.
    r = client.get("/health/ready")
    assert r.status_code in {200, 503}
