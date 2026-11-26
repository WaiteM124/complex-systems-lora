from app.model_config import get_model, load_models


def test_models_yaml_loads_per_environment() -> None:
    config = load_models()
    assert "local" in config.environments
    assert "staging" in config.environments
    assert "production" in config.environments
    chat = get_model("chat", "local")
    assert chat.id
    assert chat.max_tokens > 0
    assert chat.temperature >= 0
