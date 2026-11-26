from app.model_config import get_base_model, get_model, get_region, load_models


def test_models_yaml_loads_generator_and_judge() -> None:
    config = load_models()
    assert config.base_model.startswith("Qwen/")
    assert get_region() == "eu-west-2"
    assert get_base_model() == config.base_model
    assert "local" in config.environments
    assert "staging" in config.environments
    assert "production" in config.environments
    generator = get_model("generator", "local")
    assert generator.id == "qwen.qwen3-235b-a22b-2507-v1:0"
    assert generator.max_tokens > 0
    assert generator.temperature >= 0
    judge = get_model("judge", "local")
    assert judge.id == "deepseek.v3.2"
