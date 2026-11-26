from app.prompts import load_prompt


def test_example_prompt_loads() -> None:
    text = load_prompt("chat_system", 1)
    assert "untrusted" in text.lower()
    assert "secret" in text.lower()
