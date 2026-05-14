from mili_market_brief_agent.main import generate_market_brief


def test_generate_market_brief_contains_topic() -> None:
    topic = "AI trends"
    brief = generate_market_brief(topic)

    assert topic in brief
    assert "Market Brief" in brief
