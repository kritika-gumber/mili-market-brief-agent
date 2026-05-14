import io

from mili_market_brief_agent.agent import run_personalized_market_brief_agent
from mili_market_brief_agent.tools import parse_holdings_text


def test_parse_holdings_text():
    raw_text = "AAPL, 10, 1500, Technology\nXOM, 5, 450, Energy"
    holdings = parse_holdings_text(raw_text)

    assert len(holdings) == 2
    assert holdings[0].ticker == "AAPL"
    assert holdings[0].sector == "Technology"


def test_agent_generates_summary():
    raw_text = "AAPL, 10, 1500, Technology\nXOM, 5, 450, Energy"
    output = run_personalized_market_brief_agent(
        client_name="Test Client",
        holdings_text=raw_text,
        uploaded_file=None,
        risk_profile="Moderate",
        schedule=False,
    )

    assert "advisor_summary" in output
    assert "Test Client" in output["advisor_summary"]
    assert len(output["holdings"]) == 2
    assert output["agent_steps"][0]["tool"] == "Client Holdings Parser"
