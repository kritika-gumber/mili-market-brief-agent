import io
import os

import pytest

from mili_market_brief_agent import agent


class DummyResponse:
    def __init__(self, data):
        self._data = data

    def to_dict(self):
        return self._data


class DummyResponses:
    def create(self, **kwargs):
        input_text = kwargs.get("input", "")
        if input_text.startswith("OpenAI key validation request"):
            return DummyResponse({"output": [{"type": "output_text", "text": "OK"}]})
        return DummyResponse(
            {
                "output": [
                    {
                        "type": "function_call",
                        "name": "parse_holdings",
                        "arguments": "{\"raw_holdings\": \"AAPL, 10, 1500, Technology\", \"source\": \"text\"}",
                    }
                ]
            }
        )


class DummyAsyncOpenAIClient:
    def __init__(self, api_key=None, **kwargs):
        self.responses = DummyResponses()


def test_openai_branch_parses_holdings_and_falls_back(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", "test-key")
    monkeypatch.setattr(agent, "AsyncOpenAI", DummyAsyncOpenAIClient)

    output = agent.run_personalized_market_brief_agent(
        client_name="OpenAI Client",
        holdings_text="AAPL, 10, 1500, Technology",
        uploaded_file=None,
        risk_profile="Moderate",
        schedule=False,
    )

    assert output["advisor_summary"]
    assert any(step["tool"] == "Client Holdings Parser" for step in output["agent_steps"])
    assert any(step["tool"] == "Market Data and News Fetcher" for step in output["agent_steps"])
    assert any(step["tool"] == "Summary Builder" for step in output["agent_steps"])
    assert output["holdings"][0]["ticker"] == "AAPL"
    assert output["openai_response"] == ""


def test_openai_disabled_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    output = agent.run_personalized_market_brief_agent(
        client_name="Local Client",
        holdings_text="AAPL, 5, 750, Technology",
        uploaded_file=None,
        risk_profile="Moderate",
        schedule=False,
    )

    assert output["advisor_summary"]
    assert output["holdings"][0]["ticker"] == "AAPL"
    assert "openai_response" not in output


def test_openai_validation_falls_back(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", "invalid-key")

    class BadAsyncOpenAIClient:
        def __init__(self, api_key=None, **kwargs):
            raise RuntimeError("Invalid API key")

    monkeypatch.setattr(agent, "AsyncOpenAI", BadAsyncOpenAIClient)

    output = agent.run_personalized_market_brief_agent(
        client_name="OpenAI Client",
        holdings_text="AAPL, 10, 1500, Technology",
        uploaded_file=None,
        risk_profile="Moderate",
        schedule=False,
    )

    assert output["advisor_summary"]
    assert output["openai_key_error"] == "Invalid API key"
    assert output["holdings"][0]["ticker"] == "AAPL"
