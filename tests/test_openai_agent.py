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


class DummyOpenAIClient:
    def __init__(self, api_key=None, **kwargs):
        self.responses = DummyResponses()


def test_openai_branch_parses_holdings_and_falls_back(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", "test-key")
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.setattr(agent, "OpenAI", DummyOpenAIClient)

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


def test_gemini_branch_generates_summary(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("AI_PROVIDER", "gemini")

    class DummyGeminiResponse:
        def __init__(self, json_data):
            self._json = json_data

        def raise_for_status(self):
            return None

        def json(self):
            return self._json

    def dummy_post(url, params=None, json=None, timeout=None):
        assert params == {"key": "test-gemini-key"}
        assert url.endswith("text-bison-001:generateText")
        assert json["prompt"]["text"].startswith("You are a market brief assistant")
        assert json["temperature"] == 0.0
        assert json["max_output_tokens"] == 512
        return DummyGeminiResponse({"candidates": [{"output": "Gemini summary output."}]})

    monkeypatch.setattr(agent, "requests", type("R", (), {"post": staticmethod(dummy_post)}))

    output = agent.run_personalized_market_brief_agent(
        client_name="Gemini Client",
        holdings_text="AAPL, 10, 1500, Technology",
        uploaded_file=None,
        risk_profile="Moderate",
        schedule=True,
    )

    assert output["advisor_summary"] == "Gemini summary output."
    assert output["provider_response"] == "Gemini summary output."
    assert any(step["tool"] == "Summary Builder" for step in output["agent_steps"])


def test_huggingface_branch_generates_summary(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_API_TOKEN", "test-hf-key")
    monkeypatch.setenv("AI_PROVIDER", "huggingface")
    monkeypatch.setenv("HUGGINGFACE_MODEL", "gpt2")

    class DummyHFResponse:
        def __init__(self, json_data):
            self._json = json_data
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return self._json

    def dummy_post(url, headers=None, json=None, timeout=None):
        assert url.endswith("gpt2")
        assert headers["Authorization"] == "Bearer test-hf-key"
        assert json["inputs"].startswith("You are a market brief assistant")
        assert json["parameters"]["max_new_tokens"] == 128
        assert json["parameters"]["temperature"] == 0.0
        return DummyHFResponse([{"generated_text": "Hugging Face summary output."}])

    monkeypatch.setattr(agent, "requests", type("R", (), {"post": staticmethod(dummy_post)}))

    output = agent.run_personalized_market_brief_agent(
        client_name="HF Client",
        holdings_text="AAPL, 10, 1500, Technology",
        uploaded_file=None,
        risk_profile="Moderate",
        schedule=False,
    )

    assert output["advisor_summary"] == "Hugging Face summary output."
    assert output["provider_response"] == "Hugging Face summary output."
    assert any(step["tool"] == "Summary Builder" for step in output["agent_steps"])


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
