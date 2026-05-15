import io
import os
from typing import Any
from types import SimpleNamespace
from agents import Agent, Runner, function_tool
from agents.models.openai_responses import OpenAIResponsesModel
from openai import OpenAI
from pydantic import BaseModel

from .tools import (
    ClientHolding,
    build_personalized_summary,
    fetch_market_data,
    parse_holdings_csv,
    parse_holdings_text,
)


class _OpenAIResponsesSyncClientWrapper:
    def __init__(self, sync_client: OpenAI) -> None:
        self._sync_client = sync_client
        self.responses = self

    async def create(self, **kwargs: Any) -> Any:
        if isinstance(kwargs.get("input"), list):
            kwargs["input"] = "\n".join(
                item.get("content", "")
                for item in kwargs["input"]
                if isinstance(item, dict)
            )
        response = self._sync_client.responses.create(**kwargs)
        if not hasattr(response, "usage"):
            response_data = response.to_dict() if hasattr(response, "to_dict") else {}
            output = response_data.get("output", [])
            if isinstance(output, list):
                normalized_output = []
                for item in output:
                    if isinstance(item, dict) and item.get("type") == "function_call":
                        normalized_output.append(
                            {
                                "type": "custom_tool_call",
                                "call_id": item.get("name", "parse_holdings"),
                                "name": item.get("name"),
                                "input": item.get("arguments", ""),
                                "id": item.get("id"),
                            }
                        )
                    else:
                        normalized_output.append(item)
                output = normalized_output
            return SimpleNamespace(
                output=output,
                usage=None,
                id=response_data.get("id"),
                _request_id=getattr(response, "_request_id", None),
            )
        return response

    async def close(self) -> None:
        return None


def _build_openai_agent(sync_client: OpenAI) -> Agent:
    async_client = _OpenAIResponsesSyncClientWrapper(sync_client)
    model = OpenAIResponsesModel(os.getenv("AI_MODEL", "gpt-4.1-mini"), openai_client=async_client)
    return Agent(
        name="Mili Market Brief Agent",
        instructions=(
            "You are a wealth advisor assistant. "
            "Your job is to build a concise and professional personalized market brief. "
            "Use parse_holdings to extract structured holdings from the raw input, then use get_market_data to fetch market context for the sectors in the holdings. "
            "Finally produce a MarketBrief object with advisor_summary, talking_points, and sectors_in_focus. "
            "If the client requested a morning delivery schedule, mention it briefly in the summary."
        ),
        tools=[parse_client_holdings, get_market_data],
        output_type=MarketBrief,
        model=model,
    )


@function_tool(
    name_override="parse_holdings",
    description_override="Parse a client's holdings into structured ticker, quantity, market value, and sector records.",
)
def parse_client_holdings(raw_holdings: str, source: str = "text") -> list[dict]:
    if source == "csv":
        buffer = io.StringIO(raw_holdings)
        holdings = parse_holdings_csv(buffer)
    else:
        holdings = parse_holdings_text(raw_holdings)
    return [h.__dict__ for h in holdings]


@function_tool(
    name_override="get_market_data",
    description_override="Fetch market movers, headlines, and sector coverage for the supplied sectors.",
)
def get_market_data(sectors: list[str]) -> dict:
    return fetch_market_data(sectors)


class MarketBrief(BaseModel):
    advisor_summary: str
    talking_points: list[str]
    sectors_in_focus: list[str]


def _parse_holdings_objects(holdings_text: str, uploaded_file: io.StringIO | None) -> list[ClientHolding]:
    if uploaded_file is not None:
        return parse_holdings_csv(uploaded_file)
    return parse_holdings_text(holdings_text)


def _parse_holdings(holdings_text: str, uploaded_file: io.StringIO | None) -> list[dict]:
    return [h.__dict__ for h in _parse_holdings_objects(holdings_text, uploaded_file)]


def _extract_sectors_from_holdings(holdings: list[dict]) -> list[str]:
    return [sector for sector in {h.get("sector", "Unknown") for h in holdings} if sector and sector != "Unknown"]


def _build_talking_points() -> list[str]:
    return [
        "Confirm whether the client is comfortable with today's sector drivers and any recent performance volatility.",
        "Discuss whether the allocation remains aligned with the client’s time horizon and risk preferences.",
        "Consider whether any rebalancing or risk reduction is needed in high-concentration positions.",
    ]


def _build_agent_steps(
    holdings: list[dict],
    market_data: dict,
    provider: str,
    schedule: bool,
) -> list[dict]:
    sectors = _extract_sectors_from_holdings(holdings)
    return [
        {
            "tool": "Client Holdings Parser",
            "description": "Parsed the client's raw holdings into structured positions.",
            "result_count": len(holdings),
            "sectors": sectors,
            "scheduled": schedule,
        },
        {
            "tool": "Market Data and News Fetcher",
            "description": "Fetched market data and sector headlines for the client's holdings.",
            "result_count": len(market_data.get("top_movers", [])),
            "sectors": sectors,
            "scheduled": schedule,
        },
        {
            "tool": "Summary Builder",
            "description": "Built the advisor-ready summary based on holdings, risk profile, and market context.",
            "result_count": len(_build_talking_points()),
            "sectors": sectors,
            "scheduled": schedule,
        },
    ]


def _validate_openai_api_key(openai_client: OpenAI) -> None:
    openai_client.responses.create(
        model=os.getenv("AI_MODEL", "gpt-4.1-mini"),
        input="OpenAI key validation request",
        max_output_tokens=16,
    )


def _run_local_workflow(
    client_name: str,
    holdings_text: str,
    uploaded_file: io.StringIO | None,
    risk_profile: str,
    schedule: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    holding_objects = _parse_holdings_objects(holdings_text, uploaded_file)
    holdings = [h.__dict__ for h in holding_objects]
    sectors = _extract_sectors_from_holdings(holdings)
    market_data = fetch_market_data(sectors)
    summary = build_personalized_summary(client_name, holding_objects, market_data, risk_profile)
    output = {
        "advisor_summary": summary,
        "talking_points": _build_talking_points(),
        "sectors_in_focus": sectors,
        "holdings": holdings,
        "market_data": market_data,
        "agent_steps": _build_agent_steps(holdings, market_data, provider="local", schedule=schedule),
        "provider_response": "",
    }
    if extra:
        output.update(extra)
    return output


def run_personalized_market_brief_agent(
    client_name: str,
    holdings_text: str,
    uploaded_file: io.StringIO | None,
    risk_profile: str,
    schedule: bool,
) -> dict[str, Any]:
    openai_api_key = os.getenv("AI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return _run_local_workflow(client_name, holdings_text, uploaded_file, risk_profile, schedule)

    openai_client = OpenAI(api_key=openai_api_key)
    try:
        _validate_openai_api_key(openai_client)
    except Exception as exc:
        return _run_local_workflow(
            client_name,
            holdings_text,
            uploaded_file,
            risk_profile,
            schedule,
            extra={"openai_key_error": str(exc), "openai_response": ""},
        )

    prompt = (
        f"Client: {client_name}\nRisk: {risk_profile}\nSchedule: {schedule}\n"
        f"Holdings:\n{holdings_text}"
    )
    try:
        agent = _build_openai_agent(openai_client)
        result = Runner.run_sync(agent, input=prompt, max_turns=20)
        brief = result.final_output_as(MarketBrief, raise_if_incorrect_type=False)
        if isinstance(brief, MarketBrief):
            advisor_summary = brief.advisor_summary
            talking_points = brief.talking_points
            sectors = brief.sectors_in_focus
        else:
            advisor_summary = str(result.final_output)
            talking_points = _build_talking_points()
            sectors = []
    except Exception as exc:
        return _run_local_workflow(
            client_name,
            holdings_text,
            uploaded_file,
            risk_profile,
            schedule,
            extra={"openai_key_error": str(exc), "openai_response": ""},
        )

    holdings = _parse_holdings(holdings_text, uploaded_file)
    market_data = fetch_market_data(sectors or _extract_sectors_from_holdings(holdings))
    return {
        "advisor_summary": advisor_summary,
        "talking_points": talking_points,
        "sectors_in_focus": sectors,
        "holdings": holdings,
        "market_data": market_data,
        "agent_steps": _build_agent_steps(holdings, market_data, provider="openai", schedule=schedule),
        "provider_response": "",
        "openai_response": "",
    }
