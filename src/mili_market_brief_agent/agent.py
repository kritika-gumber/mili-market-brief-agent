import io
import json
import os
from typing import Any, Dict, List, Optional

import requests

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

from .tools import (
    ClientHolding,
    build_json_output,
    build_personalized_summary,
    fetch_market_data,
    parse_holdings_csv,
    parse_holdings_text,
)

def _ai_provider() -> str:
    return os.getenv("AI_PROVIDER", "openai").lower()


def _ai_model() -> str:
    provider = _ai_provider()
    if provider == "openai":
        return os.getenv("AI_MODEL") or "gpt-4.1-mini"
    if provider == "gemini":
        return os.getenv("AI_MODEL") or "text-bison-001"
    if provider == "huggingface":
        return os.getenv("HUGGINGFACE_MODEL") or os.getenv("AI_MODEL") or "gpt2"
    return os.getenv("AI_MODEL") or "gpt-4.1-mini"


def _normalize_gemini_model(model: str) -> str:
    normalized = model.strip()
    if normalized.startswith("models/"):
        normalized = normalized[len("models/"):]
    if ":" in normalized:
        normalized = normalized.split(":", 1)[0]
    return normalized


def _build_openai_tools() -> List[Dict[str, Any]]:
    return [
        {
            "name": "parse_holdings",
            "description": "Parse client holdings input into structured ticker, quantity, market value, and sector items.",
            "type": "function",
            "parameters": {
                "type": "object",
                "properties": {
                    "raw_holdings": {
                        "type": "string",
                        "description": "The raw holdings text or CSV content.",
                    },
                    "source": {
                        "type": "string",
                        "enum": ["text", "csv"],
                        "description": "The format of the provided holdings.",
                    },
                },
                "required": ["raw_holdings", "source"],
            },
        },
        {
            "name": "fetch_market_data",
            "description": "Fetch mocked market commentary and sector coverage for a list of sectors.",
            "type": "function",
            "parameters": {
                "type": "object",
                "properties": {
                    "sectors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The list of sectors to collect market commentary for.",
                    }
                },
                "required": ["sectors"],
            },
        },
        {
            "name": "build_personalized_summary",
            "description": "Create an advisor-ready personalized market brief using parsed holdings, market data, and risk profile.",
            "type": "function",
            "parameters": {
                "type": "object",
                "properties": {
                    "client_name": {"type": "string"},
                    "holdings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "ticker": {"type": "string"},
                                "quantity": {"type": "number"},
                                "market_value": {"type": "number"},
                                "sector": {"type": "string"},
                            },
                            "required": ["ticker", "quantity", "market_value", "sector"],
                        },
                    },
                    "market_data": {"type": "object"},
                    "risk_profile": {"type": "string"},
                    "schedule": {"type": "boolean"},
                },
                "required": ["client_name", "holdings", "market_data", "risk_profile", "schedule"],
            },
        },
    ]


def _parse_openai_tool_arguments(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return {"raw_arguments": arguments}
    if isinstance(arguments, dict):
        return arguments
    return {}


def _extract_function_calls(response_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    output_items = response_data.get("output") or []
    return [item for item in output_items if isinstance(item, dict) and item.get("type") == "function_call"]


def _run_openai_tool_call(call: Dict[str, Any]) -> Dict[str, Any]:
    name = call.get("name", "")
    arguments = _parse_openai_tool_arguments(call.get("arguments", {}))

    if name == "parse_holdings":
        raw_holdings = arguments.get("raw_holdings", "")
        source = arguments.get("source", "text")
        holdings = (
            parse_holdings_csv(io.StringIO(raw_holdings))
            if source == "csv"
            else parse_holdings_text(raw_holdings)
        )
        return {
            "tool": "Client Holdings Parser",
            "description": "Parsed client holdings from OpenAI tool call.",
            "result": [holding.__dict__ for holding in holdings],
            "result_count": len(holdings),
        }

    if name == "fetch_market_data":
        sectors = arguments.get("sectors", [])
        market_data = fetch_market_data(sectors if isinstance(sectors, list) else [])
        return {
            "tool": "Market Data and News Fetcher",
            "description": "Fetched mock market commentary for relevant sectors.",
            "result": market_data,
            "sectors": sectors,
        }

    if name == "build_personalized_summary":
        client_name = arguments.get("client_name", "")
        holdings_data = arguments.get("holdings", [])
        market_data = arguments.get("market_data", {})
        risk_profile = arguments.get("risk_profile", "")
        schedule = bool(arguments.get("schedule", False))
        holdings = [
            ClientHolding(
                ticker=str(entry.get("ticker", "")).upper(),
                quantity=float(entry.get("quantity", 0) or 0),
                market_value=float(entry.get("market_value", 0) or 0),
                sector=str(entry.get("sector", "Unknown") or "Unknown"),
            )
            for entry in holdings_data
            if isinstance(entry, dict)
        ]
        summary = build_personalized_summary(client_name, holdings, market_data, risk_profile)
        return {
            "tool": "Summary Builder",
            "description": "Built a personalized advisor summary from the tool inputs.",
            "result": summary,
        }

    raise ValueError(f"Unsupported OpenAI tool call: {name}")


def _hf_api_token() -> Optional[str]:
    return os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("AI_API_KEY")


def _ai_api_key() -> Optional[str]:
    return os.getenv("AI_API_KEY") or os.getenv("OPENAI_API_KEY")


def _openai_enabled() -> bool:
    return _ai_provider() == "openai" and OpenAI is not None and bool(_ai_api_key())


def _huggingface_enabled() -> bool:
    return _ai_provider() == "huggingface" and bool(_hf_api_token())


def _run_openai_agent(
    client_name: str,
    holdings_text: str,
    uploaded_file: Optional[io.StringIO],
    risk_profile: str,
    schedule: bool,
) -> Dict[str, object]:
    if _ai_provider() != "openai":
        raise RuntimeError(f"Unsupported AI provider for OpenAI flow: {_ai_provider()}")

    print(f"[Mili Agent] OpenAI enabled: using response tool flow with model {_ai_model()}")
    raw_holdings = ""
    if uploaded_file is not None:
        uploaded_file.seek(0)
        raw_holdings = uploaded_file.read()
    else:
        raw_holdings = holdings_text

    client = OpenAI(api_key=_ai_api_key())
    instructions = (
        "You are a market brief assistant. Use the available tools to parse holdings "
        "and establish market commentary. Prefer calling the provided functions "
        "instead of guessing."
    )

    response = client.responses.create(
        model=_ai_model(),
        input=(
            "Client holdings input and risk profile are provided below. "
            "Use tools when appropriate.\n\n"
            f"Risk profile: {risk_profile}\n"
            f"Schedule delivery: {schedule}\n\n"
            f"Holdings:\n{raw_holdings}"
        ),
        instructions=instructions,
        tools=_build_openai_tools(),
        max_output_tokens=512,
        max_tool_calls=3,
        temperature=0.0,
    )

    response_data = response.to_dict()
    tool_calls = _extract_function_calls(response_data)
    tool_steps: List[Dict[str, Any]] = []
    parsed_holdings: List[ClientHolding] = []
    market_data: Dict[str, Any] = {}
    advisor_summary = ""

    for tool_call in tool_calls:
        try:
            tool_result = _run_openai_tool_call(tool_call)
            tool_steps.append(
                {
                    "tool": tool_result["tool"],
                    "description": tool_result["description"],
                    "result_count": tool_result.get("result_count"),
                    "sectors": tool_result.get("sectors"),
                }
            )
            if tool_result["tool"] == "Client Holdings Parser":
                parsed_holdings = [
                    ClientHolding(
                        ticker=str(entry.get("ticker", "")).upper(),
                        quantity=float(entry.get("quantity", 0) or 0),
                        market_value=float(entry.get("market_value", 0) or 0),
                        sector=str(entry.get("sector", "Unknown") or "Unknown"),
                    )
                    for entry in tool_result["result"]
                    if isinstance(entry, dict)
                ]
            elif tool_result["tool"] == "Market Data and News Fetcher":
                market_data = tool_result["result"]
            elif tool_result["tool"] == "Summary Builder":
                advisor_summary = tool_result["result"]
        except Exception:
            continue

    if not parsed_holdings:
        if uploaded_file is not None:
            uploaded_file.seek(0)
            parsed_holdings = parse_holdings_csv(uploaded_file)
        else:
            parsed_holdings = parse_holdings_text(holdings_text)
        tool_steps.insert(
            0,
            {
                "tool": "Client Holdings Parser",
                "description": "Parsed holdings locally after OpenAI tool path.",
                "result_count": len(parsed_holdings),
            },
        )

    sectors = [holding.sector for holding in parsed_holdings if holding.sector]
    if not market_data:
        market_data = fetch_market_data(sectors)
        tool_steps.append(
            {
                "tool": "Market Data and News Fetcher",
                "description": "Fetched market commentary locally after OpenAI tool path.",
                "sectors": sectors,
            }
        )

    if not advisor_summary:
        advisor_summary = build_personalized_summary(client_name, parsed_holdings, market_data, risk_profile)
        tool_steps.append(
            {
                "tool": "Summary Builder",
                "description": "Built personalized advisor summary locally after OpenAI tool path.",
                "scheduled": schedule,
            }
        )
    else:
        tool_steps.append(
            {
                "tool": "Summary Builder",
                "description": "Used OpenAI tool output summary.",
                "scheduled": schedule,
            }
        )

    output = build_json_output(parsed_holdings, market_data, tool_steps)
    output["advisor_summary"] = advisor_summary
    output["provider_response"] = response_data.get("output_text") or ""
    output["openai_response"] = response_data.get("output_text") or ""
    return output


def _run_gemini_agent(
    client_name: str,
    holdings_text: str,
    uploaded_file: Optional[io.StringIO],
    risk_profile: str,
    schedule: bool,
) -> Dict[str, object]:
    if _ai_provider() != "gemini":
        raise RuntimeError(f"Unsupported AI provider for Gemini flow: {_ai_provider()}")

    api_key = _ai_api_key()
    if not api_key:
        raise RuntimeError("No AI_API_KEY provided for Gemini provider")
    if api_key.strip().lower().startswith("your_") or "your_api_key" in api_key.strip().lower():
        raise RuntimeError(
            "Gemini API key appears to be a placeholder. "
            "Set AI_API_KEY to a real Google Cloud API key for Generative Language."
        )

    gemini_model = _normalize_gemini_model(_ai_model())
    print(f"[Mili Agent] Gemini enabled: using model {gemini_model}")

    if uploaded_file is not None:
        uploaded_file.seek(0)
        raw_holdings = uploaded_file.read()
    else:
        raw_holdings = holdings_text

    holdings = parse_holdings_csv(io.StringIO(raw_holdings)) if uploaded_file is not None else parse_holdings_text(raw_holdings)
    sectors = [holding.sector for holding in holdings if holding.sector]
    market_data = fetch_market_data(sectors)

    prompt = (
        "You are a market brief assistant. Generate a concise, advisor-ready morning brief "
        "for the client using the holdings, risk profile, schedule preference, and market commentary below. "
        "Do not invent holdings. Use the provided information directly.\n\n"
        f"Client name: {client_name}\n"
        f"Risk profile: {risk_profile}\n"
        f"Schedule delivery: {schedule}\n\n"
        "Holdings:\n"
        + "\n".join(
            f"- {holding.ticker}: {holding.quantity} shares, ${holding.market_value:,.0f}, {holding.sector}"
            for holding in holdings
        )
        + "\n\n"
        "Market commentary:\n"
        + "\n".join(f"- {headline}" for headline in market_data["headlines"]) + "\n\n"
        "Write the summary as a clear advisor briefing."
    )

    endpoint = f"https://generativelanguage.googleapis.com/v1beta2/models/{gemini_model}:generateText"
    try:
        response = requests.post(
            endpoint,
            params={"key": api_key},
            json={
                "prompt": {"text": prompt},
                "temperature": 0.0,
                "max_output_tokens": 512,
            },
            timeout=30,
        )
        response.raise_for_status()
    except requests.HTTPError as exc:
        response_text = getattr(response, "text", "")
        if response.status_code == 400:
            raise RuntimeError(
                "Gemini request rejected with HTTP 400. "
                "This usually means the request body or model access is invalid. "
                "Verify the model and API key, and ensure the Generative Language API is enabled. "
                f"Response: {response_text}"
            ) from exc
        if response.status_code == 404:
            raise RuntimeError(
                f"Gemini endpoint not found for model {gemini_model}. "
                "Verify the Generative Language API is enabled for your Google Cloud project "
                "and that the API key has access to the requested model."
            ) from exc
        if response.status_code in {401, 403}:
            raise RuntimeError(
                "Gemini API key authorization failed. Verify the key is valid "
                "and the Generative Language API is enabled in Google Cloud."
            ) from exc
        raise RuntimeError(
            f"Gemini request failed with HTTP {response.status_code}: {response_text}"
        ) from exc
    response_data = response.json()
    provider_text = ""
    if "candidates" in response_data and response_data["candidates"]:
        provider_text = response_data["candidates"][0].get("output", "")

    advisor_summary = provider_text or build_personalized_summary(client_name, holdings, market_data, risk_profile)
    if not provider_text:
        print("[Mili Agent] Gemini response did not include text; using local summary fallback")

    tool_steps = [
        {
            "tool": "Client Holdings Parser",
            "description": "Parsed client holdings for Gemini flow.",
            "result_count": len(holdings),
        },
        {
            "tool": "Market Data and News Fetcher",
            "description": "Gathered mock market commentary for Gemini flow.",
            "sectors": sectors,
        },
        {
            "tool": "Summary Builder",
            "description": "Generated advisor summary via Gemini provider.",
            "scheduled": schedule,
        },
    ]

    output = build_json_output(holdings, market_data, tool_steps)
    output["advisor_summary"] = advisor_summary
    output["provider_response"] = provider_text
    return output


def _run_huggingface_agent(
    client_name: str,
    holdings_text: str,
    uploaded_file: Optional[io.StringIO],
    risk_profile: str,
    schedule: bool,
) -> Dict[str, object]:
    if _ai_provider() != "huggingface":
        raise RuntimeError(f"Unsupported AI provider for Hugging Face flow: {_ai_provider()}")

    api_key = _hf_api_token()
    if not api_key:
        raise RuntimeError(
            "No Hugging Face API token provided. Set HUGGINGFACE_API_TOKEN or AI_API_KEY."
        )

    model_name = _ai_model()
    print(f"[Mili Agent] Hugging Face enabled: using model {model_name}")

    if uploaded_file is not None:
        uploaded_file.seek(0)
        raw_holdings = uploaded_file.read()
    else:
        raw_holdings = holdings_text

    holdings = parse_holdings_csv(io.StringIO(raw_holdings)) if uploaded_file is not None else parse_holdings_text(raw_holdings)
    sectors = [holding.sector for holding in holdings if holding.sector]
    market_data = fetch_market_data(sectors)

    prompt = (
        "You are a market brief assistant. Generate a concise, advisor-ready morning brief "
        "for the client using the holdings, risk profile, schedule preference, and market commentary below. "
        "Do not invent holdings. Use the provided information directly.\n\n"
        f"Client name: {client_name}\n"
        f"Risk profile: {risk_profile}\n"
        f"Schedule delivery: {schedule}\n\n"
        "Holdings:\n"
        + "\n".join(
            f"- {holding.ticker}: {holding.quantity} shares, ${holding.market_value:,.0f}, {holding.sector}"
            for holding in holdings
        )
        + "\n\n"
        "Market commentary:\n"
        + "\n".join(f"- {headline}" for headline in market_data["headlines"]) + "\n\n"
        "Write the summary as a clear advisor briefing."
    )

    endpoint = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        endpoint,
        headers=headers,
        json={
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 128,
                "temperature": 0.0,
                "return_full_text": False,
            },
        },
        timeout=30,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        response_text = response.text if hasattr(response, "text") else ""
        if response.status_code in {401, 403}:
            raise RuntimeError(
                "Hugging Face API token authorization failed. Verify your token and model access."
            ) from exc
        if response.status_code == 404:
            raise RuntimeError(
                f"Hugging Face model not found or inaccessible: {model_name}. "
                "Verify the model name, access rights, and that the token has inference access."
            ) from exc
        raise RuntimeError(
            f"Hugging Face inference failed with HTTP {response.status_code}: {response_text}"
        ) from exc

    response_data = response.json()
    provider_text = ""
    if isinstance(response_data, list) and response_data:
        provider_text = response_data[0].get("generated_text", "")
    elif isinstance(response_data, dict):
        provider_text = response_data.get("generated_text", "")

    advisor_summary = provider_text or build_personalized_summary(client_name, holdings, market_data, risk_profile)
    if not provider_text:
        print("[Mili Agent] Hugging Face response did not include text; using local summary fallback")

    tool_steps = [
        {
            "tool": "Client Holdings Parser",
            "description": "Parsed client holdings for Hugging Face flow.",
            "result_count": len(holdings),
        },
        {
            "tool": "Market Data and News Fetcher",
            "description": "Gathered mock market commentary for Hugging Face flow.",
            "sectors": sectors,
        },
        {
            "tool": "Summary Builder",
            "description": "Generated advisor summary via Hugging Face provider.",
            "scheduled": schedule,
        },
    ]

    output = build_json_output(holdings, market_data, tool_steps)
    output["advisor_summary"] = advisor_summary
    output["provider_response"] = provider_text
    return output


def run_personalized_market_brief_agent(
    client_name: str,
    holdings_text: str,
    uploaded_file: Optional[io.StringIO],
    risk_profile: str,
    schedule: bool = False,
) -> Dict[str, object]:
    """Execute the personalized market brief agent workflow."""
    if _ai_provider() == "openai" and _openai_enabled():
        try:
            return _run_openai_agent(client_name, holdings_text, uploaded_file, risk_profile, schedule)
        except Exception:
            print("[Mili Agent] OpenAI flow failed; falling back to local workflow")
            pass
    elif _ai_provider() == "gemini" and _ai_api_key():
        try:
            return _run_gemini_agent(client_name, holdings_text, uploaded_file, risk_profile, schedule)
        except Exception as exc:
            print(f"[Mili Agent] Gemini flow failed: {exc}; falling back to local workflow")
            pass
    elif _huggingface_enabled():
        try:
            return _run_huggingface_agent(client_name, holdings_text, uploaded_file, risk_profile, schedule)
        except Exception as exc:
            print(f"[Mili Agent] Hugging Face flow failed: {exc}; falling back to local workflow")
            pass
    else:
        if _ai_provider() not in {"openai", "gemini"}:
            print(f"[Mili Agent] AI_PROVIDER={_ai_provider()} is not currently supported; using local fallback workflow")
        else:
            print("[Mili Agent] AI API key not found; using local fallback workflow")

    steps = []
    if uploaded_file is not None:
        uploaded_file.seek(0)
        holdings = parse_holdings_csv(uploaded_file)
        steps.append({
            "tool": "Client Holdings Parser",
            "description": "Parsed client holdings from uploaded CSV file.",
            "result_count": len(holdings),
        })
    else:
        holdings = parse_holdings_text(holdings_text)
        steps.append({
            "tool": "Client Holdings Parser",
            "description": "Parsed client holdings from pasted text.",
            "result_count": len(holdings),
        })

    sectors = [holding.sector for holding in holdings if holding.sector]
    market_data = fetch_market_data(sectors)
    steps.append({
        "tool": "Market Data and News Fetcher",
        "description": "Collected market movers, headlines, and sector coverage relevant to the client holdings.",
        "sectors": sectors,
    })

    summary = build_personalized_summary(client_name, holdings, market_data, risk_profile)
    if schedule:
        steps.append({
            "tool": "Scheduler",
            "description": "Marked the brief for scheduled delivery in the morning digest.",
            "scheduled": True,
        })
    else:
        steps.append({
            "tool": "Scheduler",
            "description": "Generated the brief on demand.",
            "scheduled": False,
        })

    output = build_json_output(holdings, market_data, steps)
    output["advisor_summary"] = summary
    return output
